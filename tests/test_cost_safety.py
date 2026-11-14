"""Phase 5 cost safety: failure shutdown, startup timeout, alerts, schema drift.

Fake provider, fake monitor, simulated monotonic clock; no real services.
"""

import json
import os
from dataclasses import replace
from pathlib import Path
from threading import Event

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

from simplegen import cost_safety, engine
from simplegen import global_session as gs
from simplegen.global_session import GlobalSession
from simplegen.settings import Settings

OUTPUT = {"id": "o1", "prompt_id": "p1", "filename": "out.mp4", "subfolder": "", "type": "output"}


class Provider:
    def __init__(self, state, *, stuck=False):
        self.state, self.stuck = state, stuck
        self.present, self.deletes = True, []
        self.client = type("Client", (), {"close": lambda self: None})()

    def get(self, pod_id):
        return {"id": pod_id, "name": "sg-owned-gpu", "desiredStatus": "RUNNING"} if self.present else None

    def delete(self, pod_id):
        # Durable intent must exist before the paid-resource call.
        assert self.state["managed_session"]["delete_intent"] == "failure_idle"
        self.deletes.append(pod_id)
        if not self.stuck:
            self.present = False

    def deploy(self, *args, **kwargs):
        pytest.fail("no create request")

    def reconcile_exact_name(self, *args):
        pytest.fail("no create reconciliation")


class Monitor:
    def __init__(self):
        self.observation = {"ready": True, "healthy": True, "auth_verified": True, "queue_empty": True,
                            "jobs": {"p1": "success", "p2": "error"}, "outputs": {"p1": [OUTPUT]},
                            "unregistered_jobs": False}
        self.client = type("Client", (), {"close": lambda self: None})()

    def poll(self, _jobs):
        return json.loads(json.dumps(self.observation))


@pytest.fixture
def clock(monkeypatch, tmp_path):
    monkeypatch.setenv("SIMPLEGEN_HOME", str(tmp_path / "home"))
    now = [1000.0]
    monkeypatch.setattr(gs.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(engine, "probe_video", lambda path: {})
    return now


def member(folder, run_id, prompt_id, downloaded):
    folder.mkdir(parents=True)
    state = {"run_id": run_id, "prompt_id": prompt_id, "downloads": {}, "download_files": {}}
    if downloaded:
        (folder / "out.mp4").write_bytes(b"video")
        state["downloads"]["k"] = {"path": "out.mp4", "size_bytes": 5, "verification": "size"}
        state["download_files"]["k"] = {"filename": "out.mp4", "subfolder": "", "type": "output"}
    (folder / "state.json").write_text(json.dumps(state))


def batch_owner(tmp_path, *, downloaded=True, recovered=True, kind="batch", **settings):
    member(tmp_path / "m1", "m1", "p1", downloaded)
    member(tmp_path / "m2", "m2", "p2", False)
    saved = {"backend": "global_desktop", "sid": "s", "pod_id": "pod", "name": "sg-owned-gpu",
             "gpu_id": "NVIDIA RTX PRO 6000", "actual_hourly": 1.89,
             "session_started_at": "2026-09-28T00:00:00+00:00",
             "jobs": {"m1": {"prompt_id": "p1", "status": "running", "outputs": []},
                      "m2": {"prompt_id": "p2", "status": "running", "outputs": []}}}
    state = {"id": "b" * 32, "members": [{"folder": str(tmp_path / "m1")}, {"folder": str(tmp_path / "m2")}],
             "managed_session": saved}
    config = Settings(cloud=True, managed_runpod=True, global_desktop=True, **settings)
    cost_safety.record_owner_failure(state, config, "ComfyUI execution failed on m2", kind)
    if recovered:
        saved["recovery_completed_at"] = "2026-09-28T01:00:00+00:00"
    return state, config


def poller(tmp_path, state, config, provider, monitor):
    def poll():
        session = GlobalSession(config, tmp_path, state, lambda: None, lambda _: None, Event(), Event())
        session._provider, session._monitor = provider, monitor
        return session.status()
    return poll


def run_for(clock, poll, seconds, step=10):
    for _ in range(int(seconds // step)):
        clock[0] += step
        poll()


def events(state):
    return [entry["event"] for entry in state["managed_session"].get("alert_log", [])]


# 1. Batch failure → one failure_idle DELETE after the grace period


def test_batch_failure_retires_after_grace_period(tmp_path, clock):
    state, config = batch_owner(tmp_path)
    provider, monitor = Provider(state), Monitor()
    poll = poller(tmp_path, state, config, provider, monitor)
    assert poll()["failure_remaining_seconds"] == 900
    run_for(clock, poll, 890)
    assert provider.deletes == [] and state["managed_session"]["failure_remaining_seconds"] == 10
    run_for(clock, poll, 10)
    saved = state["managed_session"]
    assert provider.deletes == ["pod"]  # Exactly one DELETE.
    assert saved["retired"] is True and saved["billing_confirmed_stopped"] is True
    assert saved["delete_intent"] == "failure_idle"
    summary = saved["failure_summary"]
    assert summary["failed_jobs"] == ["m2"] and summary["succeeded_jobs"] == ["m1"]
    assert summary["lost_outputs"] == [] and summary["collected_outputs"] == 1
    assert events(state) == ["owner_failure", "failure_countdown", "failure_retired"]
    poll()
    assert provider.deletes == ["pod"]


# 2. Undownloaded successful output → never deletes


def test_uncollected_output_never_deletes_and_alerts(tmp_path, clock):
    state, config = batch_owner(tmp_path, downloaded=False)
    provider = Provider(state)
    poll = poller(tmp_path, state, config, provider, Monitor())
    poll()
    run_for(clock, poll, 3600)
    saved = state["managed_session"]
    assert provider.deletes == [] and not saved.get("retired")
    assert saved["attention"] == "outputs_not_collected"
    assert saved["uncollected_outputs"] == [{"prompt_id": "p1", "filename": "out.mp4"}]
    assert events(state).count("outputs_not_collected") == 1
    # A partially written file is not collected either.
    (tmp_path / "m1" / "out.mp4").write_bytes(b"vid")
    run_for(clock, poll, 1200)
    assert provider.deletes == []


# 3. App stopped mid-recovery → no delete


def test_batch_without_recovery_marker_never_deletes(tmp_path, clock):
    state, config = batch_owner(tmp_path, recovered=False)
    provider = Provider(state)
    poll = poller(tmp_path, state, config, provider, Monitor())
    poll()
    run_for(clock, poll, 3600)
    assert provider.deletes == [] and state["managed_session"]["failure_remaining_seconds"] is None


# 4. Countdown resets


@pytest.mark.parametrize("disruption", ["activity", "gap", "process", "unhealthy", "unauthenticated", "running"])
def test_countdown_resets(tmp_path, clock, monkeypatch, disruption):
    state, config = batch_owner(tmp_path)
    provider, monitor = Provider(state), Monitor()
    poll = poller(tmp_path, state, config, provider, monitor)
    poll()
    run_for(clock, poll, 600)
    assert state["managed_session"]["failure_remaining_seconds"] == 300
    if disruption == "gap":
        clock[0] += gs.POLL_GAP_SECONDS + 1
        poll()
    elif disruption == "process":
        monkeypatch.setattr(gs, "PROCESS_ID", "another-desktop-process")
        clock[0] += 10
        poll()
    else:
        original = dict(monitor.observation)
        if disruption == "activity":
            monitor.observation["queue_empty"] = False
        elif disruption == "unhealthy":
            monitor.observation["healthy"] = False
        elif disruption == "unauthenticated":
            monitor.observation["auth_verified"] = False
        else:
            monitor.observation["jobs"] = {"p1": "success", "p2": "running"}
        clock[0] += 10
        poll()
        monitor.observation = original
        clock[0] += 10
        poll()
    assert state["managed_session"]["failure_remaining_seconds"] == 900
    run_for(clock, poll, 890)
    assert provider.deletes == []
    run_for(clock, poll, 10)
    assert provider.deletes == ["pod"]


# 5. Unregistered jobs


def test_unregistered_jobs_never_auto_delete(tmp_path, clock):
    state, config = batch_owner(tmp_path)
    provider, monitor = Provider(state), Monitor()
    monitor.observation["unregistered_jobs"] = True
    poll = poller(tmp_path, state, config, provider, monitor)
    poll()
    monitor.observation["unregistered_jobs"] = False
    run_for(clock, poll, 3600)
    assert provider.deletes == [] and state["managed_session"]["unregistered_jobs_seen"] is True


# 6. Detach and cancel are not owner failures (one-off run path)


class FakeManaged:
    retired = False

    def __init__(self, interrupt=None):
        self.shutdowns, self.interrupt = 0, interrupt

    def connect(self, *args):
        if self.interrupt is not None:
            self.interrupt.set()  # The user cancels or detaches while the provider call fails.
        raise RuntimeError("provider exploded")

    def shutdown(self):
        self.shutdowns += 1

    def close(self):
        pass


@pytest.mark.parametrize("mode", ["fail", "cancel", "detach"])
def test_only_real_failures_record_owner_failure(tmp_path, monkeypatch, mode):
    monkeypatch.setenv("SIMPLEGEN_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(engine.agents, "executable", lambda name: f"/usr/bin/{name}")
    cancel, detach = Event(), Event()
    managed = FakeManaged({"fail": None, "cancel": cancel, "detach": detach}[mode])
    monkeypatch.setattr(gs, "session_for", lambda *args: managed)
    folder = tmp_path / "run"
    folder.mkdir()
    settings = Settings(cloud=True, managed_runpod=True, global_desktop=True, output_folder=str(tmp_path))
    state = {"schema_version": 4, "run_id": "r", "client_id": "c", "settings": settings.public_dict(),
             "concept": "A meadow", "status": "generating", "downloads": {}, "prompt_id": "job-1",
             "managed_session": {"backend": "global_desktop", "sid": "s", "name": "sg-owned-gpu", "pod_id": "pod"}}
    with pytest.raises(RuntimeError):
        engine._run_locked(settings, "A meadow", folder, state, lambda _: None, cancel, detach)
    saved = json.loads((folder / "state.json").read_text())["managed_session"]
    if mode == "fail":
        assert saved["owner_failure"]["kind"] == "run" and "provider exploded" in saved["owner_failure"]["reason"]
    else:
        assert "owner_failure" not in saved
    assert managed.shutdowns == (1 if mode == "cancel" else 0)  # Cancel behavior unchanged.


# 7. Resume clears the failure and stops the countdown


def test_run_resume_clears_failure_before_new_work(tmp_path, clock, monkeypatch):
    state, config = batch_owner(tmp_path)
    state.pop("members")
    state.update(schema_version=4, run_id="r", concept="A meadow", settings=config.public_dict())
    folder = tmp_path / "run"
    folder.mkdir()
    state["managed_session"]["jobs"] = {}
    (folder / "state.json").write_text(json.dumps(state))
    seen = []
    monkeypatch.setattr(engine, "_run_locked", lambda settings, concept, f, s, *a, **k: seen.append(
        json.loads(json.dumps(s["managed_session"]))) or {})
    engine.run(config, "", lambda _: None, Event(), resume_folder=folder)
    assert "owner_failure" not in seen[0] and "failure_idle_since" not in seen[0]
    saved = json.loads((folder / "state.json").read_text())
    provider = Provider(saved)
    poll = poller(folder, saved, config, provider, Monitor())
    poll()
    run_for(clock, poll, 3600)
    assert provider.deletes == [] and saved["managed_session"]["failure_remaining_seconds"] is None


# 8. Startup timeout


def test_startup_failure_without_jobs_retires_after_grace(tmp_path, clock):
    state, config = batch_owner(tmp_path, kind="startup")
    state["managed_session"]["jobs"] = {}

    class NoMonitor(Monitor):
        def poll(self, _jobs):
            pytest.fail("a pod that never became ready needs no queue observation")
    provider = Provider(state)
    poll = poller(tmp_path, state, config, provider, NoMonitor())
    assert poll()["phase"] == "startup_failed"
    run_for(clock, poll, 890)
    assert provider.deletes == []
    run_for(clock, poll, 10)
    assert provider.deletes == ["pod"] and state["managed_session"]["retired"] is True
    assert "startup_timeout" in events(state)


def test_startup_failure_with_pending_submission_never_deletes(tmp_path, clock):
    state, config = batch_owner(tmp_path, kind="startup")
    state["managed_session"]["jobs"] = {"m1": {"prompt_id": None, "status": "submission_pending", "outputs": []}}
    monitor = Monitor()
    monitor.observation.update(ready=False, healthy=False, queue_empty=None, jobs={}, outputs={})
    provider = Provider(state)
    poll = poller(tmp_path, state, config, provider, monitor)
    poll()
    run_for(clock, poll, 3600)
    assert provider.deletes == [] and not state["managed_session"].get("retired")


# 9. Uncertain DELETE


def test_uncertain_delete_is_not_retirement_and_retries(tmp_path, clock):
    state, config = batch_owner(tmp_path)
    provider = Provider(state, stuck=True)
    poll = poller(tmp_path, state, config, provider, Monitor())
    poll()
    run_for(clock, poll, 900)
    saved = state["managed_session"]
    assert provider.deletes == ["pod"]
    assert not saved.get("retired") and not saved.get("billing_confirmed_stopped")
    assert saved["delete_uncertain_at"] and "delete_uncertain" in events(state)
    run_for(clock, poll, 10)
    assert provider.deletes == ["pod", "pod"]  # Retried on a later poll; never a create.
    provider.stuck = False
    run_for(clock, poll, 10)
    assert saved["retired"] is True and events(state).count("delete_uncertain") == 1


# 10. Disabled


def test_zero_minutes_disables_failure_shutdown(tmp_path, clock):
    state, config = batch_owner(tmp_path, failure_shutdown_minutes=0)
    provider = Provider(state)
    poll = poller(tmp_path, state, config, provider, Monitor())
    poll()
    run_for(clock, poll, 7200)
    assert provider.deletes == [] and state["managed_session"]["failure_remaining_seconds"] is None


# 11. Webhook


def test_webhook_posts_once_per_event_without_secrets(tmp_path, clock, monkeypatch):
    posted = []
    monkeypatch.setattr(cost_safety, "_post", lambda url, text: posted.append((url, text)))
    url = "https://ntfy.example/secret-topic"
    Settings(alert_webhook_url=url).save()  # Global settings only.
    state, config = batch_owner(tmp_path, browser_login_file=str(tmp_path / "login.json"),
                                comfy_url="https://pod-8189.proxy.runpod.net")
    provider = Provider(state)
    poll = poller(tmp_path, state, config, provider, Monitor())
    for _ in range(3):
        poll()
    run_for(clock, poll, 900)
    run_for(clock, poll, 30)
    assert [text.splitlines()[1] for _, text in posted] == [
        cost_safety.EVENTS["owner_failure"], cost_safety.EVENTS["failure_countdown"],
        cost_safety.EVENTS["failure_retired"]]
    for target, text in posted:
        assert target == url
        assert "NVIDIA RTX PRO 6000" in text and "$1.89/h" in text and "Batch bbbbbbbb" in text
        for secret in (url, "secret-topic", "proxy.runpod.net", "login", str(tmp_path), "/tmp", "pod-"):
            assert secret not in text, secret


def test_webhook_failure_never_affects_lifecycle(tmp_path, clock, monkeypatch):
    def broken(url, text):
        raise OSError("network down")
    monkeypatch.setattr(cost_safety, "_post", broken)
    Settings(alert_webhook_url="https://ntfy.example/topic").save()
    state, config = batch_owner(tmp_path)
    provider = Provider(state)
    poll = poller(tmp_path, state, config, provider, Monitor())
    poll()
    run_for(clock, poll, 900)
    assert provider.deletes == ["pod"] and state["managed_session"]["retired"] is True
    assert state["managed_session"]["alert_errors"]


# 12. Schema drift


def test_schema_drift_is_recorded_without_touching_the_cache(tmp_path, clock):
    cache = tmp_path / "cloud.json"
    cached = {"UNETLoader": {"input": {"required": {"unet_name": [["bf16.safetensors"]]}}},
              "OldNode": {"input": {}}}
    cache.write_text(json.dumps({"object_info": cached}))
    before = (cache.read_bytes(), cache.stat().st_mtime_ns)
    live = {"UNETLoader": {"input": {"required": {"unet_name": [["bf16.safetensors", "int8.safetensors"]]}}},
            "NewNode": {"input": {}}}
    state = {"managed_session": {"backend": "global_desktop", "sid": "sid1", "name": "sg-owned-gpu"}}
    notes = []
    session = GlobalSession(Settings(cloud=True, managed_runpod=True, global_desktop=True,
                                     runpod_schema_file=str(cache)),
                            tmp_path, state, lambda: None, notes.append, Event(), Event())
    session._check_drift(live)
    drift = state["managed_session"]["schema_drift"]
    assert drift["drift"] and drift["added_node_types"] == ["NewNode"] and drift["removed_node_types"] == ["OldNode"]
    assert drift["changed_model_choices"] == [{"class_type": "UNETLoader", "input": "unet_name",
                                               "added": ["int8.safetensors"], "removed": []}]
    assert json.loads((tmp_path / "live-object-info-sid1.json").read_text()) == live
    assert (cache.read_bytes(), cache.stat().st_mtime_ns) == before  # Never adopted automatically.
    assert notes and "differs" in notes[0]["message"]
    session._check_drift({"Other": {"input": {}}})  # Once per session.
    assert json.loads((tmp_path / "live-object-info-sid1.json").read_text()) == live

    result = cost_safety.adopt_live_schema(session.settings)
    assert json.loads(cache.read_text()) == live
    assert (tmp_path / result["backup"].rsplit("/", 1)[1]).read_bytes() == before[0]
    assert not list(tmp_path.glob("*.adopt.tmp"))


def test_drift_does_not_change_armed_batch_fingerprint(tmp_path, clock):
    from simplegen import batches
    cache = tmp_path / "cloud.json"
    cache.write_text(json.dumps({"A": {"input": {}}}))
    target = tmp_path / "member"
    target.mkdir()
    (target / "state.json").write_text(json.dumps({"concept": "A meadow",
                                                   "settings": Settings(runpod_schema_file=str(cache)).public_dict()}))
    for name in ("workflow.json", "plan.json", "template.original.json"):
        (target / name).write_text("{}")
    before = batches._fingerprint(target)  # Includes the cache bytes.
    state = {"managed_session": {"backend": "global_desktop", "sid": "sid2", "name": "sg-owned-gpu"}}
    GlobalSession(Settings(runpod_schema_file=str(cache)), tmp_path, state, lambda: None, lambda _: None,
                  Event(), Event())._check_drift({"B": {"input": {}}})
    assert state["managed_session"]["schema_drift"]["drift"]
    assert batches._fingerprint(target) == before


# 13. Old saved sessions


def test_old_sessions_and_settings_load_and_behave_as_before(tmp_path, clock):
    old_settings = {k: v for k, v in Settings(cloud=True, managed_runpod=True, global_desktop=True).public_dict().items()
                    if k not in ("failure_shutdown_minutes", "startup_timeout_minutes", "alert_webhook_url")}
    config = Settings(**old_settings)
    assert (config.failure_shutdown_minutes, config.startup_timeout_minutes, config.alert_webhook_url) == (15, 45, "")
    member(tmp_path / "m1", "m1", "p1", True)
    saved = {"backend": "global_desktop", "sid": "s", "pod_id": "pod", "name": "sg-owned-gpu",
             "jobs": {"m1": {"prompt_id": "p1", "status": "running", "outputs": []}}}
    state = {"managed_session": saved}
    provider, monitor = Provider(state), Monitor()
    monitor.observation["jobs"] = {"p1": "error"}
    monitor.observation["outputs"] = {}
    poll = poller(tmp_path, state, config, provider, monitor)
    poll()
    run_for(clock, poll, 7200)  # A failed job without an owner failure record: unchanged, never deleted.
    assert provider.deletes == [] and "owner_failure" not in saved and saved["failure_remaining_seconds"] is None


def test_settings_validation():
    for bad in ({"failure_shutdown_minutes": 241}, {"failure_shutdown_minutes": -1},
                {"startup_timeout_minutes": 4}, {"alert_webhook_url": "ftp://x"}):
        with pytest.raises(ValueError):
            Settings(**bad).validate(require_folders=False)
    Settings(failure_shutdown_minutes=0, alert_webhook_url="https://ntfy.sh/t").validate(require_folders=False)


# Desktop alerts


@pytest.mark.usefixtures("_no_desktop_audio")
def test_desktop_alerts_show_once(tmp_path, monkeypatch):
    monkeypatch.setenv("SIMPLEGEN_HOME", str(tmp_path / "home"))
    from PySide6.QtWidgets import QApplication

    from simplegen import ui
    app = QApplication.instance() or QApplication([])
    window = ui.MainWindow(Settings(output_folder=str(tmp_path)))
    alert = {"key": "s:owner_failure:x", "message": "simpleGen · Batch 1\nFailed"}
    window.notify_alerts([alert])
    assert "ALERT" in window.global_status.text() and "Failed" in window.global_status.text()
    window.global_status.setText("")
    window.notify_alerts([alert])
    assert window.global_status.text() == ""
    window.close()
    window.deleteLater()
    app.processEvents()


# Review follow-up


def test_success_without_observed_outputs_is_unresolved(tmp_path, clock):
    state, config = batch_owner(tmp_path)
    provider, monitor = Provider(state), Monitor()
    monitor.observation["outputs"] = {}  # Success reported, outputs never observed.
    poll = poller(tmp_path, state, config, provider, monitor)
    poll()
    run_for(clock, poll, 3600)
    saved = state["managed_session"]
    assert provider.deletes == [] and not saved.get("retired")
    assert saved["attention"] == "outputs_not_collected"
    assert saved["uncollected_outputs"] == [{"prompt_id": "p1", "filename": None, "note": "no outputs observed"}]
    assert saved["failure_remaining_seconds"] is None
    assert events(state).count("outputs_not_collected") == 1


@pytest.mark.parametrize("status", ["missing", "submission_pending"])
def test_unresolved_jobs_alert_once_and_keep_the_gpu(tmp_path, clock, monkeypatch, status):
    posted = []
    monkeypatch.setattr(cost_safety, "_post", lambda url, text: posted.append(text))
    Settings(alert_webhook_url="https://ntfy.example/topic").save()
    state, config = batch_owner(tmp_path)
    monitor = Monitor()
    if status == "missing":
        monitor.observation["jobs"] = {"p1": "success", "p2": "missing"}
    else:
        state["managed_session"]["jobs"]["m2"] = {"prompt_id": None, "status": "submission_pending", "outputs": []}
    provider = Provider(state)
    poll = poller(tmp_path, state, config, provider, monitor)
    poll()
    run_for(clock, poll, 3600)
    saved = state["managed_session"]
    assert provider.deletes == [] and saved["unresolved_jobs"] == ["m2"]
    assert events(state).count("unresolved_jobs") == 1
    text = next(t for t in posted if cost_safety.EVENTS["unresolved_jobs"] in t)
    assert "m2" in text and "kept and billing" in text
    assert any("no known outcome" in line for line in cost_safety.session_notices(saved))


def test_webhook_url_never_reaches_project_state(tmp_path, clock, monkeypatch):
    from simplegen import batches
    monkeypatch.setattr(cost_safety, "_post", lambda url, text: None)  # Never contact a real endpoint.
    url = "https://ntfy.example/secret-topic"
    settings = Settings(project_folder=str(tmp_path), template_folder=str(tmp_path),
                        output_folder=str(tmp_path / "out"), alert_webhook_url=url)
    settings.save()
    assert Settings.load().alert_webhook_url == url  # Kept globally.
    assert "alert_webhook_url" not in settings.public_dict()
    (tmp_path / "t.json").write_text("{}")
    folder, _ = engine.create_run(replace(settings, template_file="t.json", cloud=True, managed_runpod=True,
                                          global_desktop=True), "A meadow")
    store = batches.BatchStore(settings.output_folder)
    store.add(store.create()["id"], folder)
    state, config = batch_owner(tmp_path / "owner", alert_webhook_url=url)
    (tmp_path / "owner" / "state.json").write_text(json.dumps({**state, "settings": config.public_dict()}))
    states = list(tmp_path.rglob("state.json"))
    assert len(states) >= 4
    for path in states:
        assert url not in path.read_text() and "alert_webhook_url" not in path.read_text(), path


# Polling other owned sessions while a batch worker runs


def owned_folder(root, name, *, completed):
    folder = root / name
    folder.mkdir(parents=True)
    settings = Settings(cloud=True, managed_runpod=True, global_desktop=True, failure_shutdown_minutes=1)
    saved = {"backend": "global_desktop", "sid": name, "pod_id": f"pod-{name}", "name": "sg-owned-gpu",
             "gpu_id": "GPU", "actual_hourly": 1.0}
    if completed:
        from simplegen_runpod.outputs import manifest
        jobs = {"r": {"prompt_id": "p1", "status": "success", "outputs": [OUTPUT]}}
        saved.update(jobs=jobs, downloads_verified=True, acknowledged_manifest=manifest(jobs))
    else:
        saved.update(jobs={"r": {"prompt_id": "p2", "status": "running", "outputs": []}},
                     owner_failure={"at": "2026-09-28T00:00:00+00:00", "reason": "boom", "kind": "run"})
    state = {"concept": name, "settings": settings.public_dict(), "managed_session": saved}
    (folder / "state.json").write_text(json.dumps(state))
    return folder


@pytest.mark.usefixtures("_no_desktop_audio")
@pytest.mark.parametrize("completed", [False, True])
def test_other_sessions_retire_while_a_batch_worker_runs(tmp_path, clock, monkeypatch, completed):
    import time as real_time

    from PySide6.QtWidgets import QApplication

    from simplegen import global_registry, ui
    monkeypatch.setattr(gs, "IDLE_SECONDS", 30)
    running = tmp_path / "running-batch"
    running.mkdir()
    other = owned_folder(tmp_path, "other", completed=completed)
    global_registry.register(running)
    global_registry.register(other)
    providers = {}

    class Present(Provider):
        def delete(self, pod_id):
            self.deletes.append(pod_id)
            self.present = False
    real_session_for = gs.session_for

    def session_for(settings, folder, state, *args):
        assert Path(folder).resolve() != running.resolve(), "running batch must not be polled"
        session = real_session_for(settings, folder, state, *args)
        session._provider = providers.setdefault(str(folder), Present(state))
        monitor = Monitor()
        monitor.observation["jobs"] = {"p1": "success"} if completed else {"p2": "error"}
        monitor.observation["outputs"] = {"p1": [OUTPUT]} if completed else {}
        session._monitor = monitor
        return session
    monkeypatch.setattr(gs, "session_for", session_for)
    app = QApplication.instance() or QApplication([])
    window = ui.MainWindow(Settings(output_folder=str(tmp_path)))
    window.global_timer.stop()
    window.batch_worker = object()  # A batch is running for hours.
    window.batch_worker_folder = running
    try:
        for _ in range(12):
            window.poll_registered_global_sessions()
            end = real_time.time() + 5
            while window.global_worker is not None and real_time.time() < end:
                app.processEvents()
                real_time.sleep(0.005)
            clock[0] += 10
        saved = json.loads((other / "state.json").read_text())["managed_session"]
        assert saved["retired"] is True and saved["billing_confirmed_stopped"] is True
        assert saved.get("delete_intent") == ("verified_downloads_idle" if completed else "failure_idle")
        assert global_registry.listed() == [running.resolve()]
    finally:
        window.batch_worker = None
        window.close()
        window.deleteLater()
        app.processEvents()
