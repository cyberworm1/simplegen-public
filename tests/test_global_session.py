import copy
import json
from threading import Event

import httpx
import pytest

from simplegen.comfy import Comfy
from simplegen.global_session import GlobalSession, is_global
from simplegen.settings import Settings
from simplegen_runpod.outputs import manifest


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv("SIMPLEGEN_HOME", str(tmp_path / "simplegen-home"))


def test_batch_ceiling_applies_only_to_new_owned_session(tmp_path):
    from importlib.resources import files

    profile = tmp_path / "profile.json"
    profile.write_text(json.dumps({**json.loads(files("simplegen").joinpath("data/global-runpod-profile.json").read_text()),
                     "template_id": "tmpl-test", "volume_id": "vol-test"}))
    settings = Settings(runpod_config_file=str(profile), managed_runpod=True, global_desktop=True)
    state = {"members": [{"id": "member"}], "max_hourly": 5.5, "managed_session": {}}
    session = GlobalSession(settings, tmp_path, state, lambda: None, lambda _: None, Event(), Event())
    assert session.config().gpu_max_hourly == 5.5
    assert GlobalSession(settings, tmp_path, {"max_hourly": 5.5}, lambda: None,
                         lambda _: None, Event(), Event()).config().gpu_max_hourly == 3.0
    from dataclasses import asdict
    original = session.config()
    state["managed_session"]["frozen_profile"] = {
        "profile": asdict(original.global_profile), "gpu_tiers": original.gpu_tiers,
        "gpu_max_hourly": 4.5,
    }
    state["max_hourly"] = 9.0
    assert session.config().gpu_max_hourly == 4.5


def test_local_download_stays_checksumming_with_global_default_profile(tmp_path):
    profile = tmp_path / "global-profile.json"
    profile.write_text('{"mode":"global_desktop"}')
    settings = Settings(runpod_config_file=str(profile), cloud=False, managed_runpod=False)
    assert not is_global(settings)
    comfy = Comfy(settings, httpx.Client(transport=httpx.MockTransport(
        lambda _: httpx.Response(200, content=b"video"))))
    receipt = comfy.download({"filename": "clip.mp4"}, tmp_path / "clip.mp4", Event())
    assert "sha256" in receipt and "verification" not in receipt
    legacy = Settings(cloud=True, managed_runpod=True, runpod_config_file=str(profile))
    assert not is_global(legacy, {"managed_session": {"sid": "legacy-id"}})


def test_global_download_requires_content_length_and_uses_size_only(tmp_path):
    settings = Settings(cloud=True, managed_runpod=True, global_desktop=True,
                        comfy_url="https://pod.example")
    comfy = Comfy(settings, httpx.Client(transport=httpx.MockTransport(
        lambda _: httpx.Response(200, content=b"video"))))
    receipt = comfy.download({"filename": "clip.mp4"}, tmp_path / "clip.mp4", Event())
    assert receipt["verification"] == "size" and "sha256" not in receipt


def test_idle_window_survives_new_desktop_session_objects_but_not_monitor_gap(tmp_path, monkeypatch):
    from simplegen import global_session as module
    monkeypatch.setenv("SIMPLEGEN_HOME", str(tmp_path / "home"))
    clock = [1000.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])
    output = {"id": "file1", "prompt_id": "p", "filename": "clip.mp4",
              "subfolder": "", "type": "output"}
    jobs = {"job": {"prompt_id": "p", "status": "success", "outputs": [output]}}
    observed_outputs = [output]
    state = {"run_id": "job", "managed_session": {
        "backend": "global_desktop", "sid": "s", "pod_id": "pod", "name": "sg-owned-gpu",
        "jobs": jobs, "downloads_verified": False, "acknowledged_manifest": manifest(jobs),
    }}

    class Provider:
        def __init__(self):
            self.deleted = False
            self.client = httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(200)))
        def get(self, pod_id):
            return None if self.deleted else {"id": pod_id, "name": "sg-owned-gpu", "desiredStatus": "RUNNING"}
        def delete(self, pod_id):
            self.deleted = True

    class Monitor:
        def poll(self, _jobs):
            return {"ready": True, "healthy": True, "auth_verified": True,
                    "queue_empty": True, "jobs": {"p": "success"}, "outputs": {"p": observed_outputs[:]}}
        client = httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(200)))

    provider = Provider()
    settings = Settings(cloud=True, managed_runpod=True, global_desktop=True)
    def poll():
        session = GlobalSession(settings, tmp_path, state, lambda: None,
                                lambda _: None, Event(), Event())
        session._provider = provider
        session._monitor = Monitor()
        return session.status()

    assert poll()["idle_remaining_seconds"] is None
    clock[0] += 300
    assert poll()["idle_remaining_seconds"] is None and not provider.deleted
    state["managed_session"]["downloads_verified"] = True
    assert poll()["idle_remaining_seconds"] == 300
    clock[0] += 10
    assert poll()["idle_remaining_seconds"] == 290
    observed_outputs.append({**output, "id": "file2", "filename": "extra.mp4"})
    clock[0] += 10
    assert poll()["downloads_verified"] is False and not provider.deleted
    state["managed_session"]["acknowledged_manifest"] = manifest(state["managed_session"]["jobs"])
    state["managed_session"]["downloads_verified"] = True
    assert poll()["idle_remaining_seconds"] == 300
    clock[0] += 31  # Missing observation resets the complete idle interval.
    assert poll()["idle_remaining_seconds"] == 300
    for _ in range(30):
        clock[0] += 10
        poll()
    assert provider.deleted
    assert state["managed_session"]["billing_confirmed_stopped"] is True


def test_lost_create_reconciles_exact_owned_name_without_second_post(tmp_path, monkeypatch):
    from simplegen import global_session as module
    monkeypatch.setenv("SIMPLEGEN_HOME", str(tmp_path / "home"))
    import importlib.resources
    profile = tmp_path / "profile.json"
    profile.write_text(json.dumps({**json.loads(importlib.resources.files("simplegen").joinpath(
        "data/global-runpod-profile.json").read_text()), "template_id": "tmpl-test", "volume_id": "vol-test"}))
    schema = tmp_path / "schema.json"
    schema.write_text('{"SaveVideo":{"input":{}}}')
    settings = Settings(cloud=True, managed_runpod=True, global_desktop=True,
                        runpod_config_file=str(profile), runpod_schema_file=str(schema),
                        browser_login_file=str(tmp_path / "login.json"))
    state, writes = {"run_id": "job"}, []
    persist = lambda: writes.append(copy.deepcopy(state))
    monkeypatch.setattr(module, "_secret", lambda *_: "private-api")

    class Quote:
        client = type("Client", (), {"close": lambda self: None})()
        def __init__(self, key):
            pass
        def quote(self, payload):
            return 1.0
        def _request(self, method, path, catalog=False):
            assert "minCudaVersion=13.0" in path and "cloud=SECURE" in path
            return {"gpus": [{"id": "NVIDIA RTX PRO 6000 Blackwell Server Edition",
                              "secure": True,
                              "dataCenters": [{"id": "US-TX-1", "availability": "HIGH"}]}]}

    class Provider:
        calls = 0
        def deploy(self, profile, *, name, gpu_type_id, data_center_id):
            self.calls += 1
            assert data_center_id is None
            assert writes[-1]["managed_session"]["create_attempted"] is True
            assert writes[-1]["managed_session"]["placement"] == "global"
            raise module.CreateUncertain("lost create response")
        def reconcile_exact_name(self, name):
            return {"id": "ownedpod", "name": name, "costPerHr": 1.0}
        def get(self, pod_id):
            return {"id": pod_id, "name": state["managed_session"]["name"],
                    "desiredStatus": "RUNNING"}

    class Comfy:
        def __init__(self, settings):
            pass
        def request(self, method, path):
            return type("Response", (), {"json": lambda self: {"devices": [{"type": "cuda"}]}})()
        def object_info(self):
            return {"SaveVideo": {}}
        def close(self):
            pass

    monkeypatch.setattr(module, "RunPod", Quote)
    monkeypatch.setattr("simplegen.comfy.Comfy", Comfy)
    provider = Provider()
    choice = {"gpu_id": "NVIDIA RTX PRO 6000 Blackwell Server Edition", "gpu_count": 1,
              "volume_id": "vol-test", "hourly_usd": 1.0,
              "data_center": "US-TX-1"}
    first = GlobalSession(settings, tmp_path, state, persist, lambda _: None, Event(), Event())
    first._provider = provider
    with pytest.raises(module.CreateUncertain):
        first.connect(choice)
    assert provider.calls == 1 and "pod_id" not in state["managed_session"]
    # Phase 4: new sessions freeze the keep-warm window (Settings.keep_warm_minutes, default 5).
    assert state["managed_session"]["frozen_profile"]["keep_warm_minutes"] == settings.keep_warm_minutes == 5
    second = GlobalSession(settings, tmp_path, state, persist, lambda _: None, Event(), Event())
    second._provider = provider
    endpoint = second.connect()
    assert endpoint.comfy_url == "https://ownedpod-8189.proxy.runpod.net"
    assert provider.calls == 1


def test_saved_legacy_create_intent_keeps_its_region(tmp_path):
    from importlib.resources import files

    from simplegen.global_session import CreateUncertain

    profile = tmp_path / "profile.json"
    profile.write_text(json.dumps({**json.loads(files("simplegen").joinpath("data/global-runpod-profile.json").read_text()),
                     "template_id": "tmpl-test", "volume_id": "vol-test"}))
    schema = tmp_path / "schema.json"
    schema.write_text('{"SaveVideo":{"input":{}}}')
    state = {"managed_session": {"backend": "global_desktop", "sid": "legacy",
                                 "name": "sg-legacy-gpu", "gpu_id": "GPU",
                                 "data_center_id": "US-TX-1", "create_intent": True,
                                 "accepted_hourly": 2.0}}
    settings = Settings(cloud=True, managed_runpod=True, global_desktop=True,
                        runpod_config_file=str(profile), runpod_schema_file=str(schema))
    session = GlobalSession(settings, tmp_path, state, lambda: None,
                            lambda _: None, Event(), Event())
    class Provider:
        def deploy(self, _profile, *, name, gpu_type_id, data_center_id):
            assert name == "sg-legacy-gpu" and gpu_type_id == "GPU"
            assert data_center_id == "US-TX-1"
            raise CreateUncertain("test stops before paid create")
    session._provider = Provider()
    with pytest.raises(CreateUncertain):
        session.connect()
    assert state["managed_session"]["data_center_id"] == "US-TX-1"


def test_detach_before_fresh_global_allocation_makes_no_post(tmp_path):
    settings = Settings(cloud=True, managed_runpod=True, global_desktop=True)
    detached = Event()
    detached.set()
    state = {"run_id": "job"}
    session = GlobalSession(settings, tmp_path, state, lambda: None,
                            lambda _: None, Event(), detached)
    with pytest.raises(RuntimeError, match="cancelled"):
        session.connect({"gpu_id": "GPU"})
    assert "managed_session" not in state


@pytest.mark.parametrize("timeout, ready_after, succeeds", [(25, 1300, True), (5, None, False)])
def test_global_readiness_uses_configured_timeout_without_deleting_pod(
    tmp_path, monkeypatch, timeout, ready_after, succeeds
):
    from simplegen import global_session as module
    from simplegen.comfy import ComfyRequestError

    clock = [1000.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(module.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds))
    state = {"managed_session": {"backend": "global_desktop", "sid": "existing",
                                 "pod_id": "ownedpod", "name": "sg-owned-gpu",
                                 "actual_hourly": 1.0, "accepted_hourly": 3.0}}
    session = GlobalSession(
        Settings(cloud=True, managed_runpod=True, global_desktop=True,
                 startup_timeout_minutes=timeout, generation_timeout=21600), tmp_path, state,
        lambda: None, lambda _: None, Event(), Event(),
    )
    monkeypatch.setattr(session, "schema", lambda: {"SaveVideo": {}})
    monkeypatch.setattr(session, "config", lambda: module.GlobalConfig(
        module.GlobalGPUProfile(template_id="tmpl-test", volume_id="vol-test"), ["GPU A"], 3.0))
    monkeypatch.setattr(session, "_owned_pod", lambda: {
        "id": "ownedpod", "name": "sg-owned-gpu", "desiredStatus": "RUNNING"})
    monkeypatch.setattr(session, "_endpoint", lambda _: session.settings)

    class Comfy:
        def __init__(self, settings):
            pass
        def request(self, method, path):
            if ready_after is None or clock[0] - 1000 < ready_after:
                raise ComfyRequestError("startup pending", retryable=True)
            return type("Response", (), {"json": lambda self: {"devices": [{"type": "cuda"}]}})()
        def object_info(self):
            return {"SaveVideo": {}}
        def close(self):
            pass

    monkeypatch.setattr("simplegen.comfy.Comfy", Comfy)
    if succeeds:
        assert session.connect() is session.settings
        assert clock[0] - 1000 >= 1300
    else:
        with pytest.raises(RuntimeError, match="readiness timed out"):
            session.connect()
        assert clock[0] - 1000 == timeout * 60  # Startup timeout, not the 6 h generation timeout.
        assert state["managed_session"]["owner_failure"]["kind"] == "startup"
    assert state["managed_session"].get("retired") is not True


def test_registry_monitors_background_projects_and_keeps_failed_entry(tmp_path, monkeypatch):
    from simplegen import engine, global_registry

    first, second = tmp_path / "one", tmp_path / "two"
    first.mkdir()
    second.mkdir()
    global_registry.register(first)
    global_registry.register(second)
    seen = []

    def status(project, action):
        seen.append((project, action))
        if project == first:
            return {"billing_confirmed_stopped": True}
        raise RuntimeError("provider unavailable")

    monkeypatch.setattr(engine, "cloud_action", status)
    result = global_registry.poll_registered()
    assert set(seen) == {(first, "status"), (second, "status")}
    assert result["active"] == 1 and result["needs_attention"] == [str(second)]
    assert global_registry.listed() == [second]


@pytest.mark.parametrize("actual_name", ["sg-owned", "someone-else"])
def test_normal_delete_refuses_missing_ack_and_wrong_owner(tmp_path, actual_name):
    state = {"run_id": "job", "managed_session": {
        "backend": "global_desktop", "sid": "s", "pod_id": "pod", "name": "sg-owned",
        "downloads_verified": False,
    }}
    class Provider:
        deleted = False
        def get(self, pod_id):
            return {"id": pod_id, "name": actual_name, "desiredStatus": "RUNNING"}
        def delete(self, pod_id):
            self.deleted = True
    provider = Provider()
    session = GlobalSession(Settings(cloud=True, managed_runpod=True, global_desktop=True),
                            tmp_path, state, lambda: None, lambda _: None, Event(), Event())
    session._provider = provider
    with pytest.raises(RuntimeError):
        session._delete_owned("verified_downloads_idle")
    assert not provider.deleted


def test_explicit_shutdown_deletes_only_owned_pod_and_waits_for_404(tmp_path):
    state = {"run_id": "job", "managed_session": {
        "backend": "global_desktop", "sid": "s", "pod_id": "pod", "name": "sg-owned",
        "jobs": {}, "downloads_verified": False,
    }}
    class Provider:
        deleted = False
        def get(self, pod_id):
            return None if self.deleted else {"id": pod_id, "name": "sg-owned",
                                                 "desiredStatus": "RUNNING"}
        def delete(self, pod_id):
            assert state["managed_session"]["delete_intent"] == "explicit_user_shutdown"
            self.deleted = True
    provider = Provider()
    session = GlobalSession(Settings(cloud=True, managed_runpod=True, global_desktop=True),
                            tmp_path, state, lambda: None, lambda _: None, Event(), Event())
    session._provider = provider
    session.status = lambda: state["managed_session"]
    session.shutdown()
    assert provider.deleted
    assert state["managed_session"]["billing_confirmed_stopped"] is True


@pytest.mark.parametrize("change, message", [
    ("deleted", "deleted in RunPod during startup"),
    ("EXITED", r"stopped in RunPod during startup \(EXITED\)"),
    ("TERMINATED", r"stopped in RunPod during startup \(TERMINATED\)"),
])
def test_pod_removed_during_startup_ends_the_wait_within_a_minute(tmp_path, monkeypatch, change, message):
    """Reported 2026-10-01: a pod deleted in RunPod mid-startup left the batch waiting the full 45 minutes."""
    from simplegen import batches
    from simplegen import global_session as module
    from simplegen.comfy import ComfyRequestError
    from simplegen_runpod.provider import ProviderError

    clock = [1000.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(module.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds))
    state = {"managed_session": {"backend": "global_desktop", "sid": "existing", "create_intent": True,
                                 "pod_id": "ownedpod", "name": "sg-owned-gpu", "jobs": {},
                                 "actual_hourly": 1.0, "accepted_hourly": 3.0}}
    session = GlobalSession(
        Settings(cloud=True, managed_runpod=True, global_desktop=True, startup_timeout_minutes=45),
        tmp_path, state, lambda: None, lambda _: None, Event(), Event(),
    )
    monkeypatch.setattr(session, "schema", lambda: {"SaveVideo": {}})
    monkeypatch.setattr(session, "config", lambda: module.GlobalConfig(module.GlobalGPUProfile(template_id="tmpl-test", volume_id="vol-test"), ["GPU A"], 3.0))
    removed_at = 300  # Seconds into startup when the user deletes or stops the pod.
    lookups = []

    def owned_pod():
        elapsed = clock[0] - 1000
        lookups.append(elapsed)
        if len(lookups) == 3:
            raise ProviderError("RunPod unreachable", status=503)  # A failed lookup never ends the wait.
        if elapsed < removed_at:
            return {"id": "ownedpod", "name": "sg-owned-gpu", "desiredStatus": "RUNNING"}
        return None if change == "deleted" else {"id": "ownedpod", "name": "sg-owned-gpu", "desiredStatus": change}
    monkeypatch.setattr(session, "_owned_pod", owned_pod)
    monkeypatch.setattr(session, "_endpoint", lambda _: session.settings)

    class Comfy:
        def __init__(self, settings):
            pass
        def request(self, method, path):
            raise ComfyRequestError("startup pending", retryable=True)
        def close(self):
            pass
    monkeypatch.setattr("simplegen.comfy.Comfy", Comfy)
    with pytest.raises(RuntimeError, match=message):
        session.connect()
    assert removed_at <= clock[0] - 1000 <= removed_at + module.STARTUP_POD_CHECK_SECONDS + 2
    saved = state["managed_session"]
    if change == "deleted":
        assert saved["retired"] and saved["billing_confirmed_stopped"] and saved["session_completed_at"]
        assert batches.retired_without_work(saved)  # The batch can be released and run again.
    else:
        # The pod still exists: it stays owned and the failure shutdown policy deletes it.
        assert not saved.get("retired") and not saved.get("billing_confirmed_stopped")
        assert saved["owner_failure"]["kind"] == "startup"
        assert not batches.retired_without_work(saved)
