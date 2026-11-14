"""Phase 4: additions to a running batch, sealing/reopen, warm-pod handoff, spend cap.

Real BatchStore.run_batch and engine with fake ComfyUI/session (from the shared
`sequence` harness), fake provider and monitor, simulated clocks.
"""

import copy
import json
import os
from importlib.resources import files
from pathlib import Path
from threading import Event

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from test_variations import sequence  # noqa: F401
from test_workflows import info, workflow  # noqa: F401

from simplegen import additions, batches, engine, global_registry, global_session, resources
from simplegen.settings import Settings
from simplegen_runpod.global_gpu import GlobalGPUProfile


def make_batch(store, settings, concepts):
    batch = store.create()
    for concept in concepts:
        folder, _ = engine.create_run(settings, concept)
        store.add(batch["id"], folder)
    return batch["id"]


@pytest.fixture
def env(sequence, info, monkeypatch):  # noqa: F811
    _, agent_calls, jobs, trace, events, fault, detach, _, root, _ = sequence
    profile = root / "global-profile.json"
    profile.write_text(json.dumps({**json.loads(files("simplegen").joinpath("data/global-runpod-profile.json").read_text()),
                     "template_id": "tmpl-test", "volume_id": "vol-test"}))
    settings = Settings(project_folder=str(root / "assets"), template_folder=str(root / "templates"),
                        template_file="template.json", output_folder=str(root / "runs"), quantity=3, duration=4,
                        managed_runpod=True, global_desktop=True, runpod_config_file=str(profile))
    store = batches.BatchStore(settings.output_folder)
    catalog = [{"gpu_id": identity, "hourly_usd": rate, "gpu_count": 1, "vram_gb": 96, "availability": "HIGH",
                "data_center": "US-TEST", "data_centers": [{"id": "US-TEST", "availability": "HIGH"}]}
               for identity, rate in [("a", 1), ("b", 2)]]
    monkeypatch.setattr(batches, "_catalog", lambda *args, **kwargs: copy.deepcopy(catalog))
    monkeypatch.setattr(resources, "assess_batch", lambda *args: pytest.fail("Unexpected GPU agent"))
    world = {"allocations": [], "completed": [], "live": copy.deepcopy(info), "live_calls": 0}

    class Session:
        retired = False

        def __init__(self, config, folder, state, persist, *args):
            self.settings, self.state, self.persist = config, state, persist

        def schema(self):
            return copy.deepcopy(info)

        def config(self):
            return global_session.GlobalConfig(GlobalGPUProfile(template_id="tmpl-test", volume_id="vol-test"), ["a", "b"], 3)

        def connect(self, choice=None):
            saved = self.state.setdefault("managed_session", {})
            if not saved.get("sid"):
                saved.update(backend="global_desktop", sid="shared", name="sg-shared-gpu", pod_id="pod",
                             create_intent=True, jobs={}, gpu_id="a", actual_hourly=1.0,
                             session_started_at="2026-09-28T00:00:00+00:00")
                world["allocations"].append("pod")
                self.persist()
            return self.settings

        def before_submit(self, job_id=None):
            registered = self.state["managed_session"]["jobs"]
            if self.state["managed_session"].get("downloads_verified"):
                raise RuntimeError("sealed")
            if job_id in registered:
                return False
            registered[job_id] = None
            self.persist()
            return True

        def bind(self, prompt_id, job_id=None):
            self.state["managed_session"]["jobs"][job_id] = prompt_id
            self.persist()

        def complete(self, receipts):
            assert len(receipts) == 2 * len(self.state["managed_session"]["jobs"])
            self.state["managed_session"]["downloads_verified"] = True
            world["completed"].append(len(receipts))
            self.persist()

        def close(self):
            pass
    monkeypatch.setattr(global_session, "session_for", Session)

    def live(endpoint):
        world["live_calls"] += 1
        return copy.deepcopy(world["live"])
    store.live = live

    def start(concepts=("First theme", "Second theme")):
        batch_id = make_batch(store, settings, concepts)
        assert store.prepare(batch_id, events.append, Event())["status"] == "prepared"
        store.configure(batch_id, catalog_options=catalog,
                        gpu_choices=[{"gpu_id": "a", "max_hourly": 1}, {"gpu_id": "b", "max_hourly": 2}])
        store.run_now(batch_id)
        return batch_id
    return {"store": store, "settings": settings, "jobs": jobs, "trace": trace, "events": events, "world": world,
            "start": start, "agent_calls": agent_calls, "detach": detach, "fault": fault}


def add(env, batch_id, concept="Third theme", quantity=3):
    prepared = additions.prepare_addition(env["store"], batch_id, concept, quantity, env["events"].append, Event())
    additions.submit_addition(env["store"], batch_id, Path(prepared["folder"]))
    return prepared


def run(env, batch_id, hook=None):
    def on_event(event):
        env["events"].append(event)
        if hook:
            hook(event)
    return env["store"].run_batch(batch_id, on_event, Event(), env["detach"], live_schema=env["store"].live)


def job_ids(env, run_id):
    folder = next(Path(m["folder"]) for m in env["store"].get(env["batch"])["members"] if m["id"] == run_id)
    state = json.loads((folder / "state.json").read_text())
    return [json.loads((folder / c["folder"] / "state.json").read_text()).get("prompt_id")
            for c in state["sequence"]["children"]]


# 1 and 3: additions accepted at the next drain point, whatever stage they arrive in


@pytest.mark.parametrize("stage", ["Staging", "Queueing", "Queued", "Generating", "last"])
def test_addition_is_accepted_at_the_next_drain_point(env, stage):
    batch_id = env["start"]()
    env["batch"] = batch_id
    sent = []

    def hook(event):
        if sent:
            return
        if stage == "last":
            members = env["store"].get(batch_id)["members"]
            done = sum(m["status"] == "complete" for m in members)
            if done != len(members) - 1 or event.get("stage", "").find("Saved") < 0:
                return
        elif stage not in event.get("stage", ""):
            return
        # Prepared outside the batch lock (run_batch holds it right now) while the pod keeps rendering.
        sent.append(add(env, batch_id))
    result = run(env, batch_id, hook)
    assert sent, "hook never fired"
    assert result["status"] == "complete" and result["sealed_at"]
    members = result["members"]
    assert len(members) == 3 and members[2]["id"] == sent[0]["run_id"] and members[2]["added_at"]
    assert all(m["status"] == "complete" for m in members)
    assert len(env["jobs"]) == 9 and env["world"]["allocations"] == ["pod"]
    assert env["world"]["completed"] == [18]  # Completion includes the addition.
    assert env["world"]["live_calls"] >= 1  # Validated against the live pod schema.
    submitted = [value["run_id"] for value in env["jobs"].values()]
    assert len(submitted) == len(set(submitted))  # Each render submitted once.
    inbox = env["store"].inbox(batch_id)
    assert not list(inbox.glob("*.json")) and (inbox / "accepted" / f"{sent[0]['run_id']}.json").is_file()
    child = json.loads((Path(sent[0]["folder"]) / "state.json").read_text())
    assert child["batch_parent"] == batch_id and "pending_batch" not in child


def test_existing_jobs_and_receipts_are_unchanged_by_an_addition(env):
    batch_id = env["start"]()
    env["batch"] = batch_id
    before = {}

    def hook(event):
        if event.get("stage") == "Queued" and not before:
            members = env["store"].get(batch_id)["members"]
            if all(m["status"] == "queued" for m in members):
                before.update({m["id"]: job_ids(env, m["id"]) for m in members})
                add(env, batch_id)
    result = run(env, batch_id, hook)
    assert before and all(job_ids(env, run_id) == ids for run_id, ids in before.items())
    assert result["status"] == "complete"


# 2: duplicates and another batch


def test_duplicate_addition_is_accepted_once_and_other_batch_is_rejected(env):
    batch_id = env["start"]()
    env["batch"] = batch_id
    prepared = additions.prepare_addition(env["store"], batch_id, "Third theme", 3, env["events"].append, Event())
    additions.submit_addition(env["store"], batch_id, Path(prepared["folder"]))
    additions.submit_addition(env["store"], batch_id, Path(prepared["folder"]))  # Same request again.
    other = make_batch(env["store"], env["settings"], ["Other theme"])
    with pytest.raises(ValueError, match="another batch"):
        additions.submit_addition(env["store"], other, Path(prepared["folder"]))
    result = run(env, batch_id)
    assert [m["id"] for m in result["members"]].count(prepared["run_id"]) == 1
    assert len(env["jobs"]) == 9
    # A replayed request after acceptance is idempotent too.
    request = json.loads((env["store"].inbox(batch_id) / "accepted" / f"{prepared['run_id']}.json").read_text())
    with pytest.raises(ValueError, match="another batch"):
        additions.accept_request({**env["store"].get(other), "id": other}, {**request, "batch_id": other}, {})


def test_addition_that_fails_the_live_schema_is_rejected_not_deleted(env):
    batch_id = env["start"]()
    env["batch"] = batch_id
    env["world"]["live"].pop("SaveVideo")  # The running pod lacks a node the cache has.
    prepared = add(env, batch_id)
    result = run(env, batch_id)
    assert len(result["members"]) == 2
    rejected = json.loads((env["store"].inbox(batch_id) / "rejected" / f"{prepared['run_id']}.json").read_text())
    assert "running pod" in rejected["reason"]
    child = json.loads((Path(prepared["folder"]) / "state.json").read_text())
    assert "pending_batch" not in child and child["batch_rejection"]["batch_id"] == batch_id
    assert Path(prepared["folder"]).is_dir()


# 3 (continued): after the seal (reopen during keep-warm), after delete_intent, after retirement


def test_addition_after_seal_reopens_during_keep_warm(env):
    batch_id = env["start"]()
    env["batch"] = batch_id
    first = run(env, batch_id)
    assert first["status"] == "complete" and first["managed_session"]["downloads_verified"]
    prepared = add(env, batch_id)
    results = env["store"].dispatch_due(env["events"].append, Event())
    assert results and results[0]["status"] == "complete"
    state = env["store"].get(batch_id)
    assert state["reopen"]["reason"] == "inbox request" and len(state["members"]) == 3
    assert env["world"]["allocations"] == ["pod"] and env["world"]["completed"] == [12, 18]
    assert prepared["run_id"] in [m["id"] for m in state["members"]]


@pytest.mark.parametrize("closed", ["delete_intent", "retired"])
def test_addition_to_a_closing_or_retired_batch_is_rejected(env, closed):
    batch_id = env["start"]()
    env["batch"] = batch_id
    run(env, batch_id)
    late = additions.prepare_addition(env["store"], batch_id, "Late theme", 3, env["events"].append, Event())
    folder = env["store"]._folder(batch_id)
    state = env["store"].get(batch_id)
    state["managed_session"].update({"delete_intent": "verified_downloads_idle"} if closed == "delete_intent"
                                    else {"retired": True, "billing_confirmed_stopped": True})
    (folder / "state.json").write_text(json.dumps(state))
    with pytest.raises(ValueError, match="new batch"):
        additions.prepare_addition(env["store"], batch_id, "Late", 1, env["events"].append, Event())
    additions.submit_addition(env["store"], batch_id, Path(late["folder"]))
    assert env["store"].service_inbox(batch_id) == "rejected"
    rejected = json.loads(next((env["store"].inbox(batch_id) / "rejected").glob("*.json")).read_text())
    assert "Start a new batch" in rejected["reason"]
    assert env["store"].get(batch_id)["status"] == "complete" and len(env["world"]["completed"]) == 1
    with pytest.raises(ValueError, match="new batch"):
        env["store"].reopen(batch_id)


# 4: crash points (inbox, drain, reopen)


def test_crash_between_acceptance_and_submission_recovers_once(env, monkeypatch):
    batch_id = env["start"]()
    env["batch"] = batch_id
    prepared = add(env, batch_id)
    real = additions.claim_child
    crashes = []

    def crash_once(*args):
        if not crashes:
            crashes.append(True)
            raise RuntimeError("power lost after acceptance was recorded")
        return real(*args)
    monkeypatch.setattr(additions, "claim_child", crash_once)
    with pytest.raises(RuntimeError, match="power lost"):
        run(env, batch_id)
    state = env["store"].get(batch_id)
    assert prepared["run_id"] in [m["id"] for m in state["members"]]  # Acceptance is durable.
    assert (env["store"].inbox(batch_id) / f"{prepared['run_id']}.json").is_file()  # Request not yet settled.
    env["store"].resume(batch_id)
    result = run(env, batch_id)
    assert result["status"] == "complete" and [m["id"] for m in result["members"]].count(prepared["run_id"]) == 1
    submitted = [value["run_id"] for value in env["jobs"].values()]
    assert len(submitted) == len(set(submitted)) == 9
    assert (env["store"].inbox(batch_id) / "accepted" / f"{prepared['run_id']}.json").is_file()


def test_crash_before_the_inbox_write_leaves_a_resubmittable_run(env, monkeypatch):
    batch_id = env["start"]()
    env["batch"] = batch_id
    prepared = additions.prepare_addition(env["store"], batch_id, "Third theme", 3, env["events"].append, Event())
    real = additions.atomic_json
    monkeypatch.setattr(additions, "atomic_json", lambda path, value: (_ for _ in ()).throw(OSError("disk"))
                        if "inbox" in str(path) else real(path, value))
    with pytest.raises(OSError):
        additions.submit_addition(env["store"], batch_id, Path(prepared["folder"]))
    assert not list(env["store"].inbox(batch_id).glob("*.json"))
    monkeypatch.setattr(additions, "atomic_json", real)
    additions.submit_addition(env["store"], batch_id, Path(prepared["folder"]))
    assert len(run(env, batch_id)["members"]) == 3


def test_reopen_is_one_atomic_state_write(env, monkeypatch):
    batch_id = env["start"]()
    env["batch"] = batch_id
    run(env, batch_id)
    before = env["store"].get(batch_id)
    monkeypatch.setattr(env["store"], "_save", lambda folder, state: (_ for _ in ()).throw(OSError("crash")))
    with pytest.raises(OSError):
        env["store"].reopen(batch_id)
    after = env["store"].get(batch_id)
    assert after["status"] == "complete" and after["managed_session"] == before["managed_session"]
    monkeypatch.undo()


# 7: spend cap


def test_spend_cap_rejects_addition_with_numbers(env):
    batch_id = env["start"]()
    env["batch"] = batch_id
    folder = env["store"]._folder(batch_id)
    state = env["store"].get(batch_id)
    state["max_total_usd"] = 0.01  # The session started hours ago at $1/h.
    (folder / "state.json").write_text(json.dumps(state))
    prepared = add(env, batch_id)
    result = run(env, batch_id)
    assert len(result["members"]) == 2
    rejected = json.loads((env["store"].inbox(batch_id) / "rejected" / f"{prepared['run_id']}.json").read_text())
    assert "spend cap" in rejected["reason"] and "billed so far $" in rejected["reason"]
    assert "cap $0.01" in rejected["reason"]
    assert result["status"] == "complete"  # Queued and running renders are never cancelled.


def test_configure_validates_the_spend_cap(env):
    with pytest.raises(ValueError, match="Spend cap"):
        env["store"].configure("0" * 32, gpu_choices=[], max_total_usd=-1)


# 4-6, 9, 10: warm-pod handoff, races, crash recovery, keep-warm, failure shutdown

OUTPUT = {"id": "o1", "prompt_id": "p1", "filename": "clip.mp4", "subfolder": "", "type": "output"}


class Provider:
    def __init__(self):
        self.present, self.deletes = True, []
        self.client = type("Client", (), {"close": lambda self: None})()

    def get(self, pod_id):
        return {"id": pod_id, "name": "sg-shared-gpu", "desiredStatus": "RUNNING"} if self.present else None

    def delete(self, pod_id):
        self.deletes.append(pod_id)
        self.present = False

    def deploy(self, *args, **kwargs):
        pytest.fail("no create request during handoff or adoption")

    def reconcile_exact_name(self, *args):
        pytest.fail("no create reconciliation")


class Monitor:
    def __init__(self, jobs=None):
        self.jobs = jobs or {"p1": "success"}
        self.client = type("Client", (), {"close": lambda self: None})()

    def poll(self, _jobs):
        return {"ready": True, "healthy": True, "auth_verified": True, "queue_empty": True,
                "jobs": dict(self.jobs), "outputs": {"p1": [OUTPUT]} if "p1" in self.jobs else {}}


class FakePod:
    """A pod's ComfyUI behind the authenticated gateway: /system_stats, /object_info, /queue, /history."""

    def __init__(self):
        self.running, self.pending, self.history = [], [], {}

    def finish(self, prompt_id, status="success"):
        outputs = {"9": {"videos": [{"filename": f"{prompt_id}.mp4", "subfolder": "", "type": "output"}]}}
        self.history[prompt_id] = {"status": {"status_str": status, "completed": status == "success"},
                                   "outputs": outputs if status == "success" else {}}

    def handler(self, request):
        import httpx
        path = request.url.path
        if "authorization" not in request.headers:
            return httpx.Response(401)
        if path == "/system_stats":
            return httpx.Response(200, json={"devices": [{"type": "cuda"}]})
        if path == "/object_info":
            return httpx.Response(200, json={"SaveVideo": {}})
        if path == "/queue":
            return httpx.Response(200, json={"queue_running": [[0, p] for p in self.running],
                                             "queue_pending": [[1, p] for p in self.pending]})
        if path == "/history":
            return httpx.Response(200, json=self.history)
        if path.startswith("/history/"):
            prompt_id = path.rsplit("/", 1)[1]
            return httpx.Response(200, json={prompt_id: self.history[prompt_id]} if prompt_id in self.history else {})
        return httpx.Response(404)


@pytest.fixture
def pods(env, monkeypatch, tmp_path):
    import httpx

    from simplegen_runpod import worker
    clock = [5000.0]
    monkeypatch.setattr(global_session.time, "monotonic", lambda: clock[0])
    provider = Provider()
    monkeypatch.setattr(global_session.GlobalSession, "_api", lambda self: provider)
    monkeypatch.setattr(global_session.GlobalSession, "_login", lambda self: ("user", "pass"))
    pod = FakePod()
    pod.finish("p1")  # Batch A's render, still in the pod's history.
    pod.finish("before-a")  # Something run on the pod before A took it over.
    # The real ComfyMonitor with a fake HTTP transport; never a real network call.
    monkeypatch.setattr(global_session, "ComfyMonitor", lambda *args, **kwargs: worker.ComfyMonitor(
        *args, client=httpx.Client(transport=httpx.MockTransport(pod.handler)), **kwargs))

    def warm_batch():
        """A completed batch A whose pod is in its keep-warm window."""
        batch_id = env["start"](("Warm theme",))
        env["batch"] = batch_id
        run(env, batch_id)
        folder = env["store"]._folder(batch_id)
        state = env["store"].get(batch_id)
        from simplegen_runpod.outputs import manifest
        jobs = {"j": {"prompt_id": "p1", "status": "success", "outputs": [OUTPUT]}}
        state["managed_session"].update(jobs=jobs, acknowledged_manifest=manifest(jobs), downloads_verified=True)
        (folder / "state.json").write_text(json.dumps(state))
        global_registry.register(folder)
        return batch_id, folder

    def poll(folder, monitor=None):
        state = json.loads((Path(folder) / "state.json").read_text())

        def persist():
            (Path(folder) / "state.json").write_text(json.dumps(state))
        session = global_session.GlobalSession(Settings(**state["settings"]), Path(folder), state, persist,
                                               lambda _: None, Event(), Event())
        session._provider = provider
        session._monitor = None if monitor == "real" else monitor or Monitor()
        with engine.run_lock(Path(folder)):
            return session.status()

    def owners(a_folder, b_folder):
        a = json.loads((a_folder / "state.json").read_text())["managed_session"]
        b = json.loads((b_folder / "state.json").read_text()).get("managed_session", {})
        result = []
        if not a.get("handed_off") and not a.get("retired"):
            result.append("A")
        if (b.get("adopted_from") or {}).get("sid") == a.get("sid"):
            result.append("B")
        return result
    return {"clock": clock, "provider": provider, "warm": warm_batch, "poll": poll, "owners": owners, "pod": pod}


def due_batch(env, concepts=("Next theme",)):
    batch_id = env["start"](concepts)
    return batch_id, env["store"]._folder(batch_id)


def test_due_batch_adopts_a_warm_pod_without_creating(env, pods):
    a_id, a_folder = pods["warm"]()
    b_id, b_folder = due_batch(env)
    assert env["store"].try_handoff(b_id, live_schema=lambda session: env["store"].live(None)) == a_id
    a = env["store"].get(a_id)["managed_session"]
    b = env["store"].get(b_id)["managed_session"]
    assert a["handed_off"]["to"] == b_id and "handoff_intent" not in a
    assert b["adopted_from"] == {"batch": a_id, "folder": str(a_folder.resolve()), "sid": a["sid"]}
    assert (b["pod_id"], b["name"], b["sid"]) == (a["pod_id"], a["name"], a["sid"]) and b["jobs"] == {}
    assert global_registry.listed() == [b_folder.resolve()]
    assert pods["owners"](a_folder, b_folder) == ["B"]
    # A's idle deletion never fires, however long the poller runs.
    for _ in range(60):
        pods["clock"][0] += 30
        assert pods["poll"](a_folder)["handed_off"]
    assert pods["provider"].deletes == []
    result = run(env, b_id)
    assert result["status"] == "complete" and env["world"]["allocations"] == ["pod"]  # No second pod.


@pytest.mark.parametrize("problem", ["gpu", "ceiling", "schema", "profile"])
def test_incompatible_batch_does_not_adopt_and_the_warm_pod_retires(env, pods, problem):
    a_id, a_folder = pods["warm"]()
    b_id, b_folder = due_batch(env)
    state = env["store"].get(b_id)
    live = lambda session: env["store"].live(None)
    if problem == "gpu":
        state["gpu_choices"] = [{"gpu_id": "b", "max_hourly": 2.0}, {"gpu_id": "c", "max_hourly": 2.0}]
    elif problem == "ceiling":
        state["gpu_choices"][0]["max_hourly"] = 0.5
    elif problem == "schema":
        def live(session):
            info = env["store"].live(None)
            info.pop("SaveVideo")
            return info
    else:
        state["settings"]["runpod_config_file"] = str(Path(state["settings"]["runpod_config_file"]).with_name("x.json"))
    (b_folder / "state.json").write_text(json.dumps(state))
    assert env["store"].try_handoff(b_id, live_schema=live) is None
    assert "handoff_intent" not in env["store"].get(a_id)["managed_session"]
    assert not env["store"].get(b_id).get("managed_session", {}).get("name")
    pods["poll"](a_folder)
    for _ in range(40):
        pods["clock"][0] += 10
        pods["poll"](a_folder)
    assert pods["provider"].deletes == ["pod"]  # A retires normally after its keep-warm window.
    assert pods["owners"](a_folder, b_folder) == []


def test_idle_deletion_and_handoff_race_has_one_outcome(env, pods):
    _, a_folder = pods["warm"]()
    b_id, b_folder = due_batch(env)
    live = lambda session: env["store"].live(None)
    # Interleaving 1: the poller holds A while B tries to adopt → no handoff; A retires.
    with engine.run_lock(a_folder):
        assert env["store"].try_handoff(b_id, live_schema=live) is None
    pods["poll"](a_folder)
    for _ in range(40):
        pods["clock"][0] += 10
        pods["poll"](a_folder)
    assert pods["provider"].deletes == ["pod"] and pods["owners"](a_folder, b_folder) == []
    assert env["store"].try_handoff(b_id, live_schema=live) is None  # A retired pod is never adopted.


def test_handoff_first_then_poller_cannot_delete(env, pods):
    a_id, a_folder = pods["warm"]()
    b_id, b_folder = due_batch(env)
    pods["poll"](a_folder)
    pods["clock"][0] += 290  # Nearly at the end of A's keep-warm window.
    assert env["store"].try_handoff(b_id, live_schema=lambda session: env["store"].live(None)) == a_id
    for _ in range(10):
        pods["clock"][0] += 10
        pods["poll"](a_folder)
    assert pods["provider"].deletes == [] and pods["owners"](a_folder, b_folder) == ["B"]


@pytest.mark.parametrize("crash_after", [1, 2, 3])
def test_crash_at_each_handoff_step_recovers_to_one_owner(env, pods, crash_after):
    a_id, a_folder = pods["warm"]()
    _, b_folder = due_batch(env)
    with pytest.raises(RuntimeError, match="simulated crash"):
        additions.perform_handoff(a_folder, b_folder, live_schema=lambda session: env["store"].live(None),
                                  crash_after=crash_after)
    # While the intent is open, the poller never deletes.
    for _ in range(40):
        pods["clock"][0] += 10
        pods["poll"](a_folder)
    assert pods["provider"].deletes == []
    outcome = additions.recover_handoff(a_folder)
    owners = pods["owners"](a_folder, b_folder)
    assert len(owners) == 1
    if crash_after == 1:
        assert outcome == "rolled_back" and owners == ["A"]
        assert global_registry.listed() == [a_folder.resolve()]
        assert "handoff_orphaned" in [e["event"] for e in env["store"].get(a_id)["managed_session"]["alert_log"]]
    else:
        assert outcome == "finished" and owners == ["B"]
        assert global_registry.listed() == [b_folder.resolve()]
    assert additions.recover_handoff(a_folder) == "none"


def test_keep_warm_minutes_for_new_sessions_old_sessions_keep_five(env, pods):
    for minutes, expected in ((20, 1200), (None, 300)):
        a_id, a_folder = pods["warm"]()
        state = env["store"].get(a_id)
        if minutes:
            state["managed_session"]["frozen_profile"] = {"keep_warm_minutes": minutes}
        (a_folder / "state.json").write_text(json.dumps(state))
        pods["provider"].present, pods["provider"].deletes = True, []
        assert pods["poll"](a_folder)["idle_remaining_seconds"] == expected
        for _ in range(expected // 10 - 1):
            pods["clock"][0] += 10
            pods["poll"](a_folder)
        assert pods["provider"].deletes == []
        pods["clock"][0] += 10
        pods["poll"](a_folder)
        assert pods["provider"].deletes == ["pod"]
    for bad in (4, 61):
        with pytest.raises(ValueError, match="Keep-warm"):
            Settings(keep_warm_minutes=bad).validate(require_folders=False)


@pytest.mark.parametrize("kind", ["adopted", "reopened"])
def test_failure_shutdown_applies_to_adopted_and_reopened_sessions(env, pods, kind):
    from simplegen import cost_safety
    a_id, a_folder = pods["warm"]()
    if kind == "adopted":
        b_id, folder = due_batch(env)
        assert env["store"].try_handoff(b_id, live_schema=lambda session: env["store"].live(None)) == a_id
    else:
        folder = a_folder
        env["store"].reopen(a_id)
    state = json.loads((folder / "state.json").read_text())
    state["managed_session"]["jobs"] = {"j": {"prompt_id": "p9", "status": "running", "outputs": []}}
    assert cost_safety.record_owner_failure(state, Settings(**state["settings"]), "render failed", "run")
    (folder / "state.json").write_text(json.dumps(state))
    monitor = Monitor({"p9": "error"})
    pods["poll"](folder, monitor)
    for _ in range(90):
        pods["clock"][0] += 10
        pods["poll"](folder, monitor)
    saved = json.loads((folder / "state.json").read_text())["managed_session"]
    assert pods["provider"].deletes == ["pod"] and saved["delete_intent"] == "failure_idle"


def test_actual_spend_over_the_cap_alerts_and_never_cancels(env, pods):
    a_id, a_folder = pods["warm"]()
    state = env["store"].get(a_id)
    state["max_total_usd"] = 0.5
    state["managed_session"]["downloads_verified"] = False  # Still working.
    (a_folder / "state.json").write_text(json.dumps(state))
    for _ in range(5):
        pods["clock"][0] += 10
        pods["poll"](a_folder)
    saved = env["store"].get(a_id)["managed_session"]
    assert [e["event"] for e in saved["alert_log"]].count("spend_cap_exceeded") == 1
    assert pods["provider"].deletes == [] and saved["jobs"]["j"]["status"] == "success"


# 8: drafts and separate slots (Qt offscreen)


@pytest.fixture
def window(tmp_path, monkeypatch, _no_desktop_audio):
    import threading
    from types import SimpleNamespace

    from PySide6.QtWidgets import QApplication
    from test_ui import FakeWorker

    from simplegen import ui
    monkeypatch.setenv("SIMPLEGEN_HOME", str(tmp_path / "home"))
    app = QApplication.instance() or QApplication([])
    (tmp_path / "project").mkdir()
    (tmp_path / "templates").mkdir()
    (tmp_path / "templates" / "workflow.json").write_text("{}")
    local = Settings(project_folder=str(tmp_path / "project"), template_folder=str(tmp_path / "templates"),
                     template_file="workflow.json", output_folder=str(tmp_path / "out"))
    main = ui.MainWindow(local)
    for timer in (main.batch_timer, main.schedule_timer, main.cloud_timer, main.global_timer):
        timer.stop()
    monkeypatch.setattr(ui, "GenerationWorker", FakeWorker)
    passed = {"status": "pass", "target": "local", "checked_at": "2026-09-28T00:00:00+00:00", "checks": []}
    monkeypatch.setattr(ui, "run_preflight", lambda settings, template_path=None, **_: passed)
    batch_settings = Settings(managed_runpod=True, global_desktop=True, cloud=True)
    batch = SimpleNamespace(cancel_event=threading.Event(), detach_event=threading.Event())
    main.batch_worker, main.batch_worker_folder = batch, tmp_path / "batch-folder"
    main.batch_line = "Generating"

    def settle():
        import time
        end = time.time() + 5
        while main.preflight_worker is not None and time.time() < end:
            app.processEvents()
            time.sleep(0.01)
    yield SimpleNamespace(main=main, app=app, batch=batch, batch_settings=batch_settings, settle=settle, ui=ui)
    main.batch_worker = None
    main.worker = None
    main.close()
    main.deleteLater()
    app.processEvents()


def test_switching_the_draft_never_touches_the_running_batch(window, monkeypatch):
    main = window.main
    folder = main.batch_worker_folder
    main.select_location(True)
    main.select_location(False)
    main.select_location(True)
    assert main.batch_worker is window.batch and main.batch_worker_folder == folder
    skips = []
    monkeypatch.setattr("simplegen.global_registry.poll_registered", lambda skip=None: skips.append(skip) or {})
    main.poll_registered_global_sessions()
    window.settle()
    import time
    end = time.time() + 5
    while main.global_worker is not None and time.time() < end:
        window.app.processEvents()
        time.sleep(0.01)
    assert skips and folder in skips[0]  # Monitoring of the batch stays with its worker.
    assert "Cloud batch" in main.jobs_status.text()
    monkeypatch.setattr(main, "confirm_shutdown", lambda batch=False: True)
    main.cloud_action("cancel")
    assert window.batch.cancel_event.is_set()  # Shutdown still targets the batch.


def test_local_run_and_local_schedule_start_while_a_batch_runs(window, monkeypatch):
    main = window.main
    main.select_location(False)
    main.concept.setPlainText("A local garden")
    main.start_generation()
    window.settle()
    assert main.worker is not None and main.worker.started and not main.worker.settings.cloud
    local = main.worker
    main.start_generation()  # A second local run is still refused.
    window.settle()
    assert main.worker is local and "already in progress" in main.status.text()
    main.concept.setPlainText("Queued garden")
    main.queue_locally()
    window.settle()
    records = main.scheduler.records()
    assert [r["concept"] for r in records] == ["Queued garden"] and records[0]["status"] == "scheduled"
    main.worker = None
    notes = []
    monkeypatch.setattr(main.scheduler, "defer", lambda identity, message: notes.append(message))
    from datetime import UTC, datetime
    main.scheduler.clock = lambda: datetime.now(UTC).timestamp() + 5
    monkeypatch.setattr(window.ui, "local_queue_empty", lambda settings: True)
    main.poll_schedules()
    assert not any("current simpleGen operation" in note for note in notes)
    assert main.schedule_probe is not None  # Local dispatch does not wait for the cloud batch.
    import time
    end = time.time() + 5
    while main.schedule_probe is not None and time.time() < end:
        window.app.processEvents()
        time.sleep(0.01)


def test_add_to_running_batch_from_generate_shows_projection(window, monkeypatch, tmp_path):
    main = window.main
    calls = {}
    store = type("Store", (), {"inbox": lambda self, batch_id: tmp_path / "inbox"})()
    monkeypatch.setattr(main, "running_batches", lambda: [(store, {"id": "b" * 32, "status": "running"})])
    projection = {"hourly_usd": 1.0, "billed_usd": 2.0, "average_render_seconds": 60.0, "remaining_renders": 1,
                  "remaining_usd": 0.02, "keep_warm_usd": 0.08, "extra_renders": 1, "extra_usd": 0.02,
                  "total_usd": 2.12, "lower_bound_usd": 2.12, "cap_usd": 5.0}
    monkeypatch.setattr(additions, "prepare_addition", lambda *a, **k: calls.setdefault("prepare", (a, k)) and {
        "folder": str(tmp_path / "run"), "run_id": "r", "batch_id": "b" * 32, "projection": projection})
    monkeypatch.setattr(additions, "submit_addition", lambda s, b, f: calls.setdefault("submit", (b, f)))
    main.settings = window.batch_settings
    main.concept.setPlainText("An addition")
    monkeypatch.setattr(main, "collect_settings", lambda: window.batch_settings)
    shown = []
    main.add_draft_to_running_batch(confirm=lambda text: shown.append(text) or True)
    import time
    end = time.time() + 5
    while main.addition_worker is not None and time.time() < end:
        window.app.processEvents()
        time.sleep(0.01)
    assert calls["prepare"][0][2] == "An addition" and calls["submit"][0] == "b" * 32
    assert "this addition $0.02" in shown[0] and "projected total $2.12 (cap $5.00)" in shown[0]



# Review follow-up: an adopted pod's history is inherited, not "unregistered" (real ComfyMonitor)


def adopt(env, pods):
    a_id, _ = pods["warm"]()
    b_id, b_folder = due_batch(env)
    assert env["store"].try_handoff(b_id, live_schema=lambda session: env["store"].live(None)) == a_id
    state = json.loads((b_folder / "state.json").read_text())
    assert set(state["managed_session"]["inherited_prompt_ids"]) >= {"p1", "before-a"}
    return b_folder, state


def give_job(folder, state, status="running"):
    job = {"prompt_id": "q1", "status": status, "outputs": []}
    state["managed_session"]["jobs"] = {"b1": job}
    (folder / "state.json").write_text(json.dumps(state))


def acknowledge(folder):
    from simplegen_runpod.outputs import manifest
    state = json.loads((folder / "state.json").read_text())
    saved = state["managed_session"]
    saved.update(downloads_verified=True, acknowledged_manifest=manifest(saved["jobs"]))
    state["status"] = "complete"
    (folder / "state.json").write_text(json.dumps(state))


def test_adopted_batch_completes_and_retires_via_keep_warm(env, pods):
    b_folder, state = adopt(env, pods)
    give_job(b_folder, state)
    pods["pod"].finish("q1")
    pods["poll"](b_folder, "real")
    acknowledge(b_folder)
    pods["poll"](b_folder, "real")
    for _ in range(31):
        pods["clock"][0] += 10
        pods["poll"](b_folder, "real")
    saved = json.loads((b_folder / "state.json").read_text())["managed_session"]
    assert not saved.get("unregistered_jobs_seen")  # A's prompts are inherited, not foreign.
    assert pods["provider"].deletes == ["pod"] and saved["retired"] is True


def test_adopted_batch_failure_retires_via_failure_shutdown(env, pods):
    from simplegen import cost_safety
    b_folder, state = adopt(env, pods)
    give_job(b_folder, state)
    pods["pod"].finish("q1", status="error")
    assert cost_safety.record_owner_failure(state, Settings(**state["settings"]), "render failed", "batch")
    state["managed_session"]["recovery_completed_at"] = "2026-09-28T01:00:00+00:00"
    (b_folder / "state.json").write_text(json.dumps(state))
    pods["poll"](b_folder, "real")
    for _ in range(91):
        pods["clock"][0] += 10
        pods["poll"](b_folder, "real")
    saved = json.loads((b_folder / "state.json").read_text())["managed_session"]
    assert not saved.get("unregistered_jobs_seen")
    assert pods["provider"].deletes == ["pod"] and saved["delete_intent"] == "failure_idle"


def test_foreign_prompt_after_adoption_still_blocks_retirement(env, pods):
    b_folder, state = adopt(env, pods)
    give_job(b_folder, state)
    pods["pod"].finish("q1")
    pods["pod"].finish("someone-else")  # Neither A's nor B's: appeared after adoption.
    pods["poll"](b_folder, "real")
    acknowledge(b_folder)
    for _ in range(60):
        pods["clock"][0] += 10
        pods["poll"](b_folder, "real")
    saved = json.loads((b_folder / "state.json").read_text())["managed_session"]
    assert saved["unregistered_jobs_seen"] is True and pods["provider"].deletes == []


def test_unreadable_pod_history_means_no_handoff(env, pods, monkeypatch):
    _, a_folder = pods["warm"]()
    b_id, b_folder = due_batch(env)
    monkeypatch.setattr(pods["pod"], "history", None)  # /history returns null.
    assert env["store"].try_handoff(b_id, live_schema=lambda session: env["store"].live(None)) is None
    assert pods["owners"](a_folder, b_folder) == ["A"]


def test_reopened_batch_with_the_real_monitor_retires(env, pods):
    a_id, a_folder = pods["warm"]()
    env["store"].reopen(a_id)
    state = json.loads((a_folder / "state.json").read_text())
    state["managed_session"]["jobs"]["j2"] = {"prompt_id": "p2", "status": "running", "outputs": []}
    (a_folder / "state.json").write_text(json.dumps(state))
    pods["pod"].history.pop("before-a")  # A's own pod: only A's prompts.
    pods["pod"].finish("p2")
    pods["poll"](a_folder, "real")
    acknowledge(a_folder)
    for _ in range(31):
        pods["clock"][0] += 10
        pods["poll"](a_folder, "real")
    saved = json.loads((a_folder / "state.json").read_text())["managed_session"]
    assert not saved.get("unregistered_jobs_seen") and pods["provider"].deletes == ["pod"]


def test_monitor_default_is_unchanged_without_inherited_ids():
    import httpx

    from simplegen_runpod import worker
    pod = FakePod()
    pod.finish("mine")
    pod.finish("foreign")
    client = httpx.Client(transport=httpx.MockTransport(pod.handler))
    jobs = {"j": {"prompt_id": "mine", "status": "running", "outputs": []}}
    assert worker.ComfyMonitor("pod", "u", "p", [], client=client).poll(jobs)["unregistered_jobs"] is True
    client = httpx.Client(transport=httpx.MockTransport(pod.handler))
    assert worker.ComfyMonitor("pod", "u", "p", [], client=client,
                               extra_known=["foreign"]).poll(jobs)["unregistered_jobs"] is False


def test_reopened_batch_records_only_its_portion_in_cost_history(env):
    import csv
    from datetime import UTC, datetime

    from simplegen.resources import _history_path, record_run
    batch_id = env["start"]()
    env["batch"] = batch_id
    run(env, batch_id)
    prepared = add(env, batch_id)
    env["store"].dispatch_due(env["events"].append, Event())  # Reopens, accepts, completes again.
    state = env["store"].get(batch_id)
    reopened_at = datetime.fromisoformat(state["reopen"]["at"])
    ended = datetime(2030, 1, 1, tzinfo=UTC)
    state["managed_session"]["session_completed_at"] = ended.isoformat()
    record_run(env["store"]._folder(batch_id), state)  # Retirement enriches the reopened portion only.
    rows = {row["run_id"]: row for row in csv.DictReader(_history_path().open())}
    first, second = rows[batch_id], rows[f"{batch_id}-reopen1"]
    started = datetime.fromisoformat(state["managed_session"]["session_started_at"])
    assert float(first["session_seconds"]) == pytest.approx((reopened_at - started).total_seconds(), abs=2)
    assert float(second["session_seconds"]) == pytest.approx((ended - reopened_at).total_seconds(), abs=2)
    assert int(second["quantity"]) == 3 and int(first["quantity"]) == 6  # Portion renders only.
    assert prepared["run_id"] in [m["id"] for m in state["members"]]


def test_main_cancel_never_shuts_down_the_batch(window, monkeypatch):
    main = window.main
    asked = []
    monkeypatch.setattr(main, "confirm_shutdown", lambda batch=False: asked.append(batch) or True)
    main.update_jobs_status()
    assert not main.stop_batch_button.isHidden() and main.cancel_local_button.isHidden()
    main.cancel_generation()  # No local run: the batch is untouched, nothing asked.
    assert not window.batch.cancel_event.is_set() and asked == []
    main.select_location(False)
    main.concept.setPlainText("A local garden")
    main.start_generation()
    window.settle()
    main.update_jobs_status()
    assert not main.cancel_local_button.isHidden()
    main.cancel_generation()
    assert main.worker.cancel_event.is_set() and not window.batch.cancel_event.is_set()
    main.stop_batch_button.click()  # Only the explicit control shuts the batch down (after asking).
    assert window.batch.cancel_event.is_set() and asked == [True]


def test_cap_with_unknown_projection_requires_explicit_confirmation():
    from simplegen import cost_safety
    projection = {"hourly_usd": 2.0, "billed_usd": 3.5, "average_render_seconds": None, "remaining_renders": 4,
                  "remaining_usd": None, "keep_warm_usd": 0.17, "extra_renders": 1, "extra_usd": None,
                  "total_usd": None, "lower_bound_usd": 3.67, "cap_usd": 10.0}
    text = cost_safety.confirmation_text(projection)
    assert "render time unknown" in text and "Billed so far: $3.50" in text and "$10.00" in text
    assert not cost_safety.over_cap(projection)  # Unknown is confirmed explicitly, never refused.
    assert "cannot be projected" not in cost_safety.confirmation_text({**projection, "cap_usd": None})
