"""Batch decisions and durable owner behavior; never contact a paid provider."""
from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import Event

import pytest
from test_variations import sequence  # noqa: F401
from test_workflows import info, workflow  # noqa: F401

from simplegen import batches
from simplegen.settings import Settings, atomic_json


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv("SIMPLEGEN_HOME", str(tmp_path / "simplegen-home"))


def _gpu(gpu_id, price, available=True):
    centers = [{"id": "US-TEST", "availability": "HIGH" if available else "UNAVAILABLE"}]
    return {"gpu_id": gpu_id, "gpu_count": 1, "hourly_usd": price,
            "data_centers": centers, "data_center": "US-TEST", "availability": centers[0]["availability"]}


def _approved():
    return [{"gpu_id": "a", "max_hourly": 2.0, "volume_id": "vol"},
            {"gpu_id": "b", "max_hourly": 2.0, "volume_id": "vol"}]


def test_priority_and_price_caps():
    approved = _approved()
    catalog = [_gpu("a", 1.0, False), _gpu("b", 1.5)]
    choice, reason = batches.select_gpu(approved, catalog, "low")
    assert choice is None and "unavailable" in reason.lower()
    choice, reason = batches.select_gpu(approved, catalog, "high")
    assert reason is None and choice["gpu_id"] == "b"
    catalog = [_gpu("a", 1.0, False), _gpu("b", 1.0)]
    choice, _ = batches.select_gpu(approved, catalog, "low")
    assert choice["gpu_id"] == "b"
    catalog[0] = _gpu("a", 1.0)
    choice, _ = batches.select_gpu(approved, catalog, "low")
    assert choice["gpu_id"] == "a"
    approved[0]["max_hourly"] = 0.5
    choice, _ = batches.select_gpu(approved, catalog, "high")
    assert choice["gpu_id"] == "b"
    choice, reason = batches.select_gpu(approved, [catalog[1]], "low")
    assert choice is None and "absent" in reason


def test_catalog_excludes_nonfinite_prices_and_memory(tmp_path):
    from importlib.resources import files

    profile = tmp_path / "profile.json"
    profile.write_text(json.dumps({**json.loads(files("simplegen").joinpath("data/global-runpod-profile.json").read_text()),
                     "template_id": "tmpl-test", "volume_id": "vol-test"}))
    class Provider:
        def _request(self, *_args, **_kwargs):
            return {"gpus": [
                {"id": "good", "secure": True, "memory": 96, "price": {"secure": 1.0}},
                {"id": "bad-price", "secure": True, "memory": 96, "price": {"secure": float("inf")}},
                {"id": "bad-memory", "secure": True, "memory": float("nan"), "price": {"secure": 1.0}},
            ]}
    result = batches._catalog(Settings(runpod_config_file=str(profile)), provider=Provider())
    assert [row["gpu_id"] for row in result] == ["good"]


def test_global_catalog_counts_unique_available_locations_and_selects_worldwide(tmp_path):
    from importlib.resources import files

    profile = tmp_path / "profile.json"
    profile.write_text(json.dumps({**json.loads(files("simplegen").joinpath("data/global-runpod-profile.json").read_text()),
                     "template_id": "tmpl-test", "volume_id": "vol-test"}))
    class Provider:
        def _request(self, _method, path, *, catalog):
            assert catalog and "minCudaVersion=13.0" in path
            assert "countryCodes" not in path
            return {"gpus": [{"id": "worldwide", "secure": True, "memory": 96,
                              "price": {"secure": 1.0}, "dataCenters": [
                                  {"id": "EU-FR-1", "availability": "HIGH"},
                                  {"id": "US-TX-1", "availability": "LOW"},
                                  {"id": "EU-FR-1", "availability": "LOW"},
                                  {"id": "AP-IN-1", "availability": "MEDIUM"},
                                  {"id": "", "availability": "HIGH"},
                                  {"id": "US-NC-1", "availability": "NONE"}]},
                             {"id": "ap-only", "secure": True, "memory": 80,
                              "price": {"secure": 2.0}, "dataCenters": [
                                  {"id": "AP-IN-1", "availability": "LOW"}]}]}
    catalog = batches._catalog(Settings(runpod_config_file=str(profile)), provider=Provider())
    row = catalog[0]
    assert row["availability"] == "HIGH"
    assert row["available_data_centers"] == ["US-TX-1", "EU-FR-1", "AP-IN-1"]
    assert row["available_location_count"] == 3
    assert row["placement_scope"] == "global"
    approved = [{"gpu_id": "worldwide", "max_hourly": 3.0, "volume_id": row["global_volume_id"]},
                {"gpu_id": "ap-only", "max_hourly": 3.0, "volume_id": row["global_volume_id"]}]
    choice, reason = batches.select_gpu(approved, catalog, "high")
    assert reason is None and choice["gpu_id"] == "worldwide"
    assert choice["available_location_count"] == 3


def test_batch_catalog_is_fresh_and_read_only_before_preparation(tmp_path, monkeypatch):
    store = batches.BatchStore(tmp_path)
    batch = store.create()
    _member(store, batch, tmp_path, 1)
    store._save(Path(batch["folder"]), batch)
    seen = []
    def catalog(settings):
        seen.append(settings.runpod_config_file)
        return [_gpu("expensive", 5.25)]
    monkeypatch.setattr(batches, "_catalog", catalog)
    assert store.catalog(batch["id"])[0]["hourly_usd"] == 5.25
    assert store.catalog(batch["id"])[0]["hourly_usd"] == 5.25
    assert len(seen) == 2
    assert store.get(batch["id"])["status"] == "draft"
    assert store.get(batch["id"])["gpu_options"] == []


def test_explicit_batch_ceiling_allows_assessed_gpu_above_profile_default(tmp_path):
    from importlib.resources import files
    store = batches.BatchStore(tmp_path)
    batch = store.create()
    _member(store, batch, tmp_path, 1)
    profile = tmp_path / "profile.json"
    profile.write_text(json.dumps({**json.loads(files("simplegen").joinpath("data/global-runpod-profile.json").read_text()),
                     "template_id": "tmpl-test", "volume_id": "vol-test"}))
    batch.update(status="prepared", gpu_options=[_gpu("a", 4.0), _gpu("b", 5.0)])
    store._save(Path(batch["folder"]), batch)
    selected = [{"gpu_id": "a", "max_hourly": 4.5}, {"gpu_id": "b", "max_hourly": 5.5}]
    with pytest.raises(ValueError, match="hourly ceiling"):
        store.configure(batch["id"], gpu_choices=selected)
    configured = store.configure(batch["id"], gpu_choices=selected, max_hourly=5.5)
    assert configured["max_hourly"] == 5.5
    assert [item["max_hourly"] for item in configured["gpu_choices"]] == [4.5, 5.5]
    with pytest.raises(ValueError, match="hourly ceiling"):
        store.configure(batch["id"], gpu_choices=selected, max_hourly=5.0)
    with pytest.raises(ValueError, match="positive finite"):
        store.configure(batch["id"], gpu_choices=selected, max_hourly=float("inf"))
    assert store.get(batch["id"])["max_hourly"] == 5.5


def _member(store, state, tmp_path, index, quantity=1):
    target = tmp_path / f"run-{index}"
    target.mkdir()
    (tmp_path / "profile.json").write_text("{}")
    (tmp_path / "schema.json").write_text("{}")
    settings = Settings(output_folder=str(tmp_path), managed_runpod=True, global_desktop=True,
                        runpod_config_file=str(tmp_path / "profile.json"),
                        runpod_schema_file=str(tmp_path / "schema.json"), quantity=quantity)
    child = {"schema_version": 4, "run_id": f"run-{index}", "concept": f"concept {index}",
             "status": "prepared", "settings": settings.public_dict(), "batch_parent": state["id"],
             "downloads": {}}
    state.setdefault("settings", settings.public_dict())
    atomic_json(target / "state.json", child)
    for name in ("workflow.json", "plan.json", "template.original.json"):
        atomic_json(target / name, {"prepared": True})
    state["members"].append({"id": child["run_id"], "folder": str(target),
                             "concept": child["concept"], "status": "prepared", "revision": 1,
                             "prepared_fingerprint": batches._fingerprint(target)})
    return target


def test_due_time_prevents_provider_access_and_wait_is_durable(tmp_path, monkeypatch):
    store = batches.BatchStore(tmp_path)
    state = store.create()
    _member(store, state, tmp_path, 1)
    settings = Settings(**json.loads((tmp_path / "run-1" / "state.json").read_text())["settings"])
    state.update(status="armed", settings=settings.public_dict(), gpu_choices=_approved(),
                 due_at=(datetime.now(UTC) + timedelta(hours=1)).isoformat(), next_check_at=None)
    store._save(Path(state["folder"]), state)
    monkeypatch.setattr(batches, "_catalog", lambda *_: pytest.fail("catalog queried before due time"))
    assert store.run_batch(state["id"], lambda _: None, Event())["status"] == "armed"
    state["due_at"] = None
    state["next_check_at"] = datetime.now(UTC).isoformat()
    store._save(Path(state["folder"]), state)
    monkeypatch.setattr(batches, "_catalog", lambda *_: [_gpu("a", 1, False), _gpu("b", 1.5, False)])
    waited = store.run_batch(state["id"], lambda _: None, Event())
    assert waited["status"] == "waiting" and waited["waiting_reason"]
    assert datetime.fromisoformat(waited["next_check_at"]) > datetime.now(UTC)
    assert store.dispatch_due(lambda _: None, Event()) == []


def test_sequential_members_one_session_one_seal(tmp_path, monkeypatch):
    store = batches.BatchStore(tmp_path)
    state = store.create()
    _member(store, state, tmp_path, 1)
    _member(store, state, tmp_path, 2, quantity=2)
    settings = Settings(**json.loads((tmp_path / "run-1" / "state.json").read_text())["settings"])
    state.update(status="armed", settings=settings.public_dict(), gpu_choices=_approved(),
                 gpu_options=[_gpu("a", 1), _gpu("b", 1.5)],
                 next_check_at=datetime.now(UTC).isoformat())
    store._save(Path(state["folder"]), state)
    monkeypatch.setattr(batches, "_catalog", lambda *_: [_gpu("a", 1), _gpu("b", 1.5)])
    calls = []
    class Session:
        def connect(self, choice):
            calls.append(("connect", choice["gpu_id"]))
        def complete(self, receipts):
            calls.append(("seal", len(receipts)))
        def close(self):
            pass
    from simplegen import global_session, variations
    monkeypatch.setattr(global_session, "session_for", lambda *args: Session())
    def one(settings, concept, target, child, callback, cancel, detach, *, sequence, batch_phase):  # noqa: F811
        calls.append((batch_phase, child["run_id"], sequence["folder"]))
        if batch_phase == "run":
            child["result"] = {"final_path": "final.mp4"}
            atomic_json(target / "state.json", child)
        return {"final_path": str(target / "final.mp4")}
    def qty(settings, concept, target, child, callback, cancel, detach, chooser, *, batch, batch_phase):
        calls.append((batch_phase, child["run_id"], batch["folder"]))
        if batch_phase == "stage":
            child["sequence"] = {"children": [{"folder": "variation-001"}]}
            variation = target / "variation-001"
            variation.mkdir()
            atomic_json(variation / "state.json", {"downloads": {}})
        if batch_phase == "run":
            child["result"] = {"final_path": "final.mp4"}
            atomic_json(target / "state.json", child)
        return {"final_path": str(target / "final.mp4")}
    monkeypatch.setattr(batches, "_run_locked", one)
    monkeypatch.setattr(variations, "run_sequence", qty)
    monkeypatch.setattr(batches, "_verified_receipts", lambda *_: [{"path": "ok"}])
    done = store.run_batch(state["id"], lambda _: None, Event())
    assert done["status"] == "complete"
    assert [x[0] for x in calls] == ["connect", "stage", "stage", "submit", "submit", "run", "run", "seal"]
    assert calls[-1] == ("seal", 2)


def test_failed_batch_requires_explicit_resume(tmp_path, monkeypatch):
    store = batches.BatchStore(tmp_path)
    state = store.create()
    state["status"] = "failed"
    state["managed_session"] = {"create_intent": True, "name": "owned"}
    store._save(Path(state["folder"]), state)
    assert store.dispatch_due(lambda _: None, Event()) == []
    resumed = store.resume(state["id"])
    assert resumed["status"] == "armed"


def test_edit_keeps_prior_revision_and_invalidates_approval(tmp_path, monkeypatch):
    store = batches.BatchStore(tmp_path)
    state = store.create()
    target = _member(store, state, tmp_path, 1)
    state.update(status="prepared", gpu_options=[_gpu("a", 1), _gpu("b", 2)],
                 gpu_choices=_approved(), prepared_at=datetime.now(UTC).isoformat())
    store._save(Path(state["folder"]), state)
    from simplegen import engine
    def create(settings, concept):
        newer = tmp_path / "revision-2"
        newer.mkdir()
        child = {"run_id": "new", "concept": concept, "settings": settings.public_dict()}
        atomic_json(newer / "state.json", child)
        return newer, child
    monkeypatch.setattr(engine, "create_run", create)
    changed = store.edit(state["id"], "run-1", "revised concept")
    assert changed["status"] == "draft" and changed["gpu_choices"] == []
    assert changed["revisions"][0]["folder"] == str(target)
    assert changed["members"][0]["id"] == "new"
    assert json.loads((target / "state.json").read_text())["concept"] == "concept 1"


def test_cross_process_style_lock_blocks_mutation(tmp_path):
    from simplegen.engine import run_lock
    store = batches.BatchStore(tmp_path)
    state = store.create()
    with run_lock(Path(state["folder"])), pytest.raises(RuntimeError, match="already running"):
        store.pause(state["id"])


def test_catalog_error_waits_without_allocation(tmp_path, monkeypatch):
    store = batches.BatchStore(tmp_path)
    state = store.create()
    _member(store, state, tmp_path, 1)
    settings = Settings(**json.loads((tmp_path / "run-1" / "state.json").read_text())["settings"])
    state.update(status="armed", settings=settings.public_dict(), gpu_choices=_approved(),
                 next_check_at=datetime.now(UTC).isoformat())
    store._save(Path(state["folder"]), state)
    monkeypatch.setattr(batches, "_catalog", lambda *_: (_ for _ in ()).throw(OSError("network down")))
    waited = store.run_batch(state["id"], lambda _: None, Event())
    assert waited["status"] == "waiting"
    assert "network down" in waited["waiting_reason"]
    assert datetime.fromisoformat(waited["next_check_at"]) > datetime.now(UTC)


def test_changed_preparation_stops_automatic_dispatch(tmp_path, monkeypatch):
    store = batches.BatchStore(tmp_path)
    state = store.create()
    target = _member(store, state, tmp_path, 1)
    state.update(status="armed", gpu_choices=_approved(), gpu_options=[_gpu("a", 1), _gpu("b", 2)],
                 next_check_at=datetime.now(UTC).isoformat())
    store._save(Path(state["folder"]), state)
    atomic_json(target / "workflow.json", {"tampered": True})
    monkeypatch.setattr(batches, "_catalog", lambda *_: pytest.fail("catalog called after stale approval"))
    with pytest.raises(ValueError, match="changed"):
        store.run_batch(state["id"], lambda _: None, Event())
    assert store.get(state["id"])["status"] == "failed"
    assert store.dispatch_due(lambda _: None, Event()) == []


def test_submitted_qty_parent_cannot_be_attached(tmp_path):
    store = batches.BatchStore(tmp_path)
    state = store.create()
    target = tmp_path / "prior-sequence"
    child_folder = target / "variation-001"
    child_folder.mkdir(parents=True)
    settings = Settings(managed_runpod=True, global_desktop=True)
    atomic_json(target / "state.json", {"run_id": "prior", "concept": "prior",
                "settings": settings.public_dict(), "sequence": {"children": [{"folder": "variation-001"}]}})
    atomic_json(child_folder / "state.json", {"prompt_id": "already-submitted"})
    with pytest.raises(ValueError, match="variation is already submitted"):
        store.add(state["id"], target)
    assert store.get(state["id"])["members"] == []


def test_explicit_cancel_shuts_shared_session_once(tmp_path, monkeypatch):
    store = batches.BatchStore(tmp_path)
    state = store.create()
    _member(store, state, tmp_path, 1)
    state.update(status="armed", gpu_choices=_approved(), gpu_options=[_gpu("a", 1), _gpu("b", 2)],
                 next_check_at=datetime.now(UTC).isoformat())
    store._save(Path(state["folder"]), state)
    monkeypatch.setattr(batches, "_catalog", lambda *_: [_gpu("a", 1), _gpu("b", 2)])
    cancelled = Event()
    shutdowns = []
    class Session:
        def __init__(self, settings, folder, owner, persist, *args):
            self.owner, self.persist = owner, persist
        def connect(self, choice):
            self.owner["managed_session"].update(sid="owned", name="pod", create_intent=True)
            self.persist()
        def shutdown(self):
            shutdowns.append(True)
        def close(self):
            pass
    from simplegen import global_session
    monkeypatch.setattr(global_session, "session_for", Session)
    def interrupted(*args, **kwargs):
        cancelled.set()
        raise RuntimeError("user cancelled")
    monkeypatch.setattr(batches, "_run_locked", interrupted)
    with pytest.raises(RuntimeError, match="cancelled"):
        store.run_batch(state["id"], lambda _: None, cancelled)
    assert shutdowns == [True]
    assert store.get(state["id"])["status"] == "cancelled"


def test_registered_roots_survive_output_folder_change(tmp_path):
    first = batches.BatchStore(tmp_path / "first")
    second = batches.BatchStore(tmp_path / "second")
    first.create()
    second.create()
    assert [s.root for s in batches.registered_stores()] == [first.root, second.root]


def test_add_owns_child_and_pending_add_recovers(tmp_path, monkeypatch):
    store = batches.BatchStore(tmp_path)
    state = store.create()
    target = tmp_path / "created-run"
    target.mkdir()
    settings = Settings(managed_runpod=True, global_desktop=True,
                        runpod_config_file=str(tmp_path / "profile.json"))
    child = {"run_id": "child", "concept": "one", "settings": settings.public_dict(),
             "status": "created"}
    atomic_json(target / "state.json", child)
    added = store.add(state["id"], target)
    assert added["members"][0]["id"] == "child"
    assert json.loads((target / "state.json").read_text())["batch_parent"] == state["id"]
    with pytest.raises(ValueError, match="already"):
        batches.BatchStore(tmp_path / "other").add(
            batches.BatchStore(tmp_path / "other").create()["id"], target)

    # Simulate interruption between durable membership intent and final commit.
    unfinished = store.create()
    second = tmp_path / "pending-run"
    second.mkdir()
    atomic_json(second / "state.json", {**child, "run_id": "pending", "batch_parent": unfinished["id"]})
    unfinished["pending_add"] = {"id": "pending", "folder": str(second)}
    store._save(Path(unfinished["folder"]), unfinished)
    with pytest.raises((ValueError, RuntimeError, OSError)):
        # Recovery adds the member, then normal preparation fails on absent
        # test workflow settings rather than losing ownership.
        store.prepare(unfinished["id"], lambda _: None, Event())
    recovered = store.get(unfinished["id"])
    assert recovered["members"][0]["id"] == "pending"
    assert "pending_add" not in recovered


def test_prepare_qty_validates_all_workflows_without_gpu_agent(sequence, info, monkeypatch):  # noqa: F811
    from importlib.resources import files

    from simplegen import engine, global_session, resources
    from simplegen_runpod.global_gpu import GlobalGPUProfile

    _, agent_calls, jobs, _trace, events, _fault, detach, _, root, _ = sequence
    profile = root / "global-profile.json"
    profile.write_text(json.dumps({**json.loads(files("simplegen").joinpath("data/global-runpod-profile.json").read_text()),
                     "template_id": "tmpl-test", "volume_id": "vol-test"}))
    settings = Settings(project_folder=str(root / "assets"), template_folder=str(root / "templates"),
                        template_file="template.json", output_folder=str(root / "runs"), quantity=3,
                        duration=4, managed_runpod=True, global_desktop=True,
                        runpod_config_file=str(profile))
    store = batches.BatchStore(settings.output_folder)
    batch = store.create()
    target, _ = engine.create_run(settings, "Prepared concept")
    store.add(batch["id"], target)
    catalog = [_gpu("a", 1), _gpu("b", 4.25)]
    monkeypatch.setattr(batches, "_catalog", lambda *_: catalog)
    class Session:
        def __init__(self, *_args):
            pass
        def schema(self):
            return info
        def config(self):
            return global_session.GlobalConfig(GlobalGPUProfile(template_id="tmpl-test", volume_id="vol-test"), ["a", "b"], 3)
        def close(self):
            pass
    monkeypatch.setattr(global_session, "session_for", Session)
    requests = []
    from simplegen import workflows
    validated = []
    original_validate = workflows.validate_graph
    def inspect_graph(graph, schema, **kwargs):
        validated.append(graph)
        return original_validate(graph, schema, **kwargs)
    monkeypatch.setattr(workflows, "validate_graph", inspect_graph)
    def fake_assess(_settings, _folder, request, _cancel, _emit):
        requests.append(request)
        return {"options": [{"gpu_id": gpu["gpu_id"], "suitable": True,
                             "recommended": gpu["gpu_id"] == "a", "reason": "enough VRAM",
                             "runtime_seconds_low": 10, "runtime_seconds_high": 20}
                            for gpu in request["available"]]}
    monkeypatch.setattr(resources, "assess", fake_assess)
    ready = store.prepare(batch["id"], events.append, Event(), detach)
    assert ready["status"] == "prepared"
    assert len(agent_calls) == 1 and not jobs
    assert requests == []
    assert len(validated) >= 3
    assert ready["gpu_options"] == []
