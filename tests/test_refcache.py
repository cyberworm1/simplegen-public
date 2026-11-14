"""Phase 6: local reference cache (capture at acceptance, use downstream, pins and cleanup)."""

import copy
import hashlib
import json
import os
import shutil
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import Event

import pytest
from conftest import PASSED

from simplegen import agents, batches, editing, engine, refcache
from simplegen.preflight import run_preflight, run_saved_preflight
from simplegen.scheduling import Scheduler
from simplegen.settings import Settings


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def check(result, identity):
    return next(c for c in result["checks"] if c["id"] == identity)


@pytest.fixture
def world(tmp_path, monkeypatch):
    """A schema 4 run through the real engine and acceptance code, with fake agent and ComfyUI."""
    monkeypatch.setenv("SIMPLEGEN_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(engine, "STATE_VERSION", 4)
    assets, templates = tmp_path / "assets", tmp_path / "templates"
    assets.mkdir()
    templates.mkdir()
    (assets / "ref.png").write_bytes(b"first reference")
    (assets / "second.png").write_bytes(b"second reference")
    (templates / "test.json").write_text("{}")
    settings = Settings(project_folder=str(assets), template_folder=str(templates),
                        output_folder=str(tmp_path / "out"), template_file="test.json")
    prepared = {
        "graph": {"image": {"class_type": "LoadImage", "inputs": {"image": str(assets / "ref.png")}},
                  "final": {"class_type": "SaveVideo", "inputs": {}}},
        "prompt_slots": [],
        "reference_slots": [{"node_id": "image", "input": "image", "value": str(assets / "ref.png")}],
        "final_node": "final", "summary": "Synthetic workflow",
    }
    monkeypatch.setattr(editing, "edit_contract", lambda *a, **k: {"instruction": "edit"})
    monkeypatch.setattr(editing, "validate_edited_workflow", lambda *a, **k: copy.deepcopy(prepared))
    agent_calls = []

    def agent(config, workspace, brief, cancel, emit):
        agent_calls.append(brief)
        (workspace / "workflow.json").write_text('{"nodes": []}')
        plan = {"script": "A scene.", "used_references": [str(assets / "ref.png"), "second.png"]}
        (workspace / "plan.json").write_text(json.dumps(plan))
        return plan
    monkeypatch.setattr(agents, "generate_plan", agent)
    monkeypatch.setattr(agents, "executable", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(engine, "validate_graph", lambda *a: None)
    monkeypatch.setattr(engine, "probe_video", lambda p: {"size_bytes": p.stat().st_size, "width": 640,
                                                          "height": 360, "duration": 5.0})
    calls = {"uploads": [], "submitted": 0, "fail": None}

    class Comfy:
        def __init__(self, settings, client=None):
            pass

        def object_info(self):
            return {}

        def queue(self):
            return {"queue_running": [], "queue_pending": []}

        def upload(self, path, run_id, cancel):
            calls["uploads"].append((Path(path), Path(path).read_bytes()))
            return f"{run_id}/{Path(path).name}"

        def submit(self, graph, run_id, client_id):
            calls["submitted"] += 1
            if calls["fail"] == "submit":
                raise RuntimeError("ComfyUI rejected the prompt")
            return "job-1"

        def find_submission(self, run_id):
            return None

        def monitor(self, prompt_id, client_id, graph, cancel, emit):
            if calls["fail"] == "cancel":
                cancel.set()
                raise agents.Cancelled("Cancelled by the user.")
            return {"status": {"completed": True, "status_str": "success"},
                    "outputs": {"final": {"videos": [{"filename": "final.mp4", "type": "output", "subfolder": ""}]}}}

        def download(self, descriptor, target, cancel, callback):
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"video")
            return {"path": str(target), "size_bytes": 5, "sha256": sha(b"video")}

        def close(self):
            pass
    monkeypatch.setattr(engine, "Comfy", Comfy)

    def prepare():
        folder, state = engine.create_run(settings, "A concept")
        with engine.run_lock(folder):
            engine._run_locked(settings, "A concept", folder, state, lambda _: None, Event(), None,
                               prepare_only=True)
        return folder

    def unmount():
        shutil.move(str(assets), str(tmp_path / "unmounted"))

    return {"settings": settings, "assets": assets, "prepare": prepare, "calls": calls, "agent": agent_calls,
            "agent_function": agent,
            "unmount": unmount, "tmp": tmp_path}


def blobs() -> set[str]:
    folder = refcache.root() / "blobs"
    return {p.name for p in folder.iterdir()} if folder.is_dir() else set()


# 1. Capture at acceptance


def test_acceptance_captures_every_reference_and_slot(world):
    assets = world["assets"]
    before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in assets.iterdir()}
    folder = world["prepare"]()
    state = json.loads((folder / "state.json").read_text())
    manifest = json.loads((folder / "references.json").read_text())
    entries = {Path(e["original"]).name: e for e in manifest["references"]}
    assert set(entries) == {"ref.png", "second.png"}  # The LoadImage slot and every used reference.
    for name, entry in entries.items():
        data = (assets / name).read_bytes()
        assert entry["sha256"] == sha(data) and entry["size"] == len(data) and entry["captured_at"]
        assert entry["blob"] == sha(data) + ".png"
        assert (refcache.root() / "blobs" / entry["blob"]).read_bytes() == data
    assert "second.png" in entries["second.png"]["requested"]  # Relative agent paths map to the entry.
    assert {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in assets.iterdir()} == before
    pin = json.loads((refcache.root() / "pins" / f"{state['run_id']}.json").read_text())
    assert set(pin["blobs"]) == {e["blob"] for e in entries.values()}
    assert refcache.complete(folder)
    lines = (folder / "used_references.txt").read_text().splitlines()
    assert sorted(lines) == sorted(e["original"] for e in entries.values())


def test_unavailable_project_folder_fails_acceptance_retryably(world, monkeypatch):
    def agent_then_unmount(*args):
        plan = world["agent_function"](*args)
        shutil.move(str(world["assets"]), str(world["tmp"] / "away"))  # The mount drops as the agent finishes.
        return plan
    monkeypatch.setattr(agents, "generate_plan", agent_then_unmount)
    folder, state = engine.create_run(world["settings"], "A concept")
    with engine.run_lock(folder), pytest.raises(RuntimeError, match="references cannot be captured"):
        engine._run_locked(world["settings"], "A concept", folder, state, lambda _: None, Event(), None,
                           prepare_only=True)
    state = json.loads((folder / "state.json").read_text())
    assert not state.get("workflow_acceptance") and not (folder / "references.json").exists()
    assert [a["status"] for a in state["workflow_edit_attempts"]] == ["running"]  # Not spent as invalid.
    shutil.move(str(world["tmp"] / "away"), str(world["assets"]))
    with engine.run_lock(folder):
        engine._run_locked(world["settings"], "A concept", folder, state, lambda _: None, Event(), None,
                           prepare_only=True)
    assert len(world["agent"]) == 1 and refcache.complete(folder)  # Resumed without another agent pass.


# 2. Mount unavailable after acceptance


def test_scheduled_local_run_proceeds_from_cache_without_the_mount(world):
    folder = world["prepare"]()
    state = json.loads((folder / "state.json").read_text())
    now = datetime.now(UTC)
    clock = [(now - timedelta(hours=1)).timestamp()]
    scheduler = Scheduler(world["tmp"] / "schedules", clock=lambda: clock[0])
    scheduler.add_prepared(folder, now - timedelta(minutes=30), preflight=PASSED)
    world["unmount"]()
    # Dispatch preflight: the linked Project folder is not needed for accepted, cached work.
    result = run_saved_preflight(folder, object_info={}, resolve=lambda name: f"/usr/bin/{name}")
    linked = check(result, "linked_folders")
    assert linked["status"] == "pass" and "cached" in linked["message"]
    clock[0] = now.timestamp()
    assert scheduler.claim(state["run_id"], PASSED) == folder
    result = engine.run(Settings(), "", lambda _: None, Event(), resume_folder=folder)
    assert Path(result["final_path"]).is_file()
    (path, data), = world["calls"]["uploads"]
    assert path.parent == refcache.root() / "blobs" and data == b"first reference"
    assert json.loads((folder / "state.json").read_text())["status"] == "complete"
    assert blobs() == set()  # Success released the pin and collected the blobs.


def test_batch_fingerprint_and_preflight_use_the_cache(world):
    folder = world["prepare"]()
    fingerprint = batches._fingerprint(folder)
    world["unmount"]()
    assert batches._fingerprint(folder) == fingerprint  # Content hashes, not the unmounted mtime.
    state = json.loads((folder / "state.json").read_text())
    result = run_preflight(Settings(**state["settings"]), target="cloud", object_info={},
                           template_path=folder / "template.original.json", compiled=False,
                           cached_references=refcache.complete(folder, state),
                           resolve=lambda name: f"/usr/bin/{name}")
    assert check(result, "linked_folders")["status"] == "pass"
    # The batch upload itself is covered by test_batch_engine_integration's missing_reference mode.


# 3. Integrity


def test_corrupt_blob_falls_back_to_a_matching_original_else_fails(world):
    folder = world["prepare"]()
    manifest = json.loads((folder / "references.json").read_text())
    entry = next(e for e in manifest["references"] if e["original"].endswith("ref.png"))
    blob = refcache.root() / "blobs" / entry["blob"]
    blob.write_bytes(b"bit rot")
    result = engine.run(Settings(), "", lambda _: None, Event(), resume_folder=folder)
    assert result["final_path"]
    (path, data), = world["calls"]["uploads"]
    assert path == world["assets"] / "ref.png" and data == b"first reference"  # Verified original.
    state = json.loads((folder / "state.json").read_text())
    assert "corrupted" in state["reference_cache_problems"][0]["message"]

    second = world["prepare"]()
    manifest = json.loads((second / "references.json").read_text())
    entry = next(e for e in manifest["references"] if e["original"].endswith("ref.png"))
    (refcache.root() / "blobs" / entry["blob"]).write_bytes(b"bit rot")
    (world["assets"] / "ref.png").write_bytes(b"edited after acceptance")
    with pytest.raises(RuntimeError, match="corrupted and the original is unavailable or has changed"):
        engine.run(Settings(), "", lambda _: None, Event(), resume_folder=second)
    assert world["calls"]["submitted"] == 1  # Only the first run submitted.
    assert refcache.pinned(json.loads((second / "state.json").read_text())["run_id"])


def test_localize_never_returns_an_unverified_blob(world, tmp_path):
    original = tmp_path / "loose.png"
    original.write_bytes(b"loose")
    manifest = refcache.snapshot("r1", [str(original)], [tmp_path], lambda v: Path(v))
    entry = manifest["references"][0]
    assert refcache.localize(entry) == refcache.root() / "blobs" / entry["blob"]
    (refcache.root() / "blobs" / entry["blob"]).unlink()
    reports = []
    assert refcache.localize(entry, reports.append) == original and reports
    assert (refcache.root() / "blobs" / entry["blob"]).read_bytes() == b"loose"  # Restored from the original.


# 4. Pins and cleanup


@pytest.mark.parametrize("fault", ["submit", "cancel"])
def test_failure_and_cancel_keep_pins(world, fault):
    folder = world["prepare"]()
    run_id = json.loads((folder / "state.json").read_text())["run_id"]
    kept = blobs()
    world["calls"]["fail"] = fault
    with pytest.raises(RuntimeError):
        engine.run(Settings(), "", lambda _: None, Event(), resume_folder=folder)
    refcache.gc()
    assert refcache.pinned(run_id) and blobs() == kept
    status = json.loads((folder / "state.json").read_text())["status"]
    assert status == ("failed" if fault == "submit" else "cancelled")


def test_attention_schedule_and_removed_schedule_keep_pins(world):
    removed, failing = world["prepare"](), world["prepare"]()
    ids = [json.loads((f / "state.json").read_text())["run_id"] for f in (removed, failing)]
    now = datetime.now(UTC)
    clock = [(now - timedelta(hours=1)).timestamp()]
    scheduler = Scheduler(world["tmp"] / "schedules", clock=lambda: clock[0])
    scheduler.add_prepared(removed, now + timedelta(hours=1), preflight=PASSED)
    scheduler.change(ids[0], "remove")
    scheduler.add_prepared(failing, now - timedelta(minutes=30), preflight=PASSED)
    clock[0] = now.timestamp()
    assert scheduler.claim(ids[1], PASSED) == failing
    world["calls"]["fail"] = "submit"
    with pytest.raises(RuntimeError):
        engine.run(Settings(), "", lambda _: None, Event(), resume_folder=failing)
    scheduler.finish(ids[1], False)
    assert scheduler.get(ids[1])["status"] == "attention"
    refcache.gc()
    assert all(refcache.pinned(run_id) for run_id in ids) and len(blobs()) == 2  # The projects still exist.


def test_shared_blob_survives_until_both_runs_release(world):
    first, second = world["prepare"](), world["prepare"]()
    ids = [json.loads((f / "state.json").read_text())["run_id"] for f in (first, second)]
    assert len(blobs()) == 2  # Same content, one blob each.
    refcache.release([ids[0]])
    assert len(blobs()) == 2
    refcache.release([ids[1]])
    assert blobs() == set()


def test_release_quietly_never_fails_success(world, monkeypatch):
    def broken():
        raise OSError("disk gone")
    monkeypatch.setattr(refcache, "cache_lock", broken)
    refcache.root().mkdir(parents=True)
    refcache.release_quietly(["run"])  # Logged, not raised.


# 5. Crash during capture; idempotent retry


def test_crash_during_capture_leaves_nothing_in_use_and_retry_is_idempotent(world, monkeypatch):
    real = refcache._copy_to_temp

    def crash(source):
        if source.name == "second.png":
            partial = refcache.root() / "blobs" / f"{refcache.TEMP_PREFIX}crashed"
            partial.write_bytes(b"sec")
            os.utime(partial, (1, 1))  # A copy abandoned long ago by a killed process.
            raise OSError("power lost")
        return real(source)
    monkeypatch.setattr(refcache, "_copy_to_temp", crash)
    folder, state = engine.create_run(world["settings"], "A concept")
    with engine.run_lock(folder), pytest.raises(RuntimeError, match="cannot be read right now"):
        engine._run_locked(world["settings"], "A concept", folder, state, lambda _: None, Event(), None,
                           prepare_only=True)
    state = json.loads((folder / "state.json").read_text())
    assert not state.get("workflow_acceptance") and not (folder / "references.json").exists()
    assert not refcache.pinned(state["run_id"])
    assert all(name.startswith(refcache.TEMP_PREFIX) for name in blobs())  # No partial blob by hash name.
    monkeypatch.setattr(refcache, "_copy_to_temp", real)
    with engine.run_lock(folder):
        engine._run_locked(world["settings"], "A concept", folder, state, lambda _: None, Event(), None,
                           prepare_only=True)
    first = json.loads((folder / "references.json").read_text())
    assert len(world["agent"]) == 1 and refcache.complete(folder)
    again = refcache.capture_run(folder, state["run_id"], world["settings"],
                                 {"reference_slots": [{"value": str(world["assets"] / "ref.png")}]},
                                 json.loads((folder / "plan.json").read_text()))
    strip = [{k: v for k, v in e.items() if k != "captured_at"} for e in first["references"]]
    assert [{k: v for k, v in e.items() if k != "captured_at"} for e in again["references"]] == strip
    refcache.gc()
    assert blobs() == {e["blob"] for e in first["references"]}  # Stale temp collected; blobs kept (pinned).


# 6. Old runs keep their behaviour


def test_old_runs_without_manifest_read_originals(world):
    folder = world["prepare"]()
    (folder / "references.json").unlink()
    run_id = json.loads((folder / "state.json").read_text())["run_id"]
    refcache.release([run_id])
    state = json.loads((folder / "state.json").read_text())
    assert not refcache.complete(folder)
    result = run_saved_preflight(folder, object_info={}, resolve=lambda name: f"/usr/bin/{name}")
    assert "cached" not in check(result, "linked_folders")["message"]
    engine.run(Settings(), "", lambda _: None, Event(), resume_folder=folder)
    (path, _), = world["calls"]["uploads"]
    assert path == world["assets"] / "ref.png"  # The original, as before.

    old = world["prepare"]()
    (old / "references.json").unlink()
    world["unmount"]()
    with pytest.raises(batches.ReferenceUnavailable):
        batches._fingerprint(old)  # mtime fingerprint, unchanged
    result = run_saved_preflight(old, object_info={}, resolve=lambda name: f"/usr/bin/{name}")
    assert check(result, "linked_folders")["status"] == "unverified"
    assert state["status"] == "prepared"


def test_cache_references_now_for_an_older_run(world):
    folder = world["prepare"]()
    (folder / "references.json").unlink()
    manifest = refcache.cache_saved(folder)
    assert {Path(e["original"]).name for e in manifest["references"]} == {"ref.png", "second.png"}
    world["unmount"]()
    assert refcache.complete(folder)
    engine.run(Settings(), "", lambda _: None, Event(), resume_folder=folder)
    (path, _), = world["calls"]["uploads"]
    assert path.parent == refcache.root() / "blobs"


def test_settings_shows_cache_size_and_clear_unused(world):
    import time

    from PySide6.QtWidgets import QApplication

    from simplegen.ui import SavedRunDialog, SettingsDialog
    app = QApplication.instance() or QApplication([])
    folder = world["prepare"]()
    state = json.loads((folder / "state.json").read_text())
    dialog = SavedRunDialog(state, folder)
    assert dialog.cache_button.isHidden()  # Already cached at acceptance.
    dialog.deleteLater()
    (folder / "references.json").unlink()
    dialog = SavedRunDialog(state, folder)
    assert not dialog.cache_button.isHidden()  # Older run: offer "Cache references now".
    dialog.deleteLater()
    refcache.release([state["run_id"]], collect=False)
    settings = SettingsDialog(world["settings"])
    assert "cached" in settings.reference_cache_label.text() and "0 bytes unused" not in settings.reference_cache_label.text()
    settings.clear_reference_cache()
    deadline = time.time() + 5
    while settings.cache_worker is not None and time.time() < deadline:
        app.processEvents()
    assert blobs() == set() and settings.reference_cache_label.text() == "0 bytes cached · 0 bytes unused"
    settings.deleteLater()
