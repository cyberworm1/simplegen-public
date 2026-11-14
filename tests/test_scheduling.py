"""Clock, persistence and dispatch boundaries without agents or real compute."""

import hashlib
import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from threading import Event
from zoneinfo import ZoneInfo

import pytest
from conftest import PASSED

from simplegen import engine, scheduling
from simplegen.scheduling import Scheduler, fingerprint, resolve_local
from simplegen.settings import Settings, atomic_json


@pytest.fixture
def scheduled(tmp_path, monkeypatch):
    monkeypatch.setenv("TZ", "America/New_York")
    source = tmp_path / "workflow.json"
    atomic_json(source, {"nodes": [], "revision": "original"})
    settings = Settings(project_folder=str(tmp_path), template_folder=str(tmp_path),
                        template_file=source.name, output_folder=str(tmp_path / "projects"), quantity=3)
    clock = [datetime(2030, 1, 1, 12, tzinfo=UTC).timestamp()]
    scheduler = Scheduler(tmp_path / "schedules", clock=lambda: clock[0])
    yield scheduler, settings, clock, source
    scheduler.close()


def future(clock, seconds=30):
    return datetime.fromtimestamp(clock[0] + seconds, UTC)


def test_schedule_freezes_request_without_agent_or_provider_work(scheduled, monkeypatch):
    scheduler, settings, clock, source = scheduled
    monkeypatch.setattr(engine.agents, "generate_plan", lambda *a: pytest.fail("agent started early"))
    monkeypatch.setattr(engine, "Comfy", lambda *a: pytest.fail("Comfy contacted early"))
    record = scheduler.add(settings, "A saved concept", future(clock), preflight=PASSED)
    folder = Path(record["folder"])
    state = json.loads((folder / "state.json").read_text())
    assert state["settings"]["quantity"] == 3
    assert state["concept"] == "A saved concept" and not state["schedule"]["released"]
    source.write_text('{"revision":"changed later"}')
    assert json.loads((folder / "template.original.json").read_text())["revision"] == "original"
    assert scheduler.tick() == [] and scheduler.claim(record["id"], PASSED) is None
    with pytest.raises(ValueError, match="scheduled"):
        engine.run(settings, "ignored", lambda _: None, Event(), folder)


def test_clock_dispatch_reservation_and_restart_need_explicit_resume(scheduled):
    scheduler, settings, clock, _source = scheduled
    record = scheduler.add(settings, "First", future(clock), preflight=PASSED)
    clock[0] += 30
    assert scheduler.tick()[0]["id"] == record["id"]
    folder = scheduler.claim(record["id"], PASSED)
    assert folder is not None and scheduler.claim(record["id"], PASSED) is None
    assert json.loads((folder / "state.json").read_text())["schedule"]["released"]
    scheduler.close()
    reopened = Scheduler(scheduler.root, clock=lambda: clock[0])
    try:
        assert reopened.get(record["id"])["status"] == "attention"
        assert reopened.tick() == []
        reopened.change(record["id"], "now")
        assert reopened.claim(record["id"], PASSED) == folder
    finally:
        reopened.close()


@pytest.mark.parametrize("restart", [False, True])
def test_missed_deadline_is_eligible_once_after_restart(scheduled, restart):
    scheduler, settings, clock, _source = scheduled
    record = scheduler.add(settings, "Missed", future(clock), preflight=PASSED)
    clock[0] += 120
    if restart:
        scheduler.close()
        scheduler = Scheduler(scheduler.root, clock=lambda: clock[0])
    try:
        assert [r["id"] for r in scheduler.tick()] == [record["id"]]
        assert scheduler.get(record["id"])["status"] in {"overdue", "waiting"}
        scheduler.defer(record["id"], "ComfyUI busy")
        assert scheduler.tick() == []
        clock[0] += 1800
        assert [r["id"] for r in scheduler.tick()] == [record["id"]]
        scheduler.change(record["id"], "reschedule", future(clock))
        assert scheduler.get(record["id"])["status"] == "scheduled"
        scheduler.change(record["id"], "remove")
        assert scheduler.get(record["id"])["status"] == "removed"
        assert Path(record["folder"]).is_dir()
        assert scheduler.tick() == []
    finally:
        if restart:
            scheduler.close()


def test_busy_waiting_is_not_a_missed_deadline(scheduled):
    scheduler, settings, clock, _source = scheduled
    record = scheduler.add(settings, "Waiting", future(clock), preflight=PASSED)
    for _ in range(60):
        clock[0] += 5
        scheduler.tick()
    assert scheduler.get(record["id"])["status"] == "waiting"
    assert scheduler.claim(record["id"], PASSED) is not None


def test_only_one_window_can_dispatch(scheduled):
    scheduler, settings, clock, _source = scheduled
    second = Scheduler(scheduler.root, clock=lambda: clock[0])
    try:
        assert not second.owner
        with pytest.raises(ValueError, match="another open"):
            second.add(settings, "Duplicate", future(clock), preflight=PASSED)
        assert second.tick() == []
    finally:
        second.close()


def test_cloud_and_past_schedules_rejected(scheduled):
    scheduler, settings, clock, _source = scheduled
    with pytest.raises(ValueError, match="Local only"):
        scheduler.add(replace(settings, cloud=True), "No rental", future(clock), preflight=PASSED)
    with pytest.raises(ValueError, match="future"):
        scheduler.add(settings, "Past", future(clock, -1), preflight=PASSED)
    assert scheduler.records() == []


@pytest.mark.parametrize("wall", ["2026-03-08T02:30", "2026-11-01T01:30"])
def test_dst_gaps_and_folds_are_explicitly_rejected(wall):
    with pytest.raises(ValueError, match="daylight saving"):
        resolve_local(datetime.fromisoformat(wall), zone=ZoneInfo("America/New_York"))


def test_local_24h_conversion_and_instant_survives_timezone_change(scheduled, monkeypatch):
    scheduler, settings, _clock, _source = scheduled
    instant = resolve_local(datetime.fromisoformat("2030-01-01T22:45"), zone=ZoneInfo("America/New_York"))
    assert instant.hour == 3 and instant.day == 2
    record = scheduler.add(settings, "Evening", instant, preflight=PASSED)
    assert scheduling.display_time(record["due_at"]) == "01/01/30 22:45"
    monkeypatch.setenv("TZ", "America/Los_Angeles")
    assert scheduling.display_time(record["due_at"]) == "01/01/30 19:45"
    assert datetime.fromisoformat(scheduler.get(record["id"])["due_at"]) == instant


def test_failed_schedule_is_not_automatically_retried(scheduled):
    scheduler, settings, clock, _source = scheduled
    record = scheduler.add(settings, "Failure", future(clock), preflight=PASSED)
    scheduler.change(record["id"], "now")
    scheduler.claim(record["id"], PASSED)
    scheduler.finish(record["id"], False)
    assert scheduler.tick() == []
    assert scheduler.get(record["id"])["status"] == "attention"


def test_completed_work_is_recovered_without_dispatch(scheduled):
    scheduler, settings, clock, _source = scheduled
    record = scheduler.add(settings, "Finished before crash", future(clock), preflight=PASSED)
    scheduler.change(record["id"], "now")
    folder = scheduler.claim(record["id"], PASSED)
    state = json.loads((folder / "state.json").read_text())
    state["status"] = "complete"
    atomic_json(folder / "state.json", state)
    scheduler.recover()
    assert scheduler.get(record["id"])["status"] == "complete"
    assert scheduler.tick() == []


def test_explicit_resume_completion_clears_only_attention_ledger(scheduled):
    scheduler, settings, clock, _ = scheduled
    record = scheduler.add(settings, "Resume", future(clock), preflight=PASSED)
    scheduler.change(record["id"], "now")
    folder = scheduler.claim(record["id"], PASSED)
    scheduler.finish(record["id"], False)
    state = json.loads((folder / "state.json").read_text())
    state["status"] = "complete"
    atomic_json(folder / "state.json", state)
    assert scheduler.reconcile_completed(folder)
    assert scheduler.get(record["id"])["status"] == "complete"
    assert not scheduler.reconcile_completed(folder)


def test_pending_edit_preserves_prior_revision_and_saved_workflow(scheduled, tmp_path):
    scheduler, settings, clock, _source = scheduled
    record = scheduler.add(settings, "First", future(clock), preflight=PASSED)
    folder = Path(record["folder"])
    old_state = (folder / "state.json").read_bytes()
    old_template = (folder / "template.original.json").read_bytes()
    alternate = tmp_path / "alternate.json"
    atomic_json(alternate, {"nodes": [], "revision": "replacement"})
    updated = scheduler.edit(record["id"], fingerprint(record),
                             replace(settings, template_file=alternate.name, steps=33, quantity=2),
                             "Revised concept", future(clock, 120))
    state = json.loads((folder / "state.json").read_text())
    assert state["concept"] == updated["concept"] == "Revised concept"
    assert state["settings"]["steps"] == 33 and state["settings"]["quantity"] == 2
    assert json.loads((folder / "template.original.json").read_text())["revision"] == "replacement"
    revision = folder / "schedule-revisions" / updated["revision"]
    assert (revision / "prior-state.json").read_bytes() == old_state
    assert (revision / "prior-template.json").read_bytes() == old_template
    assert "edit_pending" not in updated
    with pytest.raises(ValueError, match="changed or started"):
        scheduler.edit(record["id"], fingerprint(record), settings, "Stale", future(clock, 150))


def test_interrupted_pending_edit_replays_before_dispatch(scheduled, monkeypatch):
    scheduler, settings, clock, _ = scheduled
    record = scheduler.add(settings, "Original", future(clock), preflight=PASSED)
    folder = Path(record["folder"])
    original = scheduler._finish_edit

    def crash(*args, **kwargs):
        raise OSError("simulated crash after edit marker")

    monkeypatch.setattr(scheduler, "_finish_edit", crash)
    with pytest.raises(OSError, match="simulated crash"):
        scheduler.edit(record["id"], fingerprint(record), settings, "Updated", future(clock, 90))
    assert scheduler.get(record["id"])["edit_pending"]
    assert scheduler.claim(record["id"], PASSED) is None
    monkeypatch.setattr(scheduler, "_finish_edit", original)
    scheduler.close()
    reopened = Scheduler(scheduler.root, clock=lambda: clock[0])
    try:
        assert reopened.get(record["id"])["concept"] == "Updated"
        assert "edit_pending" not in reopened.get(record["id"])
        assert json.loads((folder / "state.json").read_text())["concept"] == "Updated"
    finally:
        reopened.close()


def test_edit_keeps_existing_snapshot_bytes_when_linked_file_changes(scheduled):
    scheduler, settings, clock, source = scheduled
    record = scheduler.add(settings, "Original", future(clock), preflight=PASSED)
    folder = Path(record["folder"])
    noncanonical = b'{"nodes":[],"revision":"original"}\n'
    (folder / "template.original.json").write_bytes(noncanonical)
    state = json.loads((folder / "state.json").read_text())
    state["template_sha256"] = hashlib.sha256(noncanonical).hexdigest()
    atomic_json(folder / "state.json", state)
    source.write_text('{"nodes":[],"revision":"external change"}')
    scheduler.edit(record["id"], fingerprint(record), settings, "Changed concept", future(clock, 90))
    assert (folder / "template.original.json").read_bytes() == noncanonical
    state = json.loads((folder / "state.json").read_text())
    assert state["template_sha256"] == hashlib.sha256(noncanonical).hexdigest()


@pytest.mark.parametrize("boundary", ["template", "state", "record"])
def test_interrupted_edit_recovers_at_each_materialization_boundary(scheduled, monkeypatch, boundary):
    scheduler, settings, clock, _ = scheduled
    record = scheduler.add(settings, "Before", future(clock), preflight=PASSED)
    folder = Path(record["folder"])
    if boundary in {"template", "state"}:
        real_write = scheduling.atomic_bytes

        def fail_after_write(path, data):
            real_write(path, data)
            if path.name == ("template.original.json" if boundary == "template" else "state.json"):
                raise OSError("simulated interrupted materialization")

        monkeypatch.setattr(scheduling, "atomic_bytes", fail_after_write)
    else:
        real_save = scheduler._save

        def fail_final(record_to_save):
            if record_to_save.get("revision") and not record_to_save.get("edit_pending"):
                raise OSError("simulated interrupted record commit")
            real_save(record_to_save)

        monkeypatch.setattr(scheduler, "_save", fail_final)
    with pytest.raises(OSError, match="simulated interrupted"):
        scheduler.edit(record["id"], fingerprint(record), settings, "After", future(clock, 90))
    assert scheduler.get(record["id"])["edit_pending"]
    assert scheduler.claim(record["id"], PASSED) is None
    scheduler.close()
    monkeypatch.undo()
    reopened = Scheduler(scheduler.root, clock=lambda: clock[0])
    try:
        assert reopened.get(record["id"])["concept"] == "After"
        assert json.loads((folder / "state.json").read_text())["concept"] == "After"
        assert "edit_pending" not in reopened.get(record["id"])
    finally:
        reopened.close()


def test_edit_and_dispatch_reject_external_project_lock(scheduled):
    scheduler, settings, clock, _ = scheduled
    record = scheduler.add(settings, "Locked", future(clock), preflight=PASSED)
    with engine.run_lock(Path(record["folder"])), pytest.raises(RuntimeError, match="already running"):
        scheduler.edit(record["id"], fingerprint(record), settings, "After", future(clock, 90))
    assert scheduler.get(record["id"])["concept"] == "Locked"


def test_open_edit_suspends_stale_claim(scheduled):
    scheduler, settings, clock, _ = scheduled
    record = scheduler.add(settings, "Pending", future(clock), preflight=PASSED)
    clock[0] += 30
    with scheduler.suspend(record["id"]):
        assert scheduler.tick() == []
        assert scheduler.claim(record["id"], PASSED) is None
    assert scheduler.claim(record["id"], PASSED) == Path(record["folder"])


def test_scheduled_fork_keeps_saved_workflow_snapshot(scheduled):
    scheduler, settings, clock, source = scheduled
    first = scheduler.add(settings, "First", future(clock), preflight=PASSED)
    first_folder = Path(first["folder"])
    first_state = json.loads((first_folder / "state.json").read_text())
    original_snapshot = (first_folder / "template.original.json").read_bytes()
    seed = {"folder": str(first_folder), "run_id": first["id"],
            "template_sha256": first_state["template_sha256"],
            "template_file": settings.template_file, "template_folder": settings.template_folder}
    source.write_text('{"revision":"changed on disk"}')
    second = scheduler.add(settings, "Second", future(clock, 120), fork_seed=seed, preflight=PASSED)
    second_folder = Path(second["folder"])
    assert (second_folder / "template.original.json").read_bytes() == original_snapshot
    state = json.loads((second_folder / "state.json").read_text())
    assert state["forked_from"]["run_id"] == first["id"]
    assert state["concept"] == "Second"


def test_invalid_workflow_does_not_leave_unreleased_schedule_project(scheduled):
    scheduler, settings, clock, source = scheduled
    source.write_text("not-json")
    with pytest.raises(ValueError):
        scheduler.add(settings, "Invalid", future(clock), preflight=PASSED)
    assert scheduler.records() == []
    assert not list(Path(settings.output_folder).glob("*/state.json"))


def test_corrupt_pending_edit_is_attention_and_not_dispatched(scheduled, monkeypatch):
    scheduler, settings, clock, _ = scheduled
    record = scheduler.add(settings, "Before", future(clock), preflight=PASSED)
    monkeypatch.setattr(scheduler, "_finish_edit", lambda *a, **kw: (_ for _ in ()).throw(OSError("crash")))
    with pytest.raises(OSError):
        scheduler.edit(record["id"], fingerprint(record), settings, "After", future(clock, 90))
    revision = scheduler.get(record["id"])["edit_pending"]
    manifest = Path(record["folder"]) / "schedule-revisions" / revision / "manifest.json"
    manifest.write_text("{}")
    scheduler.close()
    monkeypatch.undo()
    reopened = Scheduler(scheduler.root, clock=lambda: clock[0])
    try:
        assert reopened.get(record["id"])["status"] == "attention"
        clock[0] += 3600
        assert reopened.tick() == []
    finally:
        reopened.close()


@pytest.mark.parametrize("unsafe_status", ["failed", "cancelled", "paused"])
def test_never_started_record_with_nonpending_project_is_not_dispatched(scheduled, unsafe_status):
    scheduler, settings, clock, _ = scheduled
    record = scheduler.add(settings, "Unsafe", future(clock), preflight=PASSED)
    folder = Path(record["folder"])
    state = json.loads((folder / "state.json").read_text())
    state["status"] = unsafe_status
    atomic_json(folder / "state.json", state)
    clock[0] += 30
    assert scheduler.claim(record["id"], PASSED) is None
    assert scheduler.get(record["id"])["status"] == "attention"
    assert scheduler.tick() == []
