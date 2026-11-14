"""A retired, never-used GPU session must not soft-lock its batch; never contact a paid provider.

Reported 2026-10-01: a B300 pod never became ready, the user terminated it in
RunPod, and the batch then refused every action ("owns a GPU") while Resume
reconnected to the vanished pod.
"""
from __future__ import annotations

import csv
import json
from datetime import UTC, datetime
from pathlib import Path
from threading import Event

import pytest
from test_batches import _approved, _gpu, _member

from simplegen import batches, cost_safety, resources
from simplegen.settings import atomic_json


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv("SIMPLEGEN_HOME", str(tmp_path / "simplegen-home"))


def _retired_session(**changes):
    """The stuck batch's session record, as saved after the pod was terminated."""
    session = {"backend": "global_desktop", "sid": "sid", "name": "sg-dead-gpu", "gpu_id": "b300",
               "create_intent": True, "create_attempted": True, "pod_id": "c5aysd91b4rxky",
               "phase": "retired", "jobs": {}, "retired": True, "billing_confirmed_stopped": True,
               "actual_hourly": 7.89, "quoted_hourly": 7.89,
               "quoted_gpu_hourly_usd": 7.89, "quoted_cpu_hourly_usd": 0.0,
               "session_started_at": "2026-10-01T15:38:21+00:00",
               "session_completed_at": "2026-10-01T16:23:21+00:00",
               "owner_failure": {"at": "2026-10-01T16:23:22+00:00", "reason": "GPU not ready within 45 minutes",
                                 "kind": "startup"}}
    session.update(changes)
    return session


@pytest.fixture
def stuck(tmp_path):
    store = batches.BatchStore(tmp_path)
    state = store.create()
    target = _member(store, state, tmp_path, 1)
    state.update(status="failed", gpu_choices=_approved(), gpu_options=[_gpu("a", 1), _gpu("b", 1.5)],
                 managed_session=_retired_session())
    store._save(Path(state["folder"]), state)
    return store, state, target


def _saved(store, state):
    return json.loads((Path(state["folder"]) / "state.json").read_text())


def test_retired_session_no_longer_blocks_cancel(stuck):
    store, state, _ = stuck
    assert not batches.owns_gpu(state)  # The batch window enables Cancel.
    assert store.cancel(state["id"])["status"] == "cancelled"
    saved = _saved(store, state)
    assert saved["managed_session"] == {}
    assert saved["retired_sessions"][0]["pod_id"] == "c5aysd91b4rxky"
    assert saved["retired_sessions"][0]["released_at"]


def test_configure_can_change_gpus_after_release(stuck, tmp_path, monkeypatch):
    store, state, _ = stuck
    (tmp_path / "profile.json").write_text(json.dumps({"gpu_max_hourly": 10.0}))
    from simplegen import global_session
    monkeypatch.setattr(global_session, "load_global_profile", lambda path: type("P", (), {"volume_id": "vol"})())
    configured = store.configure(state["id"], gpu_choices=[{"gpu_id": "b", "max_hourly": 2.0},
                                                           {"gpu_id": "a", "max_hourly": 2.0}])
    assert [c["gpu_id"] for c in configured["gpu_choices"]] == ["b", "a"]


def test_resume_selects_a_fresh_gpu_instead_of_the_vanished_pod(stuck, monkeypatch):
    store, state, _ = stuck
    assert store.resume(state["id"])["status"] == "armed"
    monkeypatch.setattr(batches, "_catalog", lambda *_: [_gpu("a", 1), _gpu("b", 1.5)])
    connected = []

    class Session:
        def connect(self, choice):
            assert not _saved(store, state)["managed_session"].get("name")  # No reuse of the dead record.
            connected.append(choice["gpu_id"])

        def complete(self, receipts):
            pass

        def close(self):
            pass
    from simplegen import global_session
    monkeypatch.setattr(global_session, "session_for", lambda *args: Session())

    def run(settings, concept, target, child, callback, cancel, detach, *, sequence, batch_phase):
        if batch_phase == "run":
            child["result"] = {"final_path": "final.mp4"}
            atomic_json(target / "state.json", child)
        return {"final_path": str(target / "final.mp4")}
    monkeypatch.setattr(batches, "_run_locked", run)
    monkeypatch.setattr(batches, "_verified_receipts", lambda *_: [{"path": "ok"}])
    done = store.run_batch(state["id"], lambda _: None, Event())
    assert connected == ["a"] and done["status"] == "complete"
    assert len(done["retired_sessions"]) == 1


@pytest.mark.parametrize("change", [
    {"jobs": {"job-1": {"prompt_id": "p"}}},
    {"unresolved_jobs": ["job-1"]},
    {"retired": False, "phase": "stopped"},          # The pod still exists (stopped).
    {"billing_confirmed_stopped": False},
    {"handed_off": True},
    {"handoff_intent": {"target": "other"}},
])
def test_sessions_that_may_hold_work_or_compute_stay_locked(stuck, change):
    store, state, _ = stuck
    state["managed_session"] = _retired_session(**change)
    store._save(Path(state["folder"]), state)
    assert batches.owns_gpu(state)
    with pytest.raises(ValueError, match="owns a GPU"):
        store.cancel(state["id"])
    assert "retired_sessions" not in _saved(store, state)


@pytest.mark.parametrize("child_change", ["submitted", "prompt", "variation", "unreadable"])
def test_a_submitted_or_unknown_member_keeps_the_session(stuck, child_change):
    store, state, target = stuck
    child = json.loads((target / "state.json").read_text())
    if child_change == "submitted":
        child["submission_started"] = True
    elif child_change == "prompt":
        child["prompt_id"] = "p-1"
    elif child_change == "variation":
        child["sequence"] = {"children": [{"folder": "variation-001"}]}
        atomic_json(target / "variation-001" / "state.json", {"prompt_id": "p-2"})
    if child_change == "unreadable":
        (target / "state.json").write_text("{not json")
    else:
        atomic_json(target / "state.json", child)
    with pytest.raises(ValueError, match="owns a GPU"):
        store.cancel(state["id"])


@pytest.mark.parametrize("status", ["running", "complete", "cancelled"])
def test_only_idle_batches_release(stuck, status):
    _, state, _ = stuck
    state["status"] = status
    assert batches.release_retired_session(Path(state["folder"]), state) is False


def test_released_session_keeps_its_cost_row_and_spend(stuck):
    store, state, _ = stuck
    store.cancel(state["id"])
    saved = _saved(store, state)
    # A later session on the same batch gets its own row instead of overwriting the first.
    saved["managed_session"] = {"backend": "global_desktop", "name": "sg-next-gpu", "actual_hourly": 6.79,
                                "quoted_gpu_hourly_usd": 6.79, "quoted_cpu_hourly_usd": 0.0,
                                "session_started_at": "2026-10-01T17:00:00+00:00",
                                "session_completed_at": "2026-10-01T18:00:00+00:00"}
    resources.record_run(Path(state["folder"]), saved)
    with resources._history_path().open(newline="", encoding="utf-8") as stream:
        rows = {row["run_id"]: row for row in csv.DictReader(stream)}
    assert set(rows) == {state["run_id"], f"{state['run_id']}-session2"}
    assert float(rows[state["run_id"]]["session_seconds"]) == pytest.approx(2700)
    assert float(rows[state["run_id"]]["observed_compute_cost_estimate_usd"]) == pytest.approx(5.9175)
    # The spend cap counts the released 45 minutes at $7.89/h plus the current session.
    released = 0.75 * 7.89
    assert cost_safety.released_usd(saved) == pytest.approx(released)
    now = datetime(2026, 10, 1, 17, 30, tzinfo=UTC)
    assert cost_safety.spent_usd(saved, now) == pytest.approx(released + 0.5 * 6.79)
    assert cost_safety.spent_usd({"retired_sessions": saved["retired_sessions"]}) == pytest.approx(released)
