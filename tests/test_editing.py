"""Durability and correction boundaries independent of graph policy tests."""

import copy
import json
from threading import Event

import pytest

from simplegen import agents, editing
from simplegen.settings import Settings, atomic_json


@pytest.fixture
def edit_run(tmp_path, monkeypatch):
    source = tmp_path / "source.json"
    atomic_json(source, {"nodes": [], "valid": False})
    folder = tmp_path / "run"
    folder.mkdir()
    state = {"run_id": "test"}
    settings = Settings(project_folder=str(tmp_path), template_folder=str(tmp_path))
    state["settings"] = settings.public_dict()
    prepared = {"graph": {"1": {}}, "prompt_slots": [], "reference_slots": [], "summary": "valid"}
    monkeypatch.setattr(editing, "edit_contract", lambda *a, **k: {"instruction": "edit directly"})

    def validate(original, edited, info, **kwargs):
        assert original == {"nodes": [], "valid": False}
        if not edited.get("valid"):
            raise ValueError("Segment 2 previous link is missing")
        return copy.deepcopy(prepared)

    monkeypatch.setattr(editing, "validate_edited_workflow", validate)
    calls = []

    def generate(config, workspace, brief, cancel, emit):
        calls.append(brief)
        atomic_json(workspace / "workflow.json", {"nodes": [], "valid": True})
        response = {"script": "A scene", "used_references": []}
        atomic_json(workspace / "plan.json", response)
        return response

    monkeypatch.setattr(agents, "generate_plan", generate)

    def persist(status):
        state["status"] = status
        atomic_json(folder / "state.json", state)

    def run():
        return editing.prepare_edit(
            settings,
            folder,
            state,
            {},
            Event(),
            lambda _: None,
            persist,
            source,
            {},
            lambda *a: ([], []),
        )

    return run, folder, source, state, calls, generate


def test_direct_file_is_accepted_without_applying_prompt_replacements(edit_run):
    run, folder, source, state, calls, _ = edit_run
    result = run()
    assert result[1]["script"] == "A scene"
    assert json.loads((folder / "workflow.json").read_text())["valid"]
    assert json.loads(source.read_text())["valid"] is False
    assert state["workflow_edit_attempts"] == [{"number": 1, "status": "valid"}]
    run()
    assert len(calls) == 1  # Restart/resume revalidates without more AI usage.


def test_single_correction_receives_exact_error_and_preserves_attempt(edit_run, monkeypatch):
    run, folder, _, state, calls, original = edit_run

    def generate(*args):
        response = original(*args)
        if len(calls) == 1:
            atomic_json(args[1] / "workflow.json", {"valid": False})
        return response

    monkeypatch.setattr(agents, "generate_plan", generate)
    run()
    assert len(calls) == 2
    assert calls[1]["validation_error"] == "Segment 2 previous link is missing"
    assert calls[1]["previous_script"] == "A scene"
    assert state["workflow_acceptance"]["attempt"] == 2
    assert (folder / "agent-edits/attempt-1/validation.json").is_file()
    assert not json.loads((folder / "agent-edits/attempt-1/workflow.json").read_text())["valid"]


def test_second_invalid_edit_fails_and_resume_does_not_reset_budget(edit_run, monkeypatch):
    run, _, _, state, calls, original = edit_run

    def invalid(*args):
        original(*args)
        atomic_json(args[1] / "workflow.json", {"valid": False})

    monkeypatch.setattr(agents, "generate_plan", invalid)
    for _ in range(2):
        with pytest.raises(ValueError, match="after one correction"):
            run()
    assert len(calls) == len(state["workflow_edit_attempts"]) == 2


def test_provider_failure_has_no_automatic_retry(edit_run, monkeypatch):
    run, _, _, state, calls, _ = edit_run

    def fail(*args):
        calls.append(args[2])
        raise RuntimeError("provider unavailable")

    monkeypatch.setattr(agents, "generate_plan", fail)
    with pytest.raises(RuntimeError, match="provider unavailable"):
        run()
    assert len(calls) == len(state["workflow_edit_attempts"]) == 1


def test_crash_after_cli_outputs_recovers_without_second_agent_call(edit_run, monkeypatch):
    run, _, _, state, calls, original = edit_run

    def crash(*args):
        original(*args)
        raise RuntimeError("simulated desktop exit")

    monkeypatch.setattr(agents, "generate_plan", crash)
    with pytest.raises(RuntimeError):
        run()
    assert state["workflow_edit_attempts"][0]["status"] == "running"
    run()
    assert len(calls) == 1
    assert state["workflow_acceptance"]["attempt"] == 1


def test_modified_accepted_workflow_is_not_silently_repaired(edit_run):
    run, folder, _, _, calls, _ = edit_run
    run()
    atomic_json(folder / "workflow.json", {"valid": True, "changed": True})
    with pytest.raises(ValueError, match="changed after validation"):
        run()
    assert len(calls) == 1


def test_unreceipted_snapshot_before_first_agent_is_recovered(edit_run):
    run, folder, source, state, calls, _ = edit_run
    (folder / "template.original.json").write_bytes(source.read_bytes())
    run()
    assert state["workflow_acceptance"]
    assert len(calls) == 1


def test_orphan_workspace_is_preserved_without_consuming_agent_usage(edit_run):
    run, folder, _, state, calls, _ = edit_run
    orphan = folder / "agent-edits/attempt-1"
    orphan.mkdir(parents=True)
    (orphan / "workflow.json").write_text("interrupted copy")
    run()
    archived = next((folder / "agent-edits").glob("unreserved-*"))
    assert (archived / "workflow.json").read_text() == "interrupted copy"
    assert len(calls) == len(state["workflow_edit_attempts"]) == 1


def test_malformed_response_gets_only_one_correction(edit_run, monkeypatch):
    run, _, _, _, calls, _ = edit_run

    def malformed(*args):
        calls.append(args[2])
        raise agents.InvalidAgentResponse("missing structured result")

    monkeypatch.setattr(agents, "generate_plan", malformed)
    with pytest.raises(ValueError, match="missing structured result"):
        run()
    assert len(calls) == 2


def test_extra_response_field_is_harmless_and_override_reuses_saved_attempt(edit_run):
    run, folder, _, state, calls, _ = edit_run
    run()
    assert len(calls) == 1
    state.pop("workflow_acceptance")
    state["workflow_edit_attempts"] = [{"number": 1, "status": "invalid",
        "error": "Return a nonempty script and used_references list; edit the workflow file directly."}]
    atomic_json(folder / "state.json", state)
    workspace = folder / "agent-edits/attempt-1"
    atomic_json(workspace / "plan.json", {"script": "A scene", "used_references": [], "note": "harmless"})
    editing.request_saved_response_override(folder)
    state.update(json.loads((folder / "state.json").read_text()))
    result = run()
    assert result[1]["note"] == "harmless"
    assert len(calls) == 1
    receipt = state["workflow_acceptance"]
    assert receipt["response_override"]["reason"].startswith("Return a nonempty")
    assert receipt["workflow_sha256"] and receipt["plan_sha256"]


def test_override_refuses_changed_saved_artifact_without_agent_call(edit_run):
    run, folder, _, state, calls, _ = edit_run
    run()
    state.pop("workflow_acceptance")
    state["workflow_edit_attempts"] = [{"number": 1, "status": "invalid",
        "error": "Return a nonempty script and used_references list; edit the workflow file directly."}]
    atomic_json(folder / "state.json", state)
    atomic_json(folder / "agent-edits/attempt-1/plan.json",
                {"script": "A scene", "used_references": [], "note": "extra"})
    editing.request_saved_response_override(folder)
    state.update(json.loads((folder / "state.json").read_text()))
    atomic_json(folder / "agent-edits/attempt-1/workflow.json", {"nodes": [], "valid": False})
    with pytest.raises(ValueError, match="changed after approval"):
        run()
    assert len(calls) == 1


def test_override_cannot_accept_invalid_saved_workflow(edit_run):
    run, folder, _, state, calls, _ = edit_run
    run()
    state.pop("workflow_acceptance")
    state["workflow_edit_attempts"] = [{"number": 1, "status": "invalid",
        "error": "Return a nonempty script and used_references list; edit the workflow file directly."}]
    atomic_json(folder / "state.json", state)
    workspace = folder / "agent-edits/attempt-1"
    atomic_json(workspace / "plan.json", {"script": "A scene", "used_references": [], "note": "extra"})
    atomic_json(workspace / "workflow.json", {"nodes": [], "valid": False})
    editing.request_saved_response_override(folder)
    state.update(json.loads((folder / "state.json").read_text()))
    with pytest.raises(ValueError, match="previous link is missing"):
        run()
    assert len(calls) == 1
    assert not state.get("workflow_acceptance")
    assert not state.get("submission_started")


def test_older_format_rejection_does_not_override_latest_failure(edit_run):
    run, folder, _, state, _, _ = edit_run
    run()
    state.pop("workflow_acceptance")
    state["workflow_edit_attempts"] = [
        {"number": 1, "status": "invalid",
         "error": "Return a nonempty script and used_references list; edit the workflow file directly."},
        {"number": 2, "status": "running", "error": "Provider unavailable"},
    ]
    atomic_json(folder / "state.json", state)
    with pytest.raises(ValueError, match="No saved response-format rejection"):
        editing.inspect_saved_response(folder)


@pytest.mark.parametrize("content", ['{"nodes":[], "nodes":[]}', '{"value":NaN}', "[]"])
def test_strict_json_rejects_ambiguous_or_nonstandard_input(tmp_path, content):
    path = tmp_path / "workflow.json"
    path.write_text(content)
    with pytest.raises(ValueError):
        editing.read_json(path)


def test_symlink_workflow_is_not_accepted(tmp_path):
    source = tmp_path / "source.json"
    source.write_text("{}")
    link = tmp_path / "workflow.json"
    link.symlink_to(source)
    with pytest.raises(ValueError, match="regular JSON"):
        editing.read_json(link)


def test_preflight_failure_preserves_budget_and_retries_after_restart(edit_run, monkeypatch):
    run, folder, _, state, calls, original = edit_run
    def fail(*args):
        raise agents.AgentPreflightError('CLI authentication check failed')
    monkeypatch.setattr(agents, 'generate_plan', fail)
    for _ in range(3):
        with pytest.raises(agents.AgentPreflightError):
            run()
        assert state['workflow_edit_attempts'] == []
        saved = json.loads((folder / 'state.json').read_text())
        state.clear()
        state.update(saved)
    assert len(state['agent_preflight_failures']) == 3
    assert len(list((folder / 'agent-edits').glob('preflight-*'))) == 3
    monkeypatch.setattr(agents, 'generate_plan', original)
    run()
    assert len(calls) == 1
    assert state['workflow_acceptance']['attempt'] == 1


def test_edit_mode_drops_workspace_files_from_used_references(edit_run, monkeypatch):
    run, folder, _, _, _, _ = edit_run

    def generate(config, workspace, brief, cancel, emit):
        atomic_json(workspace / "workflow.json", {"nodes": [], "valid": True})
        response = {"script": "A scene", "used_references": [str(workspace / "workflow.json")]}
        atomic_json(workspace / "plan.json", response)
        return response

    monkeypatch.setattr(agents, "generate_plan", generate)
    assert run()[1]["used_references"] == []
    assert json.loads((folder / "plan.json").read_text())["used_references"] == []


@pytest.fixture
def snapshot_run(tmp_path):
    """A run folder whose receipted snapshot came from a linked template (as in _snapshot)."""
    linked = tmp_path / "linked"
    (linked / "workflows").mkdir(parents=True)
    source = linked / "workflows" / "chain.json"
    atomic_json(source, {"nodes": [1, 2]})
    folder = tmp_path / "run"
    atomic_json(folder / "template.original.json", {"nodes": [1, 2]})
    read_only = folder / "agent-edits" / "attempt-2" / editing.READ_ONLY_COPY
    read_only.parent.mkdir(parents=True)
    read_only.write_bytes((folder / "template.original.json").read_bytes())
    state = {"template_path": str(source),
             "template_sha256": editing.digest(folder / "template.original.json")}
    return folder, state, [linked.resolve()], source, read_only


def test_read_only_copy_is_replaced_by_the_linked_template(snapshot_run):
    folder, state, roots, source, read_only = snapshot_run
    image = str(roots[0] / "ref.png")
    plan = {"script": "s", "used_references": [image, str(read_only), str(folder / "plan.json")]}
    result, changes = editing.normalize_used_references(plan, folder, state, roots)
    assert result["used_references"] == [image, str(source.resolve())]
    assert [c["action"] for c in changes] == ["replaced", "dropped"]
    assert plan["used_references"][1] == str(read_only)  # The input plan is not mutated.


def test_read_only_copy_is_dropped_when_the_linked_template_changed(snapshot_run):
    folder, state, roots, source, read_only = snapshot_run
    atomic_json(source, {"nodes": [1, 2, 3]})
    result, changes = editing.normalize_used_references(
        {"used_references": [str(read_only)]}, folder, state, roots)
    assert result["used_references"] == [] and changes[0]["action"] == "dropped"


def test_altered_read_only_copy_is_dropped(snapshot_run):
    folder, state, roots, _, read_only = snapshot_run
    read_only.write_text('{"nodes": []}')
    result, _ = editing.normalize_used_references({"used_references": [str(read_only)]}, folder, state, roots)
    assert result["used_references"] == []


def test_paths_outside_the_run_are_left_for_validation(snapshot_run):
    folder, state, roots, _, _ = snapshot_run
    plan = {"used_references": ["/etc/hostname", "relative/notes.md", 7, str(roots[0] / "bible.md")]}
    result, changes = editing.normalize_used_references(plan, folder, state, roots)
    assert result is plan and changes == []


def _placeholder_plan(quantity=1):
    """The rejected first attempt of the 2026-10-01 cloud batch, with paths removed."""
    pending = [{"sections": {name: "pending" for name in ("summary", "detailed_description")}}]
    plan = {"script": "I'll start from the project index, read the bible and character canon.",
            "used_references": [], "timing_option": "segments-7",
            "references": {"7": "/tmp/placeholder.png"}, "segments": pending}
    if quantity > 1:
        plan["variations"] = [{"script": "pending", "segments": pending}] * (quantity - 1)
    return plan


@pytest.mark.parametrize("quantity", [1, 5])
def test_progress_note_is_an_unfinished_plan(quantity):
    assert "progress note" in editing.unfinished_plan(_placeholder_plan(quantity))


def test_plan_with_any_prompt_text_is_not_unfinished():
    plan = _placeholder_plan(5)
    plan["variations"][-1] = {"script": "s", "segments": [{"sections": {"summary": "A candle burns."}}]}
    assert editing.unfinished_plan(plan) is None
    assert editing.unfinished_plan({"script": "s", "used_references": [], "shot": {}}) is None
    # Empty or malformed plans keep the ordinary correction pass.
    assert editing.unfinished_plan({"segments": []}) is None
    assert editing.unfinished_plan({"segments": [{"sections": {"summary": ""}}]}) is None
