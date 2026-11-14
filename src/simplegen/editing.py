"""Durable, bounded direct JSON editing; validation never repairs the agent's graph."""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import UTC, datetime
from pathlib import Path

from . import agents
from .h3scene_compat import normalize_state
from .settings import atomic_json
from .workflow_edits import edit_contract, validate_edited_workflow


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _settings_digest(state: dict) -> str:
    return hashlib.sha256(json.dumps(state["settings"], sort_keys=True,
                                     ensure_ascii=False).encode()).hexdigest()


def read_json(path: Path) -> dict:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Expected a regular JSON file: {path.name}")
    if path.stat().st_size > 32 * 1024 * 1024:
        raise ValueError(f"JSON file is too large: {path.name}")

    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    def invalid_constant(value):
        raise ValueError(f"Invalid JSON number: {value}")

    result = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=unique,
        parse_constant=invalid_constant,
    )
    if not isinstance(result, dict):
        raise ValueError(f"Expected a JSON object in {path.name}")  # noqa: TRY004 -- invalid JSON document
    return normalize_state(result) if path.name == "state.json" else result


READ_ONLY_COPY = "workflow.readonly.json"
_PLACEHOLDER_TEXT = frozenset({"pending", "todo", "tbd", "placeholder", "...", "…"})


def _template_source(folder: Path, state: dict, roots: list[Path]) -> str | None:
    """The linked template the run's snapshot came from, if it still matches the receipt."""
    try:
        source = Path(state["template_path"]).expanduser().resolve()
        snapshot = folder / "template.original.json"
        if (not any(source.is_relative_to(root) for root in roots) or not source.is_file()
                or digest(snapshot) != state.get("template_sha256")
                or read_json(source) != read_json(snapshot)):
            return None
    except (KeyError, TypeError, ValueError, OSError):
        return None
    return str(source)


def normalize_used_references(plan, folder: Path, state: dict, roots: list[Path]) -> tuple:
    """Resolve the run's own files in the agent's used_references, in code.

    The agent is told to list everything it consulted, which includes the
    read-only template copy in its workspace. That copy is inside the job
    folder, not a linked folder, so it can never pass ``engine._inside``. It is
    replaced by the linked template it was copied from, when the copy, the
    snapshot and the source all still match the run's receipt; otherwise it is
    dropped. Any other path inside the job folder is a working file, not a
    production reference, and is dropped. Every other entry is left exactly as
    returned and validated as before. Returns (plan, changes); the plan is a
    new dict when anything changed, and the agent's own plan.json is untouched.
    """
    values = plan.get("used_references") if isinstance(plan, dict) else None
    if not isinstance(values, list):
        return plan, []
    folder = Path(folder).resolve()
    kept, changes, template = [], [], None
    for value in values:
        resolved = None
        if isinstance(value, str):
            try:
                path = Path(value).expanduser()
                resolved = path.resolve() if path.is_absolute() else None
            except (OSError, RuntimeError):
                resolved = None
        if resolved is None or not resolved.is_relative_to(folder):
            kept.append(value)
            continue
        copy_of_template = False
        if resolved.name == READ_ONLY_COPY and resolved.parent.parent == folder / "agent-edits":
            try:
                copy_of_template = digest(resolved) == state.get("template_sha256")
            except OSError:
                copy_of_template = False
        if copy_of_template:
            template = template or _template_source(folder, state, roots)
        if copy_of_template and template:
            if template not in kept:
                kept.append(template)
            changes.append({"path": value, "action": "replaced", "with": template})
        else:
            changes.append({"path": value, "action": "dropped"})
    if not changes:
        return plan, []
    return {**plan, "used_references": kept}, changes


def unfinished_plan(plan) -> str | None:
    """Why a compiled-chain response is a progress note rather than a plan, or None.

    Deliberately narrow so a real plan is never misread: unfinished means no
    segment section anywhere carries prompt text and at least one holds a
    placeholder word such as "pending". A plan that is merely empty or
    malformed stays an ordinary rejection. Plans without ``segments`` (the
    single-shot family) are never classified here.
    """
    if not isinstance(plan, dict) or "segments" not in plan:
        return None
    groups, placeholder = [plan["segments"]], False
    if isinstance(plan.get("variations"), list):
        groups.extend(v.get("segments") for v in plan["variations"] if isinstance(v, dict))
    for segments in groups:
        for segment in segments if isinstance(segments, list) else []:
            sections = segment.get("sections") if isinstance(segment, dict) else None
            for text in sections.values() if isinstance(sections, dict) else []:
                word = text.strip().lower() if isinstance(text, str) else ""
                if word in _PLACEHOLDER_TEXT:
                    placeholder = True
                elif word:
                    return None
    if not placeholder:
        return None
    return ("The response was a progress note, not the final plan: no segment has prompt text "
            "(every section is empty or a placeholder such as 'pending'). Consult the references, "
            "then return the complete plan with real paths and prompt text.")


def inspect_saved_response(folder: Path) -> dict:
    """Read the exact saved format complaint and artifacts eligible for recovery."""
    folder = Path(folder).expanduser().resolve()
    state = read_json(folder / "state.json")
    if (state.get("prompt_id") or state.get("submission_started")
            or state.get("workflow_acceptance") or state.get("sequence")
            or state.get("status") == "complete"):
        raise ValueError("This run has already reached submission; use Resume.")
    attempts = state.get("workflow_edit_attempts", [])
    attempt = attempts[-1] if attempts else None
    if (not attempt or attempt.get("status") != "invalid"
            or "Return a nonempty script and used_references" not in attempt.get("error", "")):
        raise ValueError("No saved response-format rejection is available for this run.")
    workspace = folder / "agent-edits" / f"attempt-{attempt['number']}"
    response = read_json(workspace / "plan.json")
    required = {"script", "used_references"} | ({"variations"} if state["settings"].get("quantity", 1) > 1 else set())
    missing = sorted(required - response.keys())
    if missing:
        raise ValueError("Saved response is missing required field(s): " + ", ".join(missing))
    if (not isinstance(response["script"], str) or not response["script"].strip()
            or not isinstance(response["used_references"], list)
            or not all(isinstance(p, str) for p in response["used_references"])):
        raise ValueError("Saved response has an invalid script or used_references list.")
    extra = sorted(response.keys() - required)
    if not extra:
        raise ValueError("Saved response has no harmless extra fields to override.")
    read_json(workspace / "workflow.json")
    checks = {
        "template_sha256": digest(folder / "template.original.json"),
        "workflow_sha256": digest(workspace / "workflow.json"),
        "plan_sha256": digest(workspace / "plan.json"),
        "settings_sha256": _settings_digest(state),
    }
    return {"attempt": attempt["number"], "reason": attempt["error"],
            "extra_fields": extra, **checks,
            "fingerprint": hashlib.sha256(json.dumps([attempt, checks, state.get("status")],
                                               sort_keys=True).encode()).hexdigest()}


def request_saved_response_override(folder: Path, expected: str | None = None) -> dict:
    """Record an explicit request to revalidate saved agent work without another agent call."""
    from .engine import run_lock

    folder = Path(folder).expanduser().resolve()
    with run_lock(folder):
        preview = inspect_saved_response(folder)
        if expected is not None and preview["fingerprint"] != expected:
            raise ValueError("Saved response changed while approval was open. Inspect it again.")
        state = read_json(folder / "state.json")
        state["response_override_request"] = {
            **preview, "requested_at": datetime.now(UTC).isoformat(),
        }
        atomic_json(folder / "state.json", state)
        return {"folder": str(folder)}


def _snapshot(folder, state, source, persist) -> dict:
    """The run's immutable original template snapshot (created once, receipted by hash)."""
    snapshot = folder / "template.original.json"
    if not snapshot.exists():
        original = read_json(source)
        expected = hashlib.sha256(
            json.dumps(original, indent=2, ensure_ascii=False, allow_nan=False).encode("utf-8")
        ).hexdigest()
        if state.get("template_sha256") not in (None, expected):
            raise ValueError("Original snapshot is missing and the linked source has changed.")
        state["template_path"] = str(source)
        state["template_sha256"] = expected
        persist("preparing")
        atomic_json(snapshot, original)
    elif not state.get("template_sha256") and not state.get("workflow_edit_attempts"):
        # Recover the previous creation ordering only before any agent ran.
        if read_json(snapshot) != read_json(source):
            raise ValueError("Unreceipted original snapshot differs from the linked template.")
        state["template_path"] = str(source)
        state["template_sha256"] = digest(snapshot)
        persist("preparing")
    template = read_json(snapshot)
    if digest(snapshot) != state.get("template_sha256"):
        raise ValueError("Original workflow snapshot changed; refusing to validate against it.")
    return template


def _capture(settings, folder, state, prepared, plan) -> None:
    """Snapshot every reference before acceptance (Phase 6).

    A reference that cannot be read raises ``ReferenceCacheError``: the attempt
    stays pending (not invalid), so Resume re-validates it without an agent turn.
    """
    from .refcache import capture_run
    capture_run(folder, state["run_id"], settings, prepared, plan)


def prepare_edit(settings, folder, state, info, cancel, emit, persist, source, brief, references):
    """Return accepted graph/response/transfers; reserve each attempt before spending usage."""
    snapshot = folder / "template.original.json"
    template = _snapshot(folder, state, source, persist)
    kwargs = {
        "megapixels": settings.megapixels,
        "duration": settings.duration,
        "aspect_ratio": settings.aspect_ratio,
        "steps": settings.steps,
        "run_id": state["run_id"],
        "check_model_availability": False,
    }
    contract = edit_contract(template, info, **kwargs)
    atomic_json(folder / "editing-contract.json", contract)
    roots = settings.reference_roots()

    def validate(workflow, response):
        if not isinstance(response, dict):
            raise ValueError("Agent response must be a JSON object.")  # noqa: TRY004 -- provider data error
        response, _ = normalize_used_references(response, folder, state, roots)
        required = {"script", "used_references"} | ({"variations"} if settings.quantity > 1 else set())
        missing = sorted(required - response.keys())
        if missing:
            raise ValueError("Agent response is missing required field(s): " + ", ".join(missing))
        if not isinstance(response["script"], str) or not response["script"].strip():
            raise ValueError("Agent response needs a nonempty script.")
        if (not isinstance(response["used_references"], list)
                or not all(isinstance(p, str) for p in response["used_references"])):
            raise ValueError("Agent response needs a list of used_references paths.")
        prepared = validate_edited_workflow(template, workflow, info, **kwargs)
        if settings.quantity > 1:
            from .variations import validate_variations, variant_workflow
            validate_variations(workflow, response, contract, settings.quantity)
            for index, variant in enumerate(response["variations"], 2):
                native = variant_workflow(workflow, variant["prompts"], contract["segment_type"],
                                          f"{state['run_id']}-{index}")
                validate_edited_workflow(template, native, info, **kwargs)
        uploads, used = references(prepared, response)
        return prepared, response, uploads, used

    accepted = state.get("workflow_acceptance")
    if accepted:
        native, plan = folder / "workflow.json", folder / "plan.json"
        if digest(native) != accepted["workflow_sha256"] or digest(plan) != accepted["plan_sha256"]:
            raise ValueError(
                "Accepted workflow or response changed after validation; refusing submission."
            )
        return validate(read_json(native), read_json(plan))

    override = state.get("response_override_request")
    if override:
        attempt = next((a for a in state.get("workflow_edit_attempts", [])
                        if a.get("number") == override.get("attempt")
                        and a.get("status") == "invalid"), None)
        if attempt is None or "Return a nonempty script and used_references" not in attempt.get("error", ""):
            raise ValueError("Saved response override is no longer eligible.")
        workspace = folder / "agent-edits" / f"attempt-{attempt['number']}"
        native, plan = workspace / "workflow.json", workspace / "plan.json"
        checks = {"template_sha256": digest(snapshot),
                  "workflow_sha256": digest(native), "plan_sha256": digest(plan),
                  "settings_sha256": _settings_digest(state)}
        if any(override.get(key) != value for key, value in checks.items()):
            raise ValueError("Saved response, workflow, template or settings changed after approval.")
        edited, response = read_json(native), read_json(plan)
        result = validate(edited, response)
        response = result[1]
        _capture(settings, folder, state, result[0], response)
        atomic_json(folder / "workflow.json", edited)
        atomic_json(folder / "plan.json", response)
        receipt = {"valid": True, "attempt": attempt["number"],
                   "workflow_sha256": digest(folder / "workflow.json"),
                   "plan_sha256": digest(folder / "plan.json"),
                   "response_override": {**override, "accepted_at": datetime.now(UTC).isoformat()}}
        atomic_json(folder / "validation.json", receipt)
        state["workflow_acceptance"] = receipt
        state.pop("response_override_request", None)
        state.pop("error", None)
        attempt["status"] = "valid_override"
        persist("validated")
        return result

    attempts = state.setdefault("workflow_edit_attempts", [])
    prior_workflow = snapshot
    feedback = ""
    prior_response = None
    while True:
        # A completed subprocess can leave its artifacts before the desktop exits.
        # Recover those without spending another agent turn or resetting the budget.
        pending = attempts[-1] if attempts and attempts[-1]["status"] == "running" else None
        if pending:
            attempt = pending
            workspace = folder / "agent-edits" / f"attempt-{attempt['number']}"
        else:
            if len(attempts) >= 2:
                raise ValueError(
                    "Workflow validation failed after one correction: "
                    + (feedback or attempts[-1].get("error", "See validation.json."))
                )
            if cancel.is_set():
                raise agents.Cancelled("Cancelled before workflow editing.")
            if attempts:
                prior = folder / "agent-edits" / f"attempt-{attempts[-1]['number']}"
                previous = prior / "workflow.json"
                if previous.is_file() and not previous.is_symlink():
                    prior_workflow = previous
                feedback = attempts[-1].get("error", feedback)
                try:
                    prior_response = read_json(prior / "plan.json")
                except (ValueError, OSError):
                    prior_response = None
            number = len(attempts) + 1
            workspace = folder / "agent-edits" / f"attempt-{number}"
            if workspace.exists():
                # The CLI starts only after its reservation persists. Preserve an
                # unreserved directory left by a crash, then recreate from source.
                if workspace.is_symlink() or not workspace.is_dir():
                    raise ValueError("Unexpected path at the reserved agent workspace.")
                workspace.rename(workspace.with_name(f"unreserved-{number}-{uuid.uuid4().hex[:8]}"))
            workspace.mkdir(parents=True, exist_ok=False)
            (workspace / "workflow.json").write_bytes(prior_workflow.read_bytes())
            attempt = {"number": number, "status": "running"}
            attempts.append(attempt)
            persist("editing")
            request = {
                **brief,
                "workflow_file": str(workspace / "workflow.json"),
                "editing_contract": contract,
                "validation_error": feedback,
                "previous_script": prior_response.get("script", "") if prior_response else "",
                "previous_used_references": prior_response.get("used_references", [])
                if prior_response
                else [],
                **({"previous_variations": prior_response.get("variations", [])}
                   if prior_response and settings.quantity > 1 else {}),
            }
            before = digest(source) if source.is_file() else None
            emit(
                {
                    "stage": "Editing workflow",
                    "message": f"Agent editing project JSON · attempt {number}/2",
                    "progress": None,
                }
            )
            try:
                agents.generate_plan(settings, workspace, request, cancel, emit)
            except agents.InvalidAgentResponse as exc:
                attempt["response_error"] = str(exc)
            except agents.AgentPreflightError as exc:
                # No generation started: preserve its workspace for diagnostics,
                # but do not spend a workflow correction on a login/CLI failure.
                archived = workspace.with_name(f"preflight-{number}-{uuid.uuid4().hex[:8]}")
                workspace.rename(archived)
                state.setdefault("agent_preflight_failures", []).append({
                    **attempt, "status": "preflight_failed", "error": str(exc),
                    "workspace": str(archived),
                })
                attempts.pop()
                persist("failed")
                raise
            # Other provider errors escape without an automatic retry. A later
            # explicit Resume consumes at most the one remaining reservation.
            finally:
                if before is not None and (not source.is_file() or digest(source) != before):
                    attempt.update(
                        status="failed",
                        error="The linked source template was changed during agent execution.",
                    )
                    persist("failed")
                    raise ValueError(attempt["error"])

        try:
            if attempt.get("response_error"):
                raise ValueError(attempt["response_error"])
            response_path = workspace / "plan.json"
            if not response_path.is_file():
                raise ValueError(
                    "Previous agent attempt did not finish its response; complete the workflow and response."
                )
            response = read_json(response_path)
            edited = read_json(workspace / "workflow.json")
            result = validate(edited, response)
            response = result[1]
        except (ValueError, KeyError, TypeError, IndexError, OSError) as exc:
            feedback = str(exc)
            attempt.update(status="invalid", error=feedback)
            atomic_json(workspace / "validation.json", {"valid": False, "error": feedback})
            persist("validating")
            emit(
                {
                    "stage": "Validation",
                    "message": f"Workflow edit rejected: {feedback}",
                    "progress": None,
                }
            )
            continue
        _capture(settings, folder, state, result[0], response)
        atomic_json(folder / "workflow.json", edited)
        atomic_json(folder / "plan.json", response)
        receipt = {
            "valid": True,
            "attempt": attempt["number"],
            "workflow_sha256": digest(folder / "workflow.json"),
            "plan_sha256": digest(folder / "plan.json"),
        }
        atomic_json(workspace / "validation.json", receipt)
        atomic_json(folder / "validation.json", receipt)
        state["workflow_acceptance"] = receipt
        attempt["status"] = "valid"
        persist("validated")
        return result


def prepare_compiled(settings, folder, state, info, cancel, emit, persist, source, brief, references):
    """Schema 5: the agent returns a creative plan (read-only); code compiles the workflow.

    Same reservation, persistence and receipt pattern as ``prepare_edit``: each
    attempt is persisted before the CLI starts, at most one correction pass, and
    the accepted plan and compiled workflow are hashed in ``workflow_acceptance``.
    """
    import copy

    normalized = normalize_state(state)
    state.clear()
    state.update(normalized)
    from . import h3scene_family, singleshot
    from .compiler import (
        COMPILER_VERSION,
        analyze,
        check_plan,
        check_shape,
        compile_workflow,
        workflow_brief,
    )
    template = _snapshot(folder, state, source, persist)
    if h3scene_family.is_scene_chain(template):
        return _prepare_scene(settings, folder, state, info, cancel, emit, persist, template, references)
    if singleshot.is_single_shot(template):
        return _prepare_single_shot(settings, folder, state, info, cancel, emit, persist, template, brief,
                                    references)
    kwargs = {
        "megapixels": settings.megapixels,
        "duration": settings.duration,
        "aspect_ratio": settings.aspect_ratio,
        "steps": settings.steps,
        "run_id": state["run_id"],
        "check_model_availability": False,
    }
    roots = settings.reference_roots()
    analysis = analyze(template, info, settings, state["run_id"])

    warnings: list[str] = []

    def build(plan, *, lint=True):
        warnings.clear()
        if lint:
            check_plan(plan, analysis, concept=state["concept"], roots=roots, quantity=settings.quantity,
                       warnings=warnings)
        else:
            check_shape(plan, analysis, settings.quantity)
        workflow = compile_workflow(template, plan, settings, state["run_id"], object_info=info)
        prepared = validate_edited_workflow(template, copy.deepcopy(workflow), info, **kwargs)
        if settings.quantity > 1:
            for index, variant in enumerate(plan["variations"], 2):
                variant_id = f"{state['run_id']}-{index}"
                native = compile_workflow(template, plan, settings, variant_id, object_info=info,
                                          segments=variant["segments"])
                validate_edited_workflow(template, native, info, **{**kwargs, "run_id": variant_id})
        uploads, used = references(prepared, plan)
        return workflow, prepared, uploads, used

    accepted = state.get("workflow_acceptance")
    if accepted:
        native, plan_path = folder / "workflow.json", folder / "plan.json"
        if digest(native) != accepted["workflow_sha256"] or digest(plan_path) != accepted["plan_sha256"]:
            raise ValueError("Accepted workflow or response changed after validation; refusing submission.")
        plan = read_json(plan_path)
        if not isinstance(plan.get("script"), str) or not isinstance(plan.get("used_references"), list):
            raise ValueError("Accepted plan is missing its script or used_references.")
        if settings.quantity > 1:
            check_shape(plan, analysis, settings.quantity)
        prepared = validate_edited_workflow(template, read_json(native), info, **kwargs)
        uploads, used = references(prepared, plan)
        return prepared, plan, uploads, used

    def request_for(read_only):
        return workflow_brief(template, info, settings, state["run_id"], read_only)

    def receipt_for(plan):
        return {"compiler_version": COMPILER_VERSION, "timing_option": plan["timing_option"]}

    return _plan_attempts(settings, folder, state, cancel, emit, persist, brief, request_for,
                          build, receipt_for, warnings)


def _prepare_single_shot(settings, folder, state, info, cancel, emit, persist, template, brief, references):
    """An h3voice single-shot template: the agent returns a shot spec; code builds the graph.

    workflow.json holds the compiled API graph (reference inputs are absolute
    sources until upload); the receipt records the h3voice format and profile.
    """
    from . import singleshot
    from .preflight import target_for
    problem = singleshot.settings_problem(settings)
    if not problem and target_for(settings) == "cloud" and singleshot.multigpu_nodes(template):
        problem = "Cloud runs need the cloud single-shot workflow (core loaders); this one uses MultiGPU loaders."
    if problem:
        raise ValueError(problem)
    roots = settings.reference_roots()
    warnings: list[str] = []

    def build(plan):
        warnings.clear()
        spec, text = singleshot.check(plan, settings, state["run_id"], concept=state["concept"], roots=roots,
                                      warnings=warnings)
        graph = singleshot.compile_graph(template, spec, text, settings, state["run_id"], info)
        singleshot.verify(graph, template)
        prepared = singleshot.prepared(graph)
        uploads, used = references(prepared, plan)
        return graph, prepared, uploads, used

    accepted = state.get("workflow_acceptance")
    if accepted:
        native, plan_path = folder / "workflow.json", folder / "plan.json"
        if digest(native) != accepted["workflow_sha256"] or digest(plan_path) != accepted["plan_sha256"]:
            raise ValueError("Accepted workflow or response changed after validation; refusing submission.")
        plan = read_json(plan_path)
        if not isinstance(plan.get("script"), str) or not isinstance(plan.get("used_references"), list):
            raise ValueError("Accepted plan is missing its script or used_references.")
        graph = read_json(native)
        singleshot.verify(graph, template)
        prepared = singleshot.prepared(graph)
        uploads, used = references(prepared, plan)
        return prepared, plan, uploads, used

    def request_for(read_only):
        return singleshot.brief(template, settings, state["run_id"], read_only)

    def receipt_for(plan):
        return {"single_shot": singleshot.KIND, "h3voice_format": singleshot.FORMAT,
                "profile": singleshot.profile(template).get("name")}

    return _plan_attempts(settings, folder, state, cancel, emit, persist, brief, request_for,
                          build, receipt_for, warnings)


def _prepare_scene(settings, folder, state, info, cancel, emit, persist, template, references):
    from . import h3scene_family as scene
    from .h3scene_review import OverflowReview
    from .preflight import target_for
    problem = scene.settings_problem(settings, target_for(settings))
    if state.get("batch_parent"):
        problem = "H3 scene chains do not support batches; use a local Qty 1 run."
    if state.get("h3scene_decision", {}).get("action") == "deny":
        problem = "This H3 Scene Chain plan was denied; use Edit as new run."
    if problem:
        raise ValueError(problem)
    roots = [Path(settings.project_folder).expanduser().resolve()]
    warnings = []

    def build(plan):
        warnings.clear()
        approved = state.get("h3scene_decision", {}).get("action") == "approve"
        if approved:
            if digest(folder / "plan.json") != state["h3scene_decision"]["plan_sha256"]:
                raise OverflowReview("Approved H3 Scene Chain plan changed; use Edit as new run.")
            # Approval covers the canonical reviewed plan, never an unreceipted attempt copy.
            reviewed = read_json(folder / "plan.json")
            plan.clear()
            plan.update(reviewed)
        duration = scene.check(plan, settings, state["run_id"], concept=state["concept"], roots=roots,
                               warnings=warnings, template=template)
        if duration["needs_review"] and not approved:
            atomic_json(folder / "plan.json", plan)
            state["h3scene_review"] = {"reason": plan["overflow"]["reason"],
                                     "plan_sha256": digest(folder / "plan.json"), **duration}
            persist("needs_review")
            raise OverflowReview(plan["overflow"]["reason"])
        requested = scene.envelope(template, settings)
        total = sum(segment["seconds"] for segment in plan["segments"])
        tested = scene.envelope(template, settings, duration=total)
        if (requested["verified"] and not tested["verified"]) or (
                tested["ram_warning"] and not requested["ram_warning"]):
            warnings.append(f"H3 Scene Chain plan total {total:g}s: {tested['message']}")
        graph = scene.compile_graph(template, plan, settings, state["run_id"], info, warnings=warnings)
        scene.verify(graph, template)
        from .workflows import validate_graph
        validate_graph(graph, info)
        prepared = scene.prepared(graph, plan)
        uploads, used = references(prepared, plan)
        return graph, prepared, uploads, used

    accepted = state.get("workflow_acceptance")
    if accepted:
        if digest(folder / "workflow.json") != accepted["workflow_sha256"] or digest(folder / "plan.json") != accepted["plan_sha256"]:
            raise ValueError("Accepted H3 Scene Chain workflow or plan changed; refusing submission.")
        graph, plan = read_json(folder / "workflow.json"), read_json(folder / "plan.json")
        scene.verify(graph, template)
        prepared = scene.prepared(graph, plan)
        uploads, used = references(prepared, plan)
        return prepared, plan, uploads, used

    def request_for(read_only):
        return scene.brief(template, settings, state["run_id"], read_only, concept=state["concept"])

    result = _plan_attempts(settings, folder, state, cancel, emit, persist, {}, request_for, build,
                            lambda plan: {"scene_chain": scene.KIND, "h3scene_format": scene.FORMAT}, warnings,
                            allow_unfinished=False)
    for warning in warnings:
        if warning.startswith("H3 Scene Chain plan total"):
            emit({"stage": "Validation", "message": f"Warning: {warning}", "progress": None})
    return result


def _plan_attempts(settings, folder, state, cancel, emit, persist, brief, request_for, build, receipt_for,
                   warnings, *, allow_unfinished=True):
    """Creative-plan attempts: reserve, run the agent read-only, validate, correct once, accept.

    ``request_for(read_only_path)`` adds the template-specific brief; ``build(plan)``
    returns (workflow, prepared, uploads, used) or raises ValueError for the
    correction pass; ``receipt_for(plan)`` adds template-specific receipt fields.

    One "unfinished" response (``unfinished_plan``) per run is re-requested
    without spending the correction; a second one is an ordinary rejection, so
    a run makes at most three agent calls.
    """
    import os

    from .compiler import dumps
    from .scheduling import atomic_bytes
    roots = settings.reference_roots()
    attempts = state.setdefault("workflow_edit_attempts", [])
    feedback, prior_plan = "", None
    while True:
        pending = attempts[-1] if attempts and attempts[-1]["status"] == "running" else None
        counted = sum(1 for a in attempts if a.get("status") not in ("unfinished", "running"))
        if pending:
            attempt = pending
            workspace = folder / "agent-edits" / f"attempt-{attempt['number']}"
        else:
            if counted >= 2:
                raise ValueError("Creative plan was rejected after one correction: "
                                 + (feedback or attempts[-1].get("error", "See validation.json.")))
            if cancel.is_set():
                raise agents.Cancelled("Cancelled before agent preparation.")
            if attempts:
                feedback = attempts[-1].get("error", feedback)
                try:
                    prior_plan = read_json(folder / "agent-edits" / f"attempt-{attempts[-1]['number']}" / "plan.json")
                except (ValueError, OSError):
                    prior_plan = None
                if attempts[-1].get("status") == "unfinished":
                    prior_plan = None  # Placeholders are not a draft worth correcting.
            number = len(attempts) + 1
            workspace = folder / "agent-edits" / f"attempt-{number}"
            if workspace.exists():
                if workspace.is_symlink() or not workspace.is_dir():
                    raise ValueError("Unexpected path at the reserved agent workspace.")
                workspace.rename(workspace.with_name(f"unreserved-{number}-{uuid.uuid4().hex[:8]}"))
            workspace.mkdir(parents=True, exist_ok=False)
            read_only = workspace / "workflow.readonly.json"
            read_only.write_bytes((folder / "template.original.json").read_bytes())
            os.chmod(read_only, 0o444)
            attempt = {"number": number, "status": "running", "mode": "creative_plan"}
            attempts.append(attempt)
            persist("editing")
            request = {
                **brief,
                **request_for(read_only),
                "validation_error": feedback,
                "previous_plan": prior_plan or {},
            }
            rerun = " · retrying an unfinished response" if attempts[-2:-1] and \
                attempts[-2].get("status") == "unfinished" else ""
            emit({"stage": "Writing plan",
                  "message": f"Agent writing script and prompts · attempt {counted + 1}/2{rerun}",
                  "progress": None})
            try:
                agents.generate_plan(settings, workspace, request, cancel, emit)
            except agents.InvalidAgentResponse as exc:
                attempt["response_error"] = str(exc)
            except agents.AgentPreflightError as exc:
                archived = workspace.with_name(f"preflight-{number}-{uuid.uuid4().hex[:8]}")
                workspace.rename(archived)
                state.setdefault("agent_preflight_failures", []).append({
                    **attempt, "status": "preflight_failed", "error": str(exc), "workspace": str(archived),
                })
                attempts.pop()
                persist("failed")
                raise
        try:
            if attempt.get("response_error"):
                raise ValueError(attempt["response_error"])
            if not (workspace / "plan.json").is_file():
                raise ValueError("Previous agent attempt did not finish its creative plan.")
            plan = read_json(workspace / "plan.json")
            unfinished = unfinished_plan(plan) if allow_unfinished else None
            if unfinished:
                if not any(a.get("status") == "unfinished" for a in attempts):
                    attempt.update(status="unfinished", error=unfinished)
                    atomic_json(workspace / "validation.json",
                                {"valid": False, "unfinished": True, "error": unfinished})
                    persist("validating")
                    emit({"stage": "Validation", "message": "Agent returned an unfinished plan; "
                          "asking again without using the correction", "progress": None})
                    continue
                raise ValueError(unfinished)
            plan, normalized = normalize_used_references(plan, folder, state, roots)
            if normalized:
                attempt["used_references_normalized"] = normalized
            workflow, prepared, uploads, used = build(plan)
        except (ValueError, KeyError, TypeError, IndexError, OSError) as exc:
            if state.get("h3scene_decision", {}).get("action") == "approve":
                raise ValueError(f"Approved H3 Scene Chain plan cannot be prepared: {exc}") from exc
            feedback = str(exc)
            attempt.update(status="invalid", error=feedback)
            atomic_json(workspace / "validation.json", {"valid": False, "error": feedback})
            persist("validating")
            emit({"stage": "Validation", "message": f"Creative plan rejected: {feedback}", "progress": None})
            continue
        _capture(settings, folder, state, prepared, plan)
        atomic_bytes(folder / "workflow.json", dumps(workflow))
        atomic_json(folder / "plan.json", plan)
        receipt = {"valid": True, "attempt": attempt["number"], **receipt_for(plan), "lint_warnings": list(warnings),
                   "workflow_sha256": digest(folder / "workflow.json"), "plan_sha256": digest(folder / "plan.json")}
        if attempt.get("used_references_normalized"):
            receipt["used_references_normalized"] = attempt["used_references_normalized"]
        atomic_json(workspace / "validation.json", receipt)
        atomic_json(folder / "validation.json", receipt)
        state["workflow_acceptance"] = receipt
        attempt["status"] = "valid"
        persist("validated")
        return prepared, plan, uploads, used
