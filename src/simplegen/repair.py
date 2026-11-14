"""Repair a prepared run's model selections as a linked revision, without an agent pass.

Deterministic only: the user picks each replacement from the target's own
choices; nothing is chosen automatically. The source project is never
modified. A revision is built in a staging folder, checked with Phase 1
preflight, and published with one atomic rename.
"""

from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import os
import re
import shutil
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from .preflight import (
    _MODEL_INPUTS,
    FAIL,
    PASS,
    UNVERIFIED,
    _choices,
    receipt,
    run_saved_preflight,
    target_schema,
)
from .settings import Settings

STAGING_PREFIX = ".repair-staging-"
PROVENANCE = "repair, no agent pass"
COPIED = ("plan.json", "script.txt", "used_references.txt")


class RepairRefused(ValueError):
    """The run is not eligible for repair; the message says why."""


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _now() -> str:
    return datetime.now(UTC).isoformat()


def submitted(folder: Path, state: dict) -> bool:
    """True once any part of the run reached (or may have reached) ComfyUI."""
    if state.get("prompt_id") or state.get("submission_started"):
        return True
    for child in state.get("sequence", {}).get("children", []):
        try:
            sub = json.loads((Path(folder) / child["folder"] / "state.json").read_text(encoding="utf-8"))
        except (OSError, ValueError, KeyError, TypeError):
            return True  # Unknown is never "not submitted".
        if sub.get("prompt_id") or sub.get("submission_started"):
            return True
    return False


def refusal(folder: Path, state: dict) -> str | None:
    """Why this saved run cannot be repaired, or ``None`` when it can."""
    from .h3scene_compat import normalize_state
    state = normalize_state(state)
    from .h3scene_family import is_scene_chain
    snapshot = Path(folder) / "template.original.json"
    try:
        scene_snapshot = is_scene_chain(json.loads(snapshot.read_text()))
    except (OSError, ValueError, TypeError, AttributeError):
        scene_snapshot = False
    if state.get("h3scene_review") or (state.get("workflow_acceptance") or {}).get("scene_chain") or scene_snapshot:
        return "Repair is not supported for H3 scene chains; use Edit as new run."
    if state.get("sequence_parent") == "..":
        return "This is one variation of a Qty > 1 sequence; repairing sequences is not yet supported."
    if state.get("batch_parent"):
        return "This run belongs to a batch; batch members cannot be repaired here."
    if state.get("settings", {}).get("quantity", 1) != 1 or state.get("sequence"):
        return "Repairing Qty > 1 sequences is not yet supported."
    if submitted(folder, state):
        return "This run was already submitted to ComfyUI; use Resume to reconnect to it."
    if state.get("status") == "complete":
        return "This run is already complete."
    if not state.get("workflow_acceptance"):
        return ("This run has no accepted workflow to repair. Use Edit as new run "
                "(needs a new agent pass).")
    if state["workflow_acceptance"].get("single_shot"):
        return ("Repairing single-shot voice runs is not yet supported. Use Edit as new run with the "
                "workflow for this destination (needs a new agent pass).")
    return None


def _read(folder: Path, name: str) -> bytes:
    path = folder / name
    if path.is_symlink() or not path.is_file():
        raise RepairRefused(f"Accepted work is missing {name}; use Edit as new run.")
    return path.read_bytes()


def _load_source(folder: Path) -> dict:
    """Read and verify the accepted work; raises ``RepairRefused``."""
    try:
        state = json.loads((folder / "state.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise RepairRefused(f"Saved run cannot be read: {error}") from error
    reason = refusal(folder, state)
    if reason:
        raise RepairRefused(reason)
    accepted = state["workflow_acceptance"]
    files = {name: _read(folder, name) for name in ("workflow.json", "plan.json", "template.original.json")}
    if (_digest(files["workflow.json"]) != accepted.get("workflow_sha256")
            or _digest(files["plan.json"]) != accepted.get("plan_sha256")):
        raise RepairRefused("Accepted workflow or response changed after validation; refusing to repair.")
    if state.get("template_sha256") and _digest(files["template.original.json"]) != state["template_sha256"]:
        raise RepairRefused("Original workflow snapshot changed; refusing to repair.")
    return {"state": state, "files": files,
            "workflow": json.loads(files["workflow.json"]), "template": json.loads(files["template.original.json"])}


# Model selections


def _containers(workflow: dict):
    """Yield (container id, nodes) for the top level and every subgraph definition."""
    if isinstance(workflow.get("nodes"), list):
        yield "", {str(n["id"]): n for n in workflow["nodes"]}
        for definition in workflow.get("definitions", {}).get("subgraphs", []):
            yield str(definition["id"]), {str(n["id"]): n for n in definition.get("nodes", [])}
    else:  # API graph
        yield "", {str(k): v for k, v in workflow.items() if isinstance(v, dict) and "class_type" in v}


def _values(node: dict, info: dict) -> tuple[dict, set]:
    """Decoded model-bearing inputs and the names fed by links (not editable here)."""
    from .workflows import _widgets
    if "class_type" in node:
        inputs = node.get("inputs", {})
        return ({k: v for k, v in inputs.items() if not isinstance(v, list)},
                {k for k, v in inputs.items() if isinstance(v, list)})
    linked = {i["name"] for i in node.get("inputs", []) if i.get("link") is not None}
    return _widgets(node, info), linked


def key(container: str, node_id: str, name: str) -> str:
    return f"{container}|{node_id}|{name}"


def selections(workflow: dict, object_info: dict) -> list[dict]:
    """Every editable model selection with its current value and the target's choices."""
    from .workflows import _expanded_specs
    found = []
    for container, nodes in _containers(workflow):
        for node_id, node in nodes.items():
            kind = node.get("class_type") or node.get("type")
            if kind not in object_info or node.get("mode", 0) != 0:
                continue
            try:
                values, linked = _values(node, object_info[kind])
                specs, _ = _expanded_specs(object_info[kind], values)
            except (ValueError, KeyError, TypeError):
                continue
            for name, value in values.items():
                if name not in _MODEL_INPUTS or name in linked or name not in specs or not isinstance(value, str):
                    continue
                choices = _choices(specs[name])
                if choices is None:
                    continue
                choices = [c for c in choices if isinstance(c, str)]
                title = node.get("title") or node.get("_meta", {}).get("title") or kind
                found.append({"key": key(container, node_id, name), "container": container,
                              "node_id": node_id, "class_type": kind, "title": title, "input": name,
                              "current": value, "available": value in choices, "choices": choices})
    return found


def _patch(workflow: dict, selection: dict, new: str, object_info: dict) -> None:
    """Change exactly one selector; every other decoded input must stay the same."""
    nodes = dict(_containers(workflow)).get(selection["container"], {})
    node = nodes.get(selection["node_id"])
    kind = selection["class_type"]
    name, old = selection["input"], selection["current"]
    if node is None or (node.get("class_type") or node.get("type")) != kind:
        raise RepairRefused(f"Cannot locate {kind} node {selection['node_id']} in the workflow.")
    if "class_type" in node:
        if node.get("inputs", {}).get(name) != old:
            raise RepairRefused(f"{kind} node {selection['node_id']} {name} changed; reopen repair.")
        node["inputs"][name] = new
        return
    before, _ = _values(node, object_info[kind])
    if before.get(name) != old:
        raise RepairRefused(f"{kind} node {selection['node_id']} {name} changed; reopen repair.")
    values = node.get("widgets_values")
    if isinstance(values, dict):
        values[name] = new
        return
    expected = {**before, name: new}
    for index, value in enumerate(values or []):
        if value != old:
            continue
        trial = copy.deepcopy(node)
        trial["widgets_values"][index] = new
        try:
            if _values(trial, object_info[kind])[0] == expected:
                values[index] = new
                return
        except ValueError:
            continue
    raise RepairRefused(f"Cannot locate the {name} selector of {kind} node {selection['node_id']}.")


# Inspect and apply


def _schema(settings: Settings, object_info, comfy_client):
    if object_info is not None:
        return object_info, {"source": "supplied"}
    return target_schema(settings, comfy_client=comfy_client)


def inspect(folder: Path, *, object_info: dict | None = None, comfy_client=None) -> dict:
    """List the accepted workflow's model selections against the run's own destination."""
    from .preflight import schema_hash, target_for
    folder = Path(folder).expanduser().resolve()
    source = _load_source(folder)
    settings = Settings(**source["state"]["settings"])
    schema, meta = _schema(settings, object_info, comfy_client)
    result = {"folder": str(folder), "run_id": source["state"]["run_id"], "target": target_for(settings),
              "schema_source": meta.get("source"), "error": meta.get("error"),
              "schema_sha256": schema_hash(schema) if schema is not None else None, "selections": []}
    if schema is not None:
        result["selections"] = selections(source["workflow"], schema)
    return result


def _write(path: Path, data: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def _dump(value) -> bytes:
    return json.dumps(value, indent=2, ensure_ascii=False).encode("utf-8")


def _fsync_dir(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@contextmanager
def _parent_lock(parent: Path):
    """Serialize repairs publishing into one output folder and clean their leftovers."""
    with (parent / ".simplegen-repair.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            for leftover in parent.glob(STAGING_PREFIX + "*"):
                if leftover.is_dir() and not leftover.is_symlink():
                    shutil.rmtree(leftover, ignore_errors=True)
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def apply(folder: Path, replacements: dict[str, str], *, override_unverified: bool = False,
          object_info: dict | None = None, comfy_client=None, **preflight_kwargs) -> dict:
    """Publish a linked revision with the chosen models, or return why not.

    Returns ``{"status": "published", "folder", "run_id", "preflight"}`` or
    ``{"status": "fail" | "unverified", "preflight"}`` with nothing published.
    """
    from .comfy import new_id
    from .engine import run_lock
    from .workflow_edits import validate_edited_workflow
    folder = Path(folder).expanduser().resolve()
    if not isinstance(replacements, dict) or not replacements:
        raise RepairRefused("Choose at least one replacement model.")
    with run_lock(folder):
        source = _load_source(folder)
        state = source["state"]
        settings = Settings(**state["settings"])
        schema, meta = _schema(settings, object_info, comfy_client)
        if schema is None:
            raise RepairRefused(f"Target models cannot be verified: {meta.get('error')}")
        found = {s["key"]: s for s in selections(source["workflow"], schema)}
        workflow, template = copy.deepcopy(source["workflow"]), copy.deepcopy(source["template"])
        template_selections = {s["key"]: s for s in selections(template, schema)}
        changes = []
        for identity, new in replacements.items():
            selection = found.get(identity)
            if selection is None:
                raise RepairRefused(f"Unknown model selection: {identity}")
            if not isinstance(new, str) or new not in selection["choices"]:
                raise RepairRefused(f"{selection['class_type']} node {selection['node_id']} {selection['input']}: "
                                    f"'{new}' is not one of the target's available models.")
            if new == selection["current"]:
                continue
            _patch(workflow, selection, new, schema)
            twin = template_selections.get(identity)
            if twin is not None:
                if twin["current"] != selection["current"]:
                    raise RepairRefused(f"Workflow and original snapshot disagree on {identity}; cannot repair.")
                _patch(template, twin, new, schema)
            changes.append({"key": identity, "container": selection["container"], "node_id": selection["node_id"],
                            "class_type": selection["class_type"], "input": selection["input"],
                            "old": selection["current"], "new": new})
        if not changes:
            raise RepairRefused("No model selection changed.")
        try:
            prepared = validate_edited_workflow(template, copy.deepcopy(workflow), schema,
                                                megapixels=settings.megapixels, duration=settings.duration,
                                                aspect_ratio=settings.aspect_ratio, steps=settings.steps,
                                                run_id=state["run_id"], check_model_availability=False)
        except ValueError as error:
            raise RepairRefused(f"Repaired workflow no longer validates: {error}") from error

        workflow_bytes, template_bytes = _dump(workflow), _dump(template)
        run_id = new_id()
        stamp = datetime.now(UTC)
        slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", state["concept"].strip())[:45].strip("-") or "concept"
        parent = folder.parent
        final = parent / f"{stamp:%Y%m%d-%H%M%S}-repair-{slug}-{run_id[:8]}"
        acceptance = {"valid": True, "attempt": state["workflow_acceptance"].get("attempt"),
                      "workflow_sha256": _digest(workflow_bytes),
                      "plan_sha256": _digest(source["files"]["plan.json"]),
                      "provenance": PROVENANCE, "accepted_at": stamp.isoformat(),
                      "source_receipt": state["workflow_acceptance"]}
        new_state = {
            "schema_version": state.get("schema_version", 4), "run_id": run_id, "client_id": new_id(),
            "created_at": stamp.isoformat(), "settings": state["settings"], "concept": state["concept"],
            "status": "validated", "downloads": {},
            "template_path": state.get("template_path"), "template_sha256": _digest(template_bytes),
            "workflow_acceptance": acceptance,
            "repaired_from": {
                "run_id": state["run_id"], "folder": str(folder), "at": stamp.isoformat(),
                "old_workflow_sha256": _digest(source["files"]["workflow.json"]),
                "new_workflow_sha256": _digest(workflow_bytes),
                "old_template_sha256": _digest(source["files"]["template.original.json"]),
                "new_template_sha256": _digest(template_bytes),
                "changes": changes, "target_schema_sha256": None,
                "previous": state.get("repaired_from"),
            },
        }
        for name in ("final_node", "summary"):
            if name in state:
                new_state[name] = state[name]
        from .preflight import schema_hash
        new_state["repaired_from"]["target_schema_sha256"] = schema_hash(schema)

        files = {name: source["files"].get(name) or (folder / name).read_bytes()
                 for name in COPIED if (folder / name).is_file()}
        files.update({"template.original.json": template_bytes, "workflow.json": workflow_bytes,
                      "validation.json": _dump(acceptance)})
        files["references.json"] = _references(folder, run_id, settings, prepared,
                                               json.loads(source["files"]["plan.json"]), final)
        result = _publish_pinned(final, files, new_state, schema=schema, override_unverified=override_unverified,
                                 **preflight_kwargs)
        return {**result, "changes": changes} if result["status"] == "published" else result


def _references(folder: Path, run_id: str, settings: Settings, prepared: dict, plan: dict, final: Path) -> bytes:
    """Phase 6: pin the revision's references, reusing the source's verified cache.

    References the source never cached are read from the originals now; an
    unreadable one refuses the revision (retry once the folder is available).
    """
    from . import engine, refcache
    roots = refcache.settings_roots(settings)
    try:
        manifest = refcache.snapshot(run_id, refcache.run_values(prepared, plan), roots,
                                     lambda value: engine._inside(value, roots),
                                     reuse=refcache.load(folder), folder=final)
    except refcache.ReferenceCacheError as error:
        raise RepairRefused(str(error)) from None
    return _dump(manifest)


def _publish_pinned(final: Path, files: dict, new_state: dict, **kwargs) -> dict:
    """``_publish``; a revision that is not published drops the pin taken for it."""
    from .refcache import release_quietly
    try:
        result = _publish(final, files, new_state, **kwargs)
    except BaseException:
        release_quietly([new_state["run_id"]])
        raise
    if result["status"] != "published":
        release_quietly([new_state["run_id"]])
    return result


def _publish(final: Path, files: dict, new_state: dict, *, schema: dict | None, check: bool = True,
             override_unverified: bool = False, **preflight_kwargs) -> dict:
    """Stage every file, optionally preflight, then publish with one atomic rename.

    An interrupted publish leaves nothing or a complete revision. ``check``
    gates on Phase 1 preflight (``fail`` never publishes; ``unverified`` needs
    a recorded override).
    """
    parent = final.parent
    with _parent_lock(parent):
        staging = parent / f"{STAGING_PREFIX}{uuid.uuid4().hex}"
        staging.mkdir()
        for name, data in files.items():
            _write(staging / name, data)
        _write(staging / "state.json", _dump(new_state))
        result = None
        if check:
            result = run_saved_preflight(staging, object_info=schema, **preflight_kwargs)
            status = result["status"]
            if status == FAIL or (status == UNVERIFIED and not override_unverified) or status not in (
                    PASS, UNVERIFIED, FAIL):
                shutil.rmtree(staging, ignore_errors=True)
                return {"status": status, "preflight": result}
            new_state["repair_preflight"] = receipt(result, override_unverified=status == UNVERIFIED)
            (staging / "state.json").unlink()
            _write(staging / "state.json", _dump(new_state))
        _fsync_dir(staging)
        os.rename(staging, final)  # Publish: all or nothing.
        _fsync_dir(parent)
    return {"status": "published", "folder": str(final), "run_id": new_state["run_id"], "preflight": result}


def unlinked_revision(source: Path, linked: set[str]) -> dict | None:
    """The newest published revision of ``source`` that no schedule record links yet.

    Lets the Scheduled-list flow reuse it instead of creating a duplicate.
    Returns ``{"folder", "run_id", "preflight", "override_unverified"}``.
    """
    source = Path(source).expanduser().resolve()
    linked = {str(Path(path).resolve()) for path in linked if path}
    found = []
    for candidate in source.parent.glob("*-repair-*"):
        if candidate.name.startswith(".") or not candidate.is_dir() or str(candidate.resolve()) in linked:
            continue
        try:
            state = json.loads((candidate / "state.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        origin = state.get("repaired_from") or {}
        if (Path(origin.get("folder", "")).resolve() != source or refusal(candidate, state)
                or state.get("schedule") or not state.get("repair_preflight")):
            continue
        found.append((state.get("created_at", ""), candidate, state))
    if not found:
        return None
    _, candidate, state = max(found, key=lambda item: item[0])
    saved = state["repair_preflight"]
    return {"folder": str(candidate), "run_id": state["run_id"], "preflight": saved["result"],
            "override_unverified": bool(saved.get("override_unverified"))}


# Phase 3: salvage (schemas 2-4) and retarget (schema 5) without an agent pass

SALVAGE_PROVENANCE = "salvage, no agent pass"
RETARGET_PROVENANCE = "retarget, no agent pass"


def _revision_path(folder: Path, label: str, concept: str, run_id: str, stamp: datetime) -> Path:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", concept.strip())[:45].strip("-") or "concept"
    return folder.parent / f"{stamp:%Y%m%d-%H%M%S}-{label}-{slug}-{run_id[:8]}"


def _chain(workflow: dict, cards: list[dict]) -> list[dict]:
    """Card order from the saved links; else from unique "SEGMENT NN" titles; else refuse."""
    by_id = {card["id"]: card for card in cards}
    try:
        by_target = {(link[3], link[4]): link for link in workflow.get("links", [])}
        previous = {}
        for card in cards:
            slot = next(i for i, inp in enumerate(card["inputs"]) if inp["name"] == "previous")
            edge = by_target.get((card["id"], slot))
            previous[card["id"]] = edge[1] if edge else None
        ordered, cursor = [], None
        while len(ordered) < len(cards):
            matches = [i for i in by_id if i not in ordered and previous[i] == cursor]
            if len(matches) != 1:
                raise ValueError
            cursor = matches[0]
            ordered.append(cursor)
        return [by_id[i] for i in ordered]
    except (ValueError, KeyError, TypeError, IndexError, StopIteration):
        pass
    numbers = {}
    for card in cards:
        match = re.match(r"\s*SEGMENT\s+(\d+)", card.get("title", ""), re.IGNORECASE)
        if not match:
            break
        numbers[card["id"]] = int(match.group(1))
    if len(numbers) == len(cards) and len(set(numbers.values())) == len(cards):
        return sorted(cards, key=lambda card: numbers[card["id"]])
    raise RepairRefused("The saved segments' order cannot be determined reliably; use Edit as new run.")


def salvage_refusal(folder: Path, state: dict, attempt: int | None = None) -> str | None:
    """Why saved agent work cannot be rebuilt, or None."""
    if state.get("schema_version") not in (2, 3, 4):
        return "Only runs from the previous workflow-editing format (schemas 2-4) can be rebuilt."
    if state.get("sequence_parent") == ".." or state.get("settings", {}).get("quantity", 1) != 1 \
            or state.get("sequence"):
        return "Rebuilding Qty > 1 sequences is not yet supported."
    attempts = state.get("workflow_edit_attempts", [])
    if attempt is None:
        if state.get("batch_parent"):
            return "This run belongs to a batch; batch members cannot be rebuilt here."
        if submitted(folder, state):
            return "This run was already submitted to ComfyUI; use Resume."
        if state.get("workflow_acceptance"):
            return "This run already has an accepted workflow; use Resume or Repair."
        if not attempts or attempts[-1].get("status") != "invalid":
            return "This run has no rejected agent attempt to rebuild."
        return None
    if not any(a.get("number") == attempt and a.get("status") == "invalid" for a in attempts):
        return f"Attempt {attempt} is not a rejected agent attempt."
    return None


def salvage(folder: Path, *, attempt: int | None = None, object_info: dict | None = None,
            comfy_client=None) -> dict:
    """Rebuild a schema 5 revision from a rejected attempt's content; no agent call.

    Extracts segment prompts, the script and reference paths, picks the timing
    option with the same segment count, then lints, compiles and validates
    against the run's own template snapshot before publishing a linked revision.
    """
    from .comfy import new_id
    from .compiler import (
        COMPILER_VERSION,
        analyze,
        check_plan,
        compile_workflow,
        dumps,
        parse_sections,
    )
    from .editing import normalize_used_references, read_json
    from .engine import STATE_VERSION, run_lock
    from .workflow_edits import validate_edited_workflow
    folder = Path(folder).expanduser().resolve()
    with run_lock(folder):
        try:
            state = json.loads((folder / "state.json").read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise RepairRefused(f"Saved run cannot be read: {error}") from error
        reason = salvage_refusal(folder, state, attempt)
        if reason:
            raise RepairRefused(reason)
        number = attempt or state["workflow_edit_attempts"][-1]["number"]
        workspace = folder / "agent-edits" / f"attempt-{number}"
        try:
            edited = read_json(workspace / "workflow.json")
            saved_plan = read_json(workspace / "plan.json")
            template_bytes = _read(folder, "template.original.json")
            template = json.loads(template_bytes)
        except (OSError, ValueError) as error:
            raise RepairRefused(f"Saved agent work is incomplete: {error}") from error
        if state.get("template_sha256") and _digest(template_bytes) != state["template_sha256"]:
            raise RepairRefused("Original workflow snapshot changed; refusing to rebuild.")
        settings = Settings(**state["settings"])
        schema, meta = _schema(settings, object_info, comfy_client)
        if schema is None:
            raise RepairRefused(f"The destination's node schema cannot be read: {meta.get('error')}")
        run_id = new_id()
        try:
            info = analyze(template, schema, settings, run_id)
        except ValueError as error:
            raise RepairRefused(f"The run's workflow is not supported by the compiler: {error}") from error
        cards = [n for n in edited.get("nodes", []) if n.get("type") == info["segment_type"]]
        if not cards:
            raise RepairRefused("The saved attempt has no segment cards.")
        segments = []
        for card in _chain(edited, cards):
            names = [i["name"] for i in card.get("inputs", []) if i.get("widget")]
            prompt = dict(zip(names, card.get("widgets_values", []))).get("prompt")
            if not isinstance(prompt, str):
                raise RepairRefused(f"Segment {card['id']} has no prompt.")
            segments.append({"sections": parse_sections(prompt, info["sections"], info["raw_prompt"])})
        option = next((o for o in info["options"] if o["segment_count"] == len(segments)), None)
        if option is None:
            raise RepairRefused(
                f"The saved work has {len(segments)} segments, but no legal timing layout for "
                f"{settings.duration:g}s has that many ({', '.join(o['id'] for o in info['options'])}). "
                "Use Edit as new run.")
        loaders = {str(n["id"]): n for n in edited.get("nodes", []) if n.get("type") == "LoadImage"}
        references = {}
        for slot in info["reference_slots"]:
            values = loaders.get(slot["node_id"], {}).get("widgets_values") or [None]
            if isinstance(values[0], str):
                references[slot["node_id"]] = values[0]
        plan = {"script": saved_plan.get("script", ""), "used_references": saved_plan.get("used_references", []),
                "timing_option": option["id"], "references": references, "segments": segments}
        roots = settings.reference_roots()
        plan, _ = normalize_used_references(plan, folder, state, roots)
        warnings: list[str] = []
        try:
            # Pacing written before speech budgets existed is kept, with a recorded warning.
            check_plan(plan, info, concept=state["concept"], roots=roots, quantity=1, warnings=warnings,
                       budget_errors=False)
            workflow = compile_workflow(template, plan, settings, run_id, object_info=schema)
            prepared = validate_edited_workflow(template, copy.deepcopy(workflow), schema,
                                                megapixels=settings.megapixels, duration=settings.duration,
                                                aspect_ratio=settings.aspect_ratio, steps=settings.steps,
                                                run_id=run_id, check_model_availability=False)
        except ValueError as error:
            raise RepairRefused(f"Saved agent work cannot be rebuilt: {error}") from error
        stamp = datetime.now(UTC)
        workflow_bytes, plan_bytes = dumps(workflow), _dump(plan)
        acceptance = {"valid": True, "attempt": None, "compiler_version": COMPILER_VERSION,
                      "timing_option": option["id"], "workflow_sha256": _digest(workflow_bytes),
                      "plan_sha256": _digest(plan_bytes), "provenance": SALVAGE_PROVENANCE,
                      "accepted_at": stamp.isoformat(), "source_attempt": number}
        new_state = {
            "schema_version": STATE_VERSION, "run_id": run_id, "client_id": new_id(),
            "created_at": stamp.isoformat(), "settings": state["settings"], "concept": state["concept"],
            "status": "validated", "downloads": {},
            "template_path": state.get("template_path"), "template_sha256": _digest(template_bytes),
            "workflow_acceptance": acceptance,
            "repaired_from": {
                "kind": "salvage", "run_id": state["run_id"], "folder": str(folder), "at": stamp.isoformat(),
                "attempt": number, "source_schema_version": state.get("schema_version"),
                "old_workflow_sha256": _digest((workspace / "workflow.json").read_bytes()),
                "new_workflow_sha256": _digest(workflow_bytes),
                "old_template_sha256": _digest(template_bytes), "new_template_sha256": _digest(template_bytes),
                "timing_option": option["id"], "previous": state.get("repaired_from"),
                "lint_warnings": warnings,
            },
        }
        files = {"plan.json": plan_bytes, "template.original.json": template_bytes,
                 "workflow.json": workflow_bytes, "validation.json": _dump(acceptance)}
        final = _revision_path(folder, "salvage", state["concept"], run_id, stamp)
        files["references.json"] = _references(folder, run_id, settings, prepared, plan, final)
        # Model availability is not gated here: Resume preflights, and Repair fixes models.
        return _publish_pinned(final, files, new_state, schema=schema, check=False)


def retarget(folder: Path, template_path: Path, *, override_unverified: bool = False,
             object_info: dict | None = None, comfy_client=None, **preflight_kwargs) -> dict:
    """Recompile a schema 5 run's saved plan against another workflow file (no agent)."""
    from dataclasses import replace

    from .comfy import new_id
    from .compiler import COMPILER_VERSION, analyze, check_plan, compile_workflow, dumps
    from .editing import read_json
    from .engine import run_lock
    from .workflow_edits import validate_edited_workflow
    folder = Path(folder).expanduser().resolve()
    template_path = Path(template_path).expanduser().resolve()
    with run_lock(folder):
        source = _load_source(folder)
        state = source["state"]
        if state.get("schema_version", 1) < 5:
            raise RepairRefused("Only compiled (schema 5) runs can switch workflow files; use Edit as new run "
                                "(needs a new agent pass), or Rebuild from saved agent work first.")
        plan = json.loads(source["files"]["plan.json"])
        if not isinstance(plan.get("segments"), list) or not isinstance(plan.get("references"), dict):
            raise RepairRefused("The saved plan is not a creative plan; use Edit as new run.")
        try:
            new_template = read_json(template_path)
            template_bytes = template_path.read_bytes()
        except (OSError, ValueError) as error:
            raise RepairRefused(f"Workflow file cannot be read: {error}") from error
        old_settings = Settings(**state["settings"])
        settings = replace(old_settings, template_file=str(template_path), template_folder=str(template_path.parent))
        schema, meta = _schema(settings, object_info, comfy_client)
        if schema is None:
            raise RepairRefused(f"The destination's node schema cannot be read: {meta.get('error')}")
        run_id = new_id()
        try:
            info = analyze(new_template, schema, settings, run_id)
        except ValueError as error:
            raise RepairRefused(f"That workflow is not supported by the compiler: {error}. "
                                "Use Edit as new run.") from error
        used = {name for segment in plan["segments"] for name in segment.get("sections", {})}
        if used != set(info["sections"]):
            raise RepairRefused("That workflow uses different prompt sections ("
                                + ", ".join(info["sections"]) + "); use Edit as new run (needs a new agent pass).")
        if len(info["reference_slots"]) != len(plan["references"]):
            raise RepairRefused(f"That workflow has {len(info['reference_slots'])} reference slots, the saved plan "
                                f"{len(plan['references'])}; use Edit as new run (needs a new agent pass).")
        try:
            old_order = [s["node_id"] for s in analyze(source["template"], schema, old_settings)["reference_slots"]]
        except ValueError:
            old_order = sorted(plan["references"], key=int)
        references = {new["node_id"]: plan["references"][old]
                      for new, old in zip(info["reference_slots"], old_order)}
        option = next((o for o in info["options"] if o["id"] == plan.get("timing_option")), None) or next(
            (o for o in info["options"] if o["segment_count"] == len(plan["segments"])), None)
        if option is None:
            raise RepairRefused("That workflow has no timing layout with the saved segment count; "
                                "use Edit as new run.")
        retargeted = {**plan, "references": references, "timing_option": option["id"]}
        roots = [Path(settings.project_folder).expanduser().resolve(), template_path.parent]
        try:
            check_plan(retargeted, info, concept=state["concept"], roots=roots, quantity=settings.quantity)
            workflow = compile_workflow(new_template, retargeted, settings, run_id, object_info=schema)
            prepared = validate_edited_workflow(new_template, copy.deepcopy(workflow), schema,
                                                megapixels=settings.megapixels, duration=settings.duration,
                                                aspect_ratio=settings.aspect_ratio, steps=settings.steps,
                                                run_id=run_id, check_model_availability=False)
        except ValueError as error:
            raise RepairRefused(f"The saved plan does not fit that workflow: {error}") from error
        stamp = datetime.now(UTC)
        workflow_bytes, plan_bytes = dumps(workflow), _dump(retargeted)
        acceptance = {"valid": True, "attempt": state["workflow_acceptance"].get("attempt"),
                      "compiler_version": COMPILER_VERSION, "timing_option": option["id"],
                      "workflow_sha256": _digest(workflow_bytes), "plan_sha256": _digest(plan_bytes),
                      "provenance": RETARGET_PROVENANCE, "accepted_at": stamp.isoformat(),
                      "source_receipt": state["workflow_acceptance"]}
        new_state = {
            "schema_version": state["schema_version"], "run_id": run_id, "client_id": new_id(),
            "created_at": stamp.isoformat(), "settings": settings.public_dict(), "concept": state["concept"],
            "status": "validated", "downloads": {},
            "template_path": str(template_path), "template_sha256": _digest(template_bytes),
            "workflow_acceptance": acceptance,
            "repaired_from": {
                "kind": "retarget", "run_id": state["run_id"], "folder": str(folder), "at": stamp.isoformat(),
                "old_template_path": state.get("template_path"), "new_template_path": str(template_path),
                "old_workflow_sha256": _digest(source["files"]["workflow.json"]),
                "new_workflow_sha256": _digest(workflow_bytes),
                "old_template_sha256": _digest(source["files"]["template.original.json"]),
                "new_template_sha256": _digest(template_bytes), "timing_option": option["id"],
                "previous": state.get("repaired_from"),
            },
        }
        files = {"plan.json": plan_bytes, "template.original.json": template_bytes,
                 "workflow.json": workflow_bytes, "validation.json": _dump(acceptance)}
        for name in ("script.txt", "used_references.txt"):
            if (folder / name).is_file():
                files[name] = (folder / name).read_bytes()
        final = _revision_path(folder, "retarget", state["concept"], run_id, stamp)
        files["references.json"] = _references(folder, run_id, settings, prepared, retargeted, final)
        return _publish_pinned(final, files, new_state, schema=schema, override_unverified=override_unverified,
                               **preflight_kwargs)
