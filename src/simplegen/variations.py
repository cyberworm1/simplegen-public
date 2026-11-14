"""One creative preparation, followed by durable, sequential individual runs."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import replace

from . import agents
from .editing import digest, read_json
from .lifecycle import Detached, Interrupt
from .settings import atomic_json
from .workflow_edits import validate_edited_workflow


def validate_variations(workflow, plan, contract, quantity):
    variants = plan.get("variations")
    if not isinstance(variants, list) or len(variants) != quantity - 1:
        raise ValueError(f"Return exactly {quantity - 1} additional variations after the base workflow.")
    expected = {str(n["id"]) for n in workflow["nodes"] if n["type"] == contract["segment_type"]}
    for index, variant in enumerate(variants, 2):
        if (not isinstance(variant, dict) or set(variant) != {"script", "prompts"}
                or not isinstance(variant["script"], str) or not variant["script"].strip()
                or not isinstance(variant["prompts"], list)):
            raise ValueError(f"Variation {index} needs a script and segment prompts.")
        seen = set()
        for prompt in variant["prompts"]:
            if not isinstance(prompt, dict) or set(prompt) != {"node_id", "text"}:
                raise ValueError("Each variation prompt needs only node_id and text.")
            node_id, text = prompt["node_id"], prompt["text"]
            if not isinstance(node_id, str) or node_id not in expected or node_id in seen:
                raise ValueError(f"Variation {index} has an unknown or duplicate native segment ID.")
            if not isinstance(text, str) or not text.strip() or text.count("<d>") != text.count("</d>"):
                raise ValueError(f"Variation {index} has an empty prompt or unbalanced dialogue tags.")
            seen.add(node_id)
        if seen != expected:
            raise ValueError(f"Variation {index} must supply a prompt for every native segment.")


def variant_workflow(base, prompts, segment_type, run_id):
    """Change only promoted prompts/seeds; topology and other settings stay exact."""
    result = copy.deepcopy(base)
    replacements = {p["node_id"]: p["text"] for p in prompts}
    for node in result["nodes"]:
        if node["type"] != segment_type:
            continue
        names = [i["name"] for i in node["inputs"] if i.get("widget")]
        for key in ("prompt", "seed"):
            if key not in names:
                continue
            if key == "prompt":
                value = replacements.get(str(node["id"]), node["widgets_values"][names.index(key)])
            else:
                # H3RCNoise caps seeds at JavaScript's safe integer range (2**53 - 1).
                value = int(hashlib.sha256(f"{run_id}:{node['id']}".encode()).hexdigest()[:13], 16)
            node["widgets_values"][names.index(key)] = value
            if isinstance(node.get("widgets_values_named"), dict):
                node["widgets_values_named"][key] = value
    return result


def run_sequence(settings, concept, folder, state, on_event, cancel, detach, choose_gpu,
                 *, prepare_only=False, batch=None, batch_phase="run"):
    from .engine import STATE_VERSION, _run_locked, _verified_receipts, new_id, now, run_lock

    interrupt = Interrupt(cancel, detach)

    def persist(status=None):
        if status:
            state["status"] = status
        state["updated_at"] = now()
        atomic_json(folder / "state.json", state)

    def emit(event):
        event = {**event, "folder": str(folder)}
        event["message"] = agents.redact(str(event.get("message", "")))
        with (folder / "sequence-events.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"time": now(), **event}, ensure_ascii=False) + "\n")
        on_event(event)

    session = None
    prepared_info = None
    try:
        if not state.get("sequence", {}).get("ready"):
            bundle = _run_locked(settings, concept, folder, state, emit, cancel, detach,
                                 choose_gpu, prepare_only=True)
            prepared_info = bundle["info"]
            interrupt.raise_if_set()
            sequence = state.setdefault("sequence", {"children": [
                {"folder": f"variation-{i + 1:03d}", "run_id": new_id(), "client_id": new_id()}
                for i in range(settings.quantity)
            ]})
            persist("preparing_variations")
            base = read_json(folder / "workflow.json")
            original = read_json(folder / "template.original.json")
            plan = bundle["plan"]
            compiled = None
            if state.get("schema_version", 1) >= 5:
                # Schema 5: compile each variation from its own segments (shared layout
                # and references); seeds follow variant_workflow's rule.
                from .compiler import compile_variations
                compiled = compile_variations(original, plan, settings,
                                              [c["run_id"] for c in sequence["children"]],
                                              object_info=bundle["info"])
            else:
                contract = read_json(folder / "editing-contract.json")
            for index, child in enumerate(sequence["children"]):
                interrupt.raise_if_set()
                target = folder / child["folder"]
                target.mkdir(exist_ok=True)
                child_state_path = target / "state.json"
                if child_state_path.exists():
                    existing = read_json(child_state_path)
                    if existing["run_id"] != child["run_id"] or existing.get("sequence_parent") != "..":
                        raise ValueError("Variation identity differs from the saved sequence.")
                    continue
                variant = plan["variations"][index - 1] if index else {"script": plan["script"], "prompts": []}
                native = (compiled[index] if compiled is not None else
                          variant_workflow(base, variant["prompts"], contract["segment_type"], child["run_id"]))
                validate_edited_workflow(original, native, bundle["info"],
                    megapixels=settings.megapixels, duration=settings.duration,
                    steps=settings.steps,
                    aspect_ratio=settings.aspect_ratio, run_id=child["run_id"], check_model_availability=False)
                atomic_json(target / "template.original.json", original)
                atomic_json(target / "workflow.json", native)
                atomic_json(target / "plan.json", {"script": variant["script"], "used_references": plan["used_references"]})
                receipt = {"valid": True, "source": "sequence", "workflow_sha256": digest(target / "workflow.json"),
                           "plan_sha256": digest(target / "plan.json")}
                atomic_json(target / "validation.json", receipt)
                atomic_json(child_state_path, {
                    "schema_version": STATE_VERSION, "run_id": child["run_id"], "client_id": child["client_id"],
                    "created_at": now(), "settings": replace(settings, quantity=1).public_dict(),
                    "concept": concept, "sequence_parent": "..", "status": "validated", "downloads": {},
                    "template_path": state["template_path"], "template_sha256": digest(target / "template.original.json"),
                    "workflow_acceptance": receipt,
                })
            sequence["ready"] = True
            persist("prepared")

        sequence = state["sequence"]
        if len(sequence["children"]) != settings.quantity:
            raise ValueError("Saved variation count differs from the sequence settings.")
        if prepare_only:
            if prepared_info is None:
                from .global_session import session_for
                schema_session = session_for(settings, folder, state, persist, emit, cancel, interrupt.detach)
                try:
                    prepared_info = schema_session.schema()
                finally:
                    schema_session.close()
            return {"prepared": True, "folder": str(folder), "info": prepared_info}
        if batch_phase not in ("run", "stage", "submit") or (batch_phase != "run" and batch is None):
            raise ValueError("Batch staging and submission require an owning session")
        context = batch or {"folder": folder, "state": state, "persist": persist}
        results, receipts = [], []
        for index, child in enumerate(sequence["children"]):
            interrupt.raise_if_set()
            state["active_variation"] = index + 1
            persist("generating" if batch_phase == "run" else "staging" if batch_phase == "stage" else "queueing")

            def child_event(event, position=index):
                stage = event.get("stage", "")
                progress = event.get("progress")
                fraction = 0
                if stage == "Generating":
                    fraction = 0.05 + 0.8 * (progress or 0) / 100
                elif stage == "Downloading":
                    fraction = 0.85 + 0.14 * (progress or 0) / 100
                elif stage == "Complete":
                    fraction = 1
                emit({**event, "stage": f"Variation {position + 1} of {settings.quantity} · {stage}",
                      "progress": 100 * (position + fraction) / settings.quantity})

            target = folder / child["folder"]
            with run_lock(target):
                child_state = read_json(target / "state.json")
                if child_state["run_id"] != child["run_id"]:
                    raise ValueError("Variation identity differs from the saved sequence.")
                if (batch and not child_state.get("prompt_id") and not child_state.get("submission_started")
                        and batch.get("validate_pending")):
                    batch["validate_pending"]()
                result = _run_locked(replace(settings, quantity=1), concept, target, child_state,
                                     child_event, cancel, detach, choose_gpu, sequence=context,
                                     batch_phase=batch_phase)
                if batch_phase != "run":
                    continue
                receipts.extend(_verified_receipts(target, child_state))
            entry = {"name": f"Variation {index + 1}", "final_path": str(
                (target / child_state["result"]["final_path"]).relative_to(folder)), "metadata": result["metadata"]}
            results.append(entry)
            state["result"] = {**results[0], "folder": ".", "variations": results.copy()}
            persist()
            emit({"stage": "Saved", "message": f"Variation {index + 1} of {settings.quantity} verified",
                  "progress": 100 * (index + 1) / settings.quantity,
                  "completed_result": {**state["result"], "folder": str(folder)}})

        if batch_phase != "run":
            persist("staged" if batch_phase == "stage" else "queued")
            return {"folder": str(folder), "phase": batch_phase, "count": len(sequence["children"])}

        interrupt.raise_if_set()
        if settings.managed_runpod and batch is None:
            from .global_session import session_for
            session = session_for(settings, folder, state, persist, emit, cancel, interrupt.detach)
            persist("acknowledging_downloads")
            session.complete(receipts)
            from .resources import _elapsed, record_run
            children = [read_json(folder / c["folder"] / "state.json") for c in sequence["children"]]
            state["generation_started_at"] = children[0].get("generation_started_at")
            state["generation_completed_at"] = children[-1].get("generation_completed_at")
            state["generation_seconds"] = sum(
                _elapsed(c.get("generation_started_at"), c.get("generation_completed_at")) or 0
                for c in children)
            persist("complete")
            record_run(folder, state)
        persist("complete")
        report = f"SUCCESS\n{settings.quantity} variations saved and verified.\n\n" + "\n".join(
            f"{r['name']}: {folder / r['final_path']}" for r in results)
        if batch:
            report += "\nThis set is downloaded. Cloud retirement waits for all jobs in the owning batch."
        elif settings.managed_runpod:
            report += "\nAll downloads acknowledged. Cloud retirement follows five minutes with both queues empty."
        elif settings.cloud:
            report += "\nUnmanaged cloud instance billing may continue."
        (folder / "report.txt").write_text(report + "\n", encoding="utf-8")
        if batch is None:
            # The whole set is complete; a batch releases its members when it completes.
            from .refcache import release_quietly
            release_quietly([state["run_id"]])
        result = {**state["result"], "folder": str(folder)}
        emit({"stage": "Complete", "message": str(folder), "progress": 100})
        return result
    except Exception as error:  # noqa: BLE001 -- durable sequence failure boundary
        status = "detached" if interrupt.detach.is_set() or isinstance(error, Detached) else "cancelled" if cancel.is_set() else "failed"
        if status == "cancelled" and batch is None and settings.managed_runpod and state.get("managed_session", {}).get("sid"):
            from .global_session import session_for
            if session is None:
                session = session_for(settings, folder, state, persist, emit, cancel, interrupt.detach)
            try:
                session.shutdown()
            except Exception:  # noqa: BLE001 -- explicit cancellation must report unconfirmed retirement
                error = RuntimeError("Shutdown could not be confirmed. Session saved; inspect cloud status or intervene manually.")
        state["error"] = agents.redact(str(error))
        if status == "failed" and batch is None and settings.managed_runpod:
            from .cost_safety import record_owner_failure
            record_owner_failure(state, settings, state["error"], "run")  # Persisted just below.
        persist(status)
        report = f"{status.upper()}\n{state['error']}\n\nSequence saved: {folder}\nResume retains completed variations and existing job IDs.\n"
        if settings.cloud:
            report += "Cloud billing may continue; normal shutdown awaits all verified downloads.\n"
        (folder / "report.txt").write_text(report, encoding="utf-8")
        emit({"stage": status.capitalize(), "message": state["error"], "progress": None})
        raise RuntimeError(state["error"]) from None
    finally:
        if session:
            session.close()
