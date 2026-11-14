"""Add work to a running batch (inbox) and hand a warm pod to another batch.

Durable-intent rules: an addition is prepared (agent work) outside the batch
lock on its own run folder and only *requested* through an atomic inbox file;
the batch owner (``BatchStore.run_batch``) is the only membership writer. A
handoff persists its intent in the source batch before the target records the
adopted session; recovery finishes or rolls back an interrupted handoff.
"""

from __future__ import annotations

import copy
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import httpx

from .editing import read_json
from .settings import Settings, atomic_json


def _now() -> str:
    return datetime.now(UTC).isoformat()


# Validation shared by acceptance and handoff


def validate_prepared(target: Path, info: dict) -> None:
    """The prepared (compiled or accepted) workflows validate against ``info`` (e.g. the live pod)."""
    from .workflow_edits import validate_edited_workflow
    from .workflows import validate_graph
    child = read_json(target / "state.json")
    settings = Settings(**child["settings"])
    jobs = [target / item["folder"] for item in child.get("sequence", {}).get("children", [])] or [target]
    for job in jobs:
        job_state = read_json(job / "state.json")
        prepared = validate_edited_workflow(
            read_json(job / "template.original.json"), read_json(job / "workflow.json"), info,
            megapixels=settings.megapixels, duration=settings.duration, steps=settings.steps,
            aspect_ratio=settings.aspect_ratio, run_id=job_state["run_id"], check_model_availability=False)
        validate_graph(prepared["graph"], info)


def _volume(config_file: str) -> str:
    from .global_session import load_global_profile
    return load_global_profile(Path(config_file).expanduser()).volume_id


def _session_volume(state: dict) -> str | None:
    frozen = (state.get("managed_session") or {}).get("frozen_profile") or {}
    volume = (frozen.get("profile") or {}).get("volume_id")
    return volume or _volume(state["settings"]["runpod_config_file"])


def _submitted(target: Path, child: dict) -> bool:
    if child.get("prompt_id") or child.get("submission_started"):
        return True
    for item in child.get("sequence", {}).get("children", []):
        sub = target / item["folder"] / "state.json"
        if sub.is_file():
            progress = read_json(sub)
            if progress.get("prompt_id") or progress.get("submission_started"):
                return True
    return False


# Preparing and requesting an addition (outside the batch lock)


def prepare_addition(store, batch_id: str, concept: str, quantity: int, on_event, cancel, detach=None,
                     base: Settings | None = None) -> dict:
    """Create and prepare a new run with the batch's settings; the pod keeps rendering meanwhile."""
    from .batches import _fingerprint, _gate, member_preflight, reopen_refusal
    from .engine import _run_locked, create_run, run_lock
    from .variations import run_sequence
    batch = store.get(batch_id)
    if batch["status"] not in ("armed", "waiting", "running") and reopen_refusal(batch):
        raise ValueError(reopen_refusal(batch) + " Start a new batch for this concept.")
    if type(quantity) is not int or quantity < 1:
        raise ValueError("Qty must be at least 1")
    shared = Settings(**batch["settings"])
    settings = replace(base or shared, quantity=quantity, managed_runpod=True, global_desktop=True, cloud=True,
                       runpod_config_file=shared.runpod_config_file, runpod_schema_file=shared.runpod_schema_file,
                       runpod_credentials_file=shared.runpod_credentials_file)
    folder, state = create_run(settings, concept)
    state["pending_batch"] = batch_id
    atomic_json(folder / "state.json", state)
    _gate(member_preflight(folder))  # Phase 1 preflight against the cached cloud schema.
    with run_lock(folder):
        child = read_json(folder / "state.json")
        if settings.quantity > 1:
            run_sequence(settings, concept, folder, child, on_event, cancel, detach, None, prepare_only=True)
        else:
            _run_locked(settings, concept, folder, child, on_event, cancel, detach, prepare_only=True)
    from .cost_safety import batch_projection
    return {"folder": str(folder), "run_id": state["run_id"], "batch_id": batch_id,
            "fingerprint": _fingerprint(folder), "core_fingerprint": _fingerprint(folder, include_references=False),
            "projection": batch_projection(store.get(batch_id), extra_renders=quantity)}


def submit_addition(store, batch_id: str, folder: Path) -> Path:
    """Write one atomic inbox request; no batch lock is needed."""
    from .batches import _fingerprint
    from .engine import run_lock
    folder = Path(folder).expanduser().resolve()
    with run_lock(folder):
        child = read_json(folder / "state.json")
        if child.get("batch_parent") not in (None, batch_id) or child.get("pending_batch") not in (None, batch_id):
            raise ValueError("This run already belongs to another batch.")
        if _submitted(folder, child):
            raise ValueError("This run was already submitted.")
        child["pending_batch"] = batch_id
        atomic_json(folder / "state.json", child)
        request = {"run_id": child["run_id"], "folder": str(folder), "batch_id": batch_id,
                   "fingerprint": _fingerprint(folder), "core_fingerprint": _fingerprint(folder, include_references=False),
                   "quantity": Settings(**child["settings"]).quantity, "requested_at": _now()}
    inbox = store.inbox(batch_id)
    inbox.mkdir(exist_ok=True)
    path = inbox / f"{request['run_id']}.json"
    atomic_json(path, request)  # Same run_id → same file: duplicates are idempotent.
    return path


# Acceptance (batch owner, under the batch lock)


def accept_request(state: dict, request: dict, info: dict) -> dict:
    """Validate one request for the running batch; returns the member record or raises ValueError."""
    from .batches import _fingerprint
    from .cost_safety import batch_projection, over_cap, projection_text
    from .engine import run_lock
    if request.get("batch_id") not in (None, state["id"]):
        raise ValueError("The request is addressed to another batch.")
    target = Path(request.get("folder", "")).expanduser().resolve()
    try:
        with run_lock(target):
            child = read_json(target / "state.json")
            if child.get("run_id") != request.get("run_id"):
                raise ValueError("The requested run's identity changed.")
            if child.get("batch_parent") not in (None, state["id"]) or child.get("pending_batch") not in (None, state["id"]):
                raise ValueError("The run already belongs to another batch.")
            if _submitted(target, child) or child.get("managed_session", {}).get("create_intent"):
                raise ValueError("The run was already submitted or owns compute.")
            settings = Settings(**child["settings"])
            if not settings.managed_runpod or not settings.global_desktop:
                raise ValueError("Additions need managed Global RunPod settings.")
            if child["settings"].get("runpod_config_file") != state["settings"]["runpod_config_file"]:
                raise ValueError("The addition uses a different managed GPU profile.")
            if _volume(child["settings"]["runpod_config_file"]) != _session_volume(state):
                raise ValueError("The addition uses a different network volume.")
            if _fingerprint(target) != request.get("fingerprint"):
                raise ValueError("The addition changed after preparation; prepare it again.")
            try:
                validate_prepared(target, info)
            except (ValueError, KeyError, TypeError, OSError) as error:
                raise ValueError(f"The addition does not validate against the running pod: {error}") from error
    except (OSError, RuntimeError) as error:
        raise ValueError(f"The addition's run folder is unavailable: {error}") from error
    projection = batch_projection(state, extra_renders=settings.quantity)
    if over_cap(projection):
        raise ValueError("The addition would exceed the batch spend cap: " + projection_text(projection)
                         + ". Raise the cap and resubmit.")
    return {"id": child["run_id"], "folder": str(target), "concept": child["concept"], "status": "prepared",
            "revision": 1, "added_at": _now(), "prepared_fingerprint": request["fingerprint"],
            "prepared_core_fingerprint": _fingerprint(target, include_references=False),
            "projection_at_acceptance": projection}


def claim_child(target: Path, batch_id: str, run_id: str) -> None:
    """After acceptance is durable in the batch: the run records its batch parent."""
    from .engine import run_lock
    with run_lock(target):
        child = read_json(target / "state.json")
        if child.get("run_id") != run_id or child.get("batch_parent") not in (None, batch_id):
            raise ValueError("Accepted addition changed identity; review it")
        if child.get("batch_parent") == batch_id and "pending_batch" not in child:
            return
        child["batch_parent"] = batch_id
        child.pop("pending_batch", None)
        atomic_json(target / "state.json", child)


# Warm-pod handoff


def _warm(state: dict) -> bool:
    saved = state.get("managed_session") or {}
    return (saved.get("backend") == "global_desktop" and bool(saved.get("pod_id"))
            and saved.get("downloads_verified") is True and state.get("status") == "complete"
            and not saved.get("retired") and not saved.get("delete_intent") and not saved.get("handed_off")
            and not saved.get("handoff_intent") and not saved.get("owner_failure"))


def handoff_candidates(stores, exclude: Path) -> list[dict]:
    rows = []
    for store in stores:
        for row in store.list():
            if Path(row["folder"]).resolve() != Path(exclude).resolve() and _warm(row):
                rows.append(row)
    return rows


def _default_live_schema(session) -> dict:
    from . import engine
    comfy = engine.Comfy(session._endpoint(session.saved["pod_id"]))
    try:
        return comfy.object_info()
    finally:
        comfy.close()


def pod_prompt_ids(session) -> set:
    """Every prompt id in the pod's /queue and /history right now (raises if unreadable)."""
    from . import global_session
    username, password = session._login()
    monitor = global_session.ComfyMonitor(session.saved["pod_id"], username, password, [])
    try:
        queue = monitor.read("/queue")
        history = monitor.read("/history?max_items=1000")
    finally:
        monitor.client.close()
    lists = [queue.get(key) for key in ("queue_running", "queue_pending")] if isinstance(queue, dict) else [None]
    if not all(isinstance(items, list) for items in lists) or not isinstance(history, dict):
        raise ValueError("The warm pod's queue or history could not be read")
    entries = lists[0] + lists[1]
    if any(not isinstance(entry, list) or len(entry) < 2 for entry in entries):
        raise ValueError("The warm pod's queue could not be read")
    return {str(entry[1]) for entry in entries} | {str(key) for key in history}


def _job_prompt_ids(saved: dict) -> set:
    ids = set(saved.get("inherited_prompt_ids") or [])
    for job in (saved.get("jobs") or {}).values():
        value = job.get("prompt_id") if isinstance(job, dict) else job
        if value:
            ids.add(str(value))
    return ids


def compatibility(source: dict, target: dict) -> str | None:
    """Why ``target`` may not adopt ``source``'s pod (approvals, ceiling, profile, volume), or None."""
    saved = source["managed_session"]
    choice = next((c for c in target.get("gpu_choices", []) if c.get("gpu_id") == saved.get("gpu_id")), None)
    if choice is None:
        return "GPU type is not approved for this batch."
    rate = saved.get("actual_hourly")
    if not isinstance(rate, (int, float)) or rate > choice["max_hourly"] or rate > target.get("max_hourly", rate):
        return "The warm pod's hourly rate exceeds this batch's approved ceiling."
    if source["settings"].get("runpod_config_file") != target["settings"].get("runpod_config_file"):
        return "Different managed GPU profile."
    if _session_volume(source) != _volume(target["settings"]["runpod_config_file"]):
        return "Different network volume."
    return None


def perform_handoff(source_folder: Path, target_folder: Path, *, live_schema=None, crash_after: int | None = None) -> bool:
    """Target batch B adopts source batch A's warm pod. Locks both folders in sorted path order.

    Steps: (1) persist A.handoff_intent; (2) write B.managed_session with
    adopted_from; (3) registry: register B, remove A; (4) mark A handed_off.
    ``crash_after`` is for tests only.
    """
    from .engine import run_lock
    first, second = sorted([Path(source_folder).resolve(), Path(target_folder).resolve()], key=str)
    try:
        with run_lock(first), run_lock(second):
            return _handoff_locked(Path(source_folder).resolve(), Path(target_folder).resolve(),
                                   live_schema, crash_after)
    except RuntimeError as error:
        if "already running in another simpleGen window" in str(error):
            return False  # The poller or a worker holds a batch: no handoff this time.
        raise


def _handoff_locked(a_path: Path, b_path: Path, live_schema, crash_after) -> bool:
    from .global_registry import register, remove
    from .global_session import GlobalSession
    a, b = read_json(a_path / "state.json"), read_json(b_path / "state.json")
    if not _warm(a) or b.get("managed_session", {}).get("name") or b["status"] not in ("armed", "waiting"):
        return False
    if compatibility(a, b):
        return False
    session = GlobalSession(Settings(**a["settings"]), a_path, a, lambda: None, lambda _event: None, None, None)
    try:
        pod = session._owned_pod()
        from simplegen_runpod.provider import gpu_is_stopped
        if pod is None or gpu_is_stopped(pod):
            return False
        info = (live_schema or _default_live_schema)(session)
        # Everything already on the pod is inherited, not "unregistered"; unreadable → no handoff.
        inherited = pod_prompt_ids(session) | _job_prompt_ids(a["managed_session"])
        from .batches import _fingerprint
        for member in b["members"]:
            target = Path(member["folder"])
            if member.get("prepared_fingerprint") != _fingerprint(target):
                return False
            validate_prepared(target, info)
    except (ValueError, KeyError, TypeError, OSError, RuntimeError, httpx.HTTPError):
        return False  # Incompatible or unverifiable: no handoff; A retires normally.
    finally:
        session.close()
    saved = a["managed_session"]
    at = _now()
    saved["handoff_intent"] = {"to": b["id"], "folder": str(b_path), "at": at}
    atomic_json(a_path / "state.json", a)
    if crash_after == 1:
        raise RuntimeError("simulated crash after handoff step 1")
    choice = next(c for c in b["gpu_choices"] if c["gpu_id"] == saved["gpu_id"])
    adopted = {key: copy.deepcopy(saved[key]) for key in (
        "backend", "sid", "name", "pod_id", "gpu_id", "placement", "data_center_id", "quoted_hourly",
        "actual_hourly", "frozen_profile", "session_started_at", "quoted_gpu_hourly_usd",
        "quoted_cpu_hourly_usd", "create_attempted") if key in saved}
    adopted.update(create_intent=True, accepted_hourly=float(choice["max_hourly"]), jobs={}, phase="adopted",
                   adopted_at=at, adopted_from={"batch": a["id"], "folder": str(a_path), "sid": saved["sid"]},
                   inherited_prompt_ids=sorted(inherited))
    b["managed_session"] = adopted
    b["resource_choice"] = {"gpu_id": saved["gpu_id"], "hourly_usd": float(choice["max_hourly"]),
                            "cpu_hourly_usd": 0.0, "adopted": True}
    atomic_json(b_path / "state.json", b)
    if crash_after == 2:
        raise RuntimeError("simulated crash after handoff step 2")
    register(b_path)
    remove(a_path)
    if crash_after == 3:
        raise RuntimeError("simulated crash after handoff step 3")
    saved.pop("handoff_intent", None)
    saved["handed_off"] = {"to": b["id"], "folder": str(b_path), "at": at}
    saved["phase"] = "handed_off"
    atomic_json(a_path / "state.json", a)
    return True


def recover_handoff(source_folder: Path) -> str:
    """Finish or roll back an interrupted handoff. Returns "finished", "rolled_back" or "none"."""
    from .cost_safety import alert
    from .engine import run_lock
    from .global_registry import register, remove
    a_path = Path(source_folder).resolve()
    a = read_json(a_path / "state.json")
    intent = a.get("managed_session", {}).get("handoff_intent")
    if not intent:
        return "none"
    b_path = Path(intent["folder"]).resolve()
    first, second = sorted([a_path, b_path], key=str)
    with run_lock(first), run_lock(second):
        a = read_json(a_path / "state.json")
        saved = a["managed_session"]
        intent = saved.get("handoff_intent")
        if not intent:
            return "none"
        try:
            b = read_json(b_path / "state.json")
        except (OSError, ValueError):
            b = {}
        adopted = (b.get("managed_session") or {}).get("adopted_from") or {}
        if adopted.get("sid") == saved.get("sid") and adopted.get("batch") == a.get("id"):
            register(b_path)  # Roll forward: B owns the pod.
            remove(a_path)
            saved.pop("handoff_intent", None)
            saved["handed_off"] = {"to": intent["to"], "folder": str(b_path), "at": intent["at"]}
            saved["phase"] = "handed_off"
            atomic_json(a_path / "state.json", a)
            return "finished"
        saved.pop("handoff_intent", None)  # Roll back: A keeps the pod and its idle countdown.
        alert(a, Settings(**a["settings"]), "handoff_orphaned", "The target batch did not record the pod; "
              "the original batch keeps it and retires it normally.", key=f"handoff_orphaned:{intent['at']}")
        atomic_json(a_path / "state.json", a)
        register(a_path)
        return "rolled_back"

