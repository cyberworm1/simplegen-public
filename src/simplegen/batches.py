"""Durable, sequential managed RunPod batches.

Only dispatch can create paid compute. All mutations and dispatch hold the batch
folder's interprocess lock; member runs retain their own immutable history.
"""
from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import math
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import Event
from urllib.parse import urlencode

from simplegen_runpod.provider import GPU_STOCK_RANK as _STOCK_RANK
from simplegen_runpod.provider import available_gpu_centers as _available_centers

from .editing import read_json
from .engine import _run_locked, _verified_receipts, now, run_lock
from .settings import Settings, atomic_json, user_dir


class ReferenceUnavailable(ValueError):
    """A prepared reference cannot be read now (e.g. an unmounted linked folder)."""


def member_preflight(target):
    """Cloud-target preflight for a batch member; its snapshot wins over the linked workflow."""
    from .preflight import run_preflight
    from .refcache import complete
    child = read_json(Path(target) / "state.json")
    snapshot = Path(target) / "template.original.json"
    return run_preflight(Settings(**child["settings"]), target="cloud",
                         template_path=snapshot if snapshot.is_file() else None,
                         compiled=child.get("schema_version", 1) >= 5,
                         cached_references=snapshot.is_file() and complete(Path(target), child),
                         agent_needed=not child.get("workflow_acceptance"))


def _gate(result):
    from .preflight import FAIL, receipt, summary
    if result["status"] == FAIL:
        raise ValueError(f"Preflight failed: {summary(result)}")
    return receipt(result)


def _time(value):
    return datetime.fromisoformat(value) if value else None


def _release_members(state):
    """Every member of a completed batch is complete: release their cached references."""
    from .refcache import release_quietly
    if all(m.get("status") == "complete" for m in state["members"]):
        release_quietly([m["id"] for m in state["members"]])


def _fingerprint(target, *, include_references=True):
    """Bind GPU approval to the exact saved preparation, including Qty children."""
    state = read_json(target / "state.json")
    paths = [target / "workflow.json", target / "plan.json", target / "template.original.json"]
    for child in state.get("sequence", {}).get("children", []):
        sub = target / child["folder"]
        paths.extend((sub / "workflow.json", sub / "plan.json", sub / "template.original.json"))
    digest = hashlib.sha256()
    semantic = ("quantity", "duration", "steps", "megapixels", "aspect_ratio",
                "project_folder", "template_folder", "template_file", "runpod_config_file",
                "runpod_schema_file")
    digest.update(json.dumps({"concept": state["concept"],
                              "settings": {k: state["settings"].get(k) for k in semantic}},
                             sort_keys=True).encode())
    if include_references:
        for setting in ("runpod_config_file", "runpod_schema_file"):
            configured = state["settings"].get(setting)
            if configured:
                source = Path(configured).expanduser()
                digest.update(hashlib.sha256(source.read_bytes()).digest() if source.is_file() else b"missing")
    for path in paths:
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Prepared file is missing: {path.name}")
        digest.update(path.relative_to(target).as_posix().encode())
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    references = target / "used_references.txt"
    from .refcache import find, manifest_for
    manifest = manifest_for(target, state) if state.get("workflow_acceptance") else None
    if include_references and references.is_file():
        for name in references.read_text(encoding="utf-8").splitlines():
            entry = find(manifest, name, [])
            if entry is not None:
                # Phase 6: the captured content hash, not the original's mtime (no mount needed).
                digest.update(name.encode())
                digest.update(b"sha256:" + entry["sha256"].encode())
                continue
            source = Path(name)
            try:
                if not source.is_file():
                    raise ReferenceUnavailable("Prepared reference is missing")
                stat = source.stat()
            except OSError as exc:
                raise ReferenceUnavailable(f"Prepared reference is unreadable ({type(exc).__name__})") from None
            digest.update(name.encode())
            digest.update(str(stat.st_size).encode())
            digest.update(str(stat.st_mtime_ns).encode())
    return digest.hexdigest()


def _catalog(settings, *, provider=None):
    """Fresh secure catalog, including unavailable types; no allocation."""
    from simplegen_runpod.provider import RunPod

    from .credentials import load_credentials

    owned = provider is None
    if owned:
        credentials = load_credentials(settings)
        provider = RunPod(credentials.get("runpod_api_key") or credentials.get("RUNPOD_API_KEY"))
    from .global_session import load_global_profile
    profile = load_global_profile(Path(settings.runpod_config_file).expanduser())
    try:
        data = provider._request("GET", "/catalog/gpus?" + urlencode({
            "include": "AVAILABILITY", "product": "POD", "cloud": "SECURE", "count": 1,
            "minCudaVersion": profile.min_cuda_version,
        }), catalog=True)
    finally:
        if owned:
            provider.client.close()
    rows = []
    for gpu in data.get("gpus", []):
        if not gpu.get("secure") or not isinstance(gpu.get("id"), str):
            continue
        rate = gpu.get("price", {}).get("secure")
        memory = gpu.get("memory")
        if (type(rate) not in (int, float) or not math.isfinite(rate) or rate <= 0
                or type(memory) not in (int, float) or not math.isfinite(memory) or memory <= 0):
            continue
        centers = _available_centers(gpu.get("dataCenters"))
        center = centers[0] if centers else {}
        rows.append({"gpu_id": gpu["id"], "gpu_count": 1, "vram_gb": memory,
                     "hourly_usd": float(rate),
                     "availability": max((c["availability"] for c in centers),
                                         key=lambda v: _STOCK_RANK[v], default="UNAVAILABLE"),
                     "data_center": center.get("id"), "data_centers": centers,
                     "available_data_centers": [c["id"] for c in centers],
                     "available_location_count": len(centers),
                     "placement_scope": "global", "global_volume_id": profile.volume_id})
    if len({r["gpu_id"] for r in rows}) != len(rows):
        raise ValueError("RunPod catalog has duplicate GPU IDs")
    return rows


def select_gpu(approved, catalog, priority):
    """Choose from fresh quotes. Low waits for the cheapest approved type."""
    if priority not in ("high", "low") or len(approved) != 2:
        raise ValueError("Approve exactly two GPU types and High or Low priority")
    by_id = {r["gpu_id"]: r for r in catalog}
    quoted = []
    for item in approved:
        row = by_id.get(item["gpu_id"])
        if row:
            quoted.append((item, row))
    if not quoted:
        return None, "Approved GPU types are absent from the fresh catalog"
    if priority == "low" and len(quoted) != 2:
        return None, "One approved GPU is absent from the fresh catalog; cheapest type is unknown"
    quoted.sort(key=lambda pair: (pair[1]["hourly_usd"], approved.index(pair[0])))
    ordered = ([pair for pair in quoted if pair[1]["hourly_usd"] == quoted[0][1]["hourly_usd"]]
               if priority == "low" else quoted)
    for item, row in ordered:
        if row["hourly_usd"] > item["max_hourly"]:
            if priority == "low":
                return None, f"Cheapest approved GPU {row['gpu_id']} exceeds its hourly ceiling"
            continue
        centers = _available_centers(row.get("data_centers"))
        center = centers[0] if centers else None
        if center:
            return {**row, "data_center": center["id"],
                    "availability": max((c["availability"] for c in centers), key=lambda v: _STOCK_RANK[v]),
                    "data_centers": centers, "available_data_centers": [c["id"] for c in centers],
                    "available_location_count": len(centers), "placement_scope": "global",
                    "global_volume_id": item["volume_id"], "volume_id": item["volume_id"],
                    "hourly_usd": row["hourly_usd"]}, None
    return None, ("Cheapest approved GPU is unavailable or above its ceiling" if priority == "low"
                  else "Neither approved GPU is available within its hourly ceiling")


class BatchStore:
    def __init__(self, output_folder, *, register=True):
        self.root = Path(output_folder).expanduser().resolve() / ".simplegen-batches"
        self.root.mkdir(parents=True, exist_ok=True)
        if register:
            _register_root(self.root.parent)

    def _folder(self, batch_id):
        if not isinstance(batch_id, str) or not batch_id or any(c not in "0123456789abcdef" for c in batch_id):
            raise ValueError("Invalid batch ID")
        return self.root / batch_id

    def _read(self, folder):
        return read_json(folder / "state.json")

    def _save(self, folder, state):
        state["updated_at"] = now()
        atomic_json(folder / "state.json", state)

    def list(self):
        return [self._read(p) for p in sorted(self.root.iterdir()) if p.is_dir() and (p / "state.json").is_file()]

    def get(self, batch_id):
        return self._read(self._folder(batch_id))

    def catalog(self, batch_id):
        """Browse fresh secure GPU types without changing batch state or allocating compute."""
        state = self.get(batch_id)
        if not state.get("settings"):
            raise ValueError("Add a run before browsing batch GPU types")
        return _catalog(Settings(**state["settings"]))

    def create(self):
        batch_id = uuid.uuid4().hex
        folder = self._folder(batch_id)
        folder.mkdir()
        state = {"schema_version": 1, "id": batch_id, "run_id": batch_id,
                 "folder": str(folder), "created_at": now(), "status": "draft", "members": [],
                 "gpu_options": [], "gpu_choices": [], "priority": "high", "retry_minutes": 5,
                 "due_at": None, "next_check_at": None, "waiting_reason": None,
                 "managed_session": {}, "revisions": []}
        self._save(folder, state)
        return state

    def _mutate(self, batch_id, function, *, allow_owned=False):
        folder = self._folder(batch_id)
        with run_lock(folder):
            state = self._read(folder)
            if release_retired_session(folder, state):
                self._save(folder, state)
            if (state["status"] in ("running", "complete", "cancelled")
                    or state.get("managed_session", {}).get("create_intent")) and not allow_owned:
                raise ValueError("This batch is executing or owns a GPU; membership cannot change")
            if state.get("pending_add"):
                self._finish_pending_add(state)
                self._save(folder, state)
            function(state)
            self._save(folder, state)
            return copy.deepcopy(state)

    def _invalidate(self, state):
        state["gpu_options"] = []
        state["gpu_choices"] = []
        state["prepared_at"] = None
        state["status"] = "draft"
        state["next_check_at"] = None
        state["waiting_reason"] = None

    def add(self, batch_id, run_folder):
        target = Path(run_folder).expanduser().resolve()
        checked = _gate(member_preflight(target))  # Before any lock: schema/mount reads may be slow.
        def change(state):
            with run_lock(target):
                child = read_json(target / "state.json")
                settings = Settings(**child["settings"])
                if not settings.managed_runpod or not settings.global_desktop:
                    raise ValueError("Batch members require managed Global RunPod settings")
                if child.get("batch_parent") not in (None, state["id"]):
                    raise ValueError("Run already belongs to another batch")
                if (child.get("prompt_id") or child.get("submission_started")
                        or child.get("managed_session", {}).get("create_intent")):
                    raise ValueError("Run is already submitted or owns compute")
                for variation in child.get("sequence", {}).get("children", []):
                    sub = target / variation["folder"] / "state.json"
                    if sub.is_file():
                        progress = read_json(sub)
                        if progress.get("prompt_id") or progress.get("submission_started"):
                            raise ValueError("A variation is already submitted")
                if any(m["id"] == child["run_id"] for m in state["members"]):
                    raise ValueError("Run is already in this batch")
                if state["members"] and state["settings"]["runpod_config_file"] != child["settings"]["runpod_config_file"]:
                    raise ValueError("Members must share the same managed GPU profile")
                state["pending_add"] = {"id": child["run_id"], "folder": str(target)}
                self._save(Path(state["folder"]), state)
                child["batch_parent"] = state["id"]
                atomic_json(target / "state.json", child)
                state.setdefault("settings", child["settings"])
                state["members"].append({"id": child["run_id"], "folder": str(target),
                                         "concept": child["concept"], "status": "draft", "revision": 1,
                                         "preflight": checked})
                state.pop("pending_add", None)
                self._invalidate(state)
        return self._mutate(batch_id, change)

    def _finish_pending_add(self, state):
        pending = state.get("pending_add")
        if not pending:
            return
        target = Path(pending["folder"])
        with run_lock(target):
            child = read_json(target / "state.json")
            if child["run_id"] != pending["id"] or child.get("batch_parent") not in (None, state["id"]):
                raise ValueError("Pending batch member changed; inspect its saved run")
            child["batch_parent"] = state["id"]
            atomic_json(target / "state.json", child)
            if not any(m["id"] == child["run_id"] for m in state["members"]):
                state.setdefault("settings", child["settings"])
                state["members"].append({"id": child["run_id"], "folder": str(target),
                                         "concept": child["concept"], "status": "draft", "revision": 1})
            state.pop("pending_add", None)
            self._invalidate(state)

    def edit(self, batch_id, member_id, concept, settings=None):
        if not isinstance(concept, str) or not concept.strip():
            raise ValueError("Enter a concept")
        def change(state):
            member = next((m for m in state["members"] if m["id"] == member_id), None)
            if member is None:
                raise ValueError("Unknown batch member")
            old = read_json(Path(member["folder"]) / "state.json")
            chosen = settings or Settings(**old["settings"])
            if not chosen.managed_runpod or not chosen.global_desktop:
                raise ValueError("Batch revisions require managed Global RunPod settings")
            if chosen.runpod_config_file != state["settings"]["runpod_config_file"]:
                raise ValueError("Batch revisions must retain the shared managed GPU profile")
            from .engine import create_run
            target, child = create_run(chosen, concept)
            child["batch_parent"] = state["id"]
            atomic_json(target / "state.json", child)
            state["revisions"].append(copy.deepcopy(member))
            member.update(id=child["run_id"], folder=str(target), concept=concept,
                          status="draft", revision=member["revision"] + 1)
            self._invalidate(state)
        return self._mutate(batch_id, change)

    def remove(self, batch_id, member_id):
        def change(state):
            found = [m for m in state["members"] if m["id"] == member_id]
            if not found:
                raise ValueError("Unknown batch member")
            state["revisions"].extend(copy.deepcopy(found))
            state["members"] = [m for m in state["members"] if m["id"] != member_id]
            self._invalidate(state)
        return self._mutate(batch_id, change)

    def prepare(self, batch_id, on_event, cancel, detach=None):
        from .variations import run_sequence
        folder = self._folder(batch_id)
        with run_lock(folder):
            state = self._read(folder)
            self._finish_pending_add(state)
            self._save(folder, state)
            if state["status"] not in ("draft", "preparing", "preparation_failed", "prepared"):
                raise ValueError("Pause or review this batch before preparation")
            if not state["members"]:
                raise ValueError("Add a run to the batch first")
            try:
                state["status"] = "preparing"
                self._save(folder, state)
                for member in state["members"]:
                    if cancel.is_set() or (detach and detach.is_set()):
                        raise RuntimeError("Preparation interrupted")
                    target = Path(member["folder"])
                    # Gate each member's agent work on a fresh cloud preflight.
                    member["preflight"] = _gate(member_preflight(target))
                    self._save(folder, state)
                    with run_lock(target):
                        child = self._read(target)
                        settings = Settings(**child["settings"])
                        from .workflows import validate_graph
                        if settings.quantity > 1:
                            bundle = run_sequence(settings, child["concept"], target, child, on_event, cancel, detach,
                                                  None, prepare_only=True)
                            info = bundle["info"]
                            from .workflow_edits import validate_edited_workflow
                            for variation in child["sequence"]["children"]:
                                sub = target / variation["folder"]
                                sub_state = self._read(sub)
                                graph = validate_edited_workflow(
                                    read_json(sub / "template.original.json"), read_json(sub / "workflow.json"), info,
                                    megapixels=settings.megapixels, duration=settings.duration,
                                    steps=settings.steps, aspect_ratio=settings.aspect_ratio,
                                    run_id=sub_state["run_id"], check_model_availability=False)["graph"]
                                validate_graph(graph, info)
                        else:
                            bundle = _run_locked(settings, child["concept"], target, child, on_event, cancel, detach,
                                                 prepare_only=True)
                            graph = bundle["prepared"]["graph"]
                            validate_graph(graph, bundle["info"])
                    member["status"] = "prepared"
                    member["prepared_fingerprint"] = _fingerprint(target)
                    member["prepared_core_fingerprint"] = _fingerprint(target, include_references=False)
                    self._save(folder, state)
                state["prepared_at"] = now()
                state["status"] = "prepared"
                state["waiting_reason"] = None
                self._save(folder, state)
                return copy.deepcopy(state)
            except Exception as exc:
                state["status"] = "preparation_failed"
                state["waiting_reason"] = str(exc)
                self._save(folder, state)
                raise

    def configure(self, batch_id, *, gpu_choices, priority="high", retry_minutes=5, due_at=None,
                  max_hourly=None, catalog_options=None, max_total_usd=None):
        """``max_total_usd`` is an optional spend cap (additions over it are rejected; never cancels)."""
        if max_total_usd is not None and (type(max_total_usd) not in (int, float) or not math.isfinite(max_total_usd)
                                          or max_total_usd <= 0):
            raise ValueError("Spend cap must be a positive amount, or empty for no cap")

        def change(state):
            if state["status"] not in ("prepared", "armed", "waiting", "paused", "failed"):
                raise ValueError("Prepare all members before approving GPUs")
            if priority not in ("high", "low") or type(retry_minutes) is not int or not 1 <= retry_minutes <= 1440:
                raise ValueError("Invalid priority or retry interval")
            if not isinstance(gpu_choices, list) or len(gpu_choices) != 2 or len({x.get("gpu_id") for x in gpu_choices}) != 2:
                raise ValueError("Approve exactly two distinct GPU types")
            options = {o["gpu_id"]: o for o in (catalog_options if catalog_options is not None else state["gpu_options"])}
            approved = []
            cfg = json.loads(Path(state["settings"]["runpod_config_file"]).expanduser().read_text())
            if max_hourly is None:
                max_rate = state.get("max_hourly", cfg["gpu_max_hourly"])
            else:
                max_rate = max_hourly
            if type(max_rate) not in (int, float) or not math.isfinite(max_rate) or max_rate <= 0:
                raise ValueError("Batch hourly ceiling must be a positive finite amount")
            from .global_session import load_global_profile
            volume = load_global_profile(Path(state["settings"]["runpod_config_file"])).volume_id
            for item in gpu_choices:
                gpu_id = item.get("gpu_id")
                cap = item.get("max_hourly")
                if (gpu_id not in options or type(cap) not in (int, float)
                        or not math.isfinite(cap) or not 0 < cap <= max_rate):
                    raise ValueError("GPU type is missing from the catalog or hourly ceiling is invalid")
                approved.append({"gpu_id": gpu_id, "max_hourly": float(cap), "volume_id": volume})
            if due_at:
                due = _time(due_at)
                if due.tzinfo is None:
                    raise ValueError("Scheduled time needs a time zone")
            state["gpu_options"] = list(options.values())
            state.update(gpu_choices=approved, max_hourly=float(max_rate), priority=priority, retry_minutes=retry_minutes,
                         due_at=due_at, next_check_at=None, waiting_reason=None,
                         max_total_usd=float(max_total_usd) if max_total_usd is not None else None)
        return self._mutate(batch_id, change)

    def arm(self, batch_id):
        def change(state):
            if state["status"] not in ("prepared", "paused", "waiting", "armed") or len(state["gpu_choices"]) != 2:
                raise ValueError("Prepare and approve two GPUs before arming")
            state["status"] = "armed"
            state["next_check_at"] = state["due_at"] or now()
            state["waiting_reason"] = "Waiting for scheduled start" if state["due_at"] else None
        return self._mutate(batch_id, change)

    def run_now(self, batch_id):
        def change(state):
            if state["status"] not in ("prepared", "armed", "waiting", "paused") or len(state["gpu_choices"]) != 2:
                raise ValueError("Prepare and approve two GPUs before starting")
            state.update(status="armed", due_at=None, next_check_at=now(), waiting_reason=None)
        return self._mutate(batch_id, change)

    def pause(self, batch_id):
        def change(state):
            if state["status"] not in ("armed", "waiting"):
                raise ValueError("Only waiting batches may be paused")
            state["status"] = "paused"
        return self._mutate(batch_id, change)

    def resume(self, batch_id):
        def change(state):
            if state["status"] not in ("failed", "detached"):
                raise ValueError("Only failed or detached batches need explicit review and resume")
            from .cost_safety import clear_owner_failure
            clear_owner_failure(state)  # Stops any failure-shutdown countdown before new work.
            state.update(status="armed", next_check_at=now(), waiting_reason=None)
        return self._mutate(batch_id, change, allow_owned=True)

    def cancel(self, batch_id):
        def change(state):
            state["status"] = "cancelled"
            state["next_check_at"] = None
        return self._mutate(batch_id, change)

    def dispatch_due(self, on_event, cancel, detach=None):
        results = []
        self.recover_handoffs()
        for row in self.list():
            if row["status"] == "complete" and self.service_inbox(row["id"]) == "reopened":
                row = self.get(row["id"])  # Reopened during keep-warm: resume it now.
            if row["status"] not in ("armed", "waiting", "running"):
                continue
            if row["status"] != "running" and _time(row.get("next_check_at")) and _time(row["next_check_at"]) > datetime.now(UTC):
                continue
            try:
                results.append(self.run_batch(row["id"], on_event, cancel, detach))
            except (RuntimeError, ValueError, OSError) as exc:
                if "already running in another simpleGen window" not in str(exc):
                    results.append({"id": row["id"], "error": str(exc)})
        return results

    def run_batch(self, batch_id, on_event, cancel, detach=None, *, live_schema=None):
        from .global_session import session_for
        from .variations import run_sequence
        folder = self._folder(batch_id)
        self.recover_handoffs()
        self.service_inbox(batch_id)
        try:
            # Adopt a compatible warm pod before any GPU selection (own lock order).
            self.try_handoff(batch_id, live_schema=live_schema)
        except RuntimeError as exc:
            if "already running in another simpleGen window" not in str(exc):
                raise
        with run_lock(folder):
            state = self._read(folder)
            if state["status"] not in ("armed", "waiting", "running"):
                raise ValueError("Batch is not armed")
            if release_retired_session(folder, state):
                self._save(folder, state)  # A fresh GPU is selected below.
            if state["status"] != "running" and _time(state.get("next_check_at")) and _time(state["next_check_at"]) > datetime.now(UTC):
                return copy.deepcopy(state)
            from .additions import claim_child
            for member in state["members"]:
                if member.get("added_at"):
                    # Recovery: an accepted addition whose run was not yet claimed (crash after acceptance).
                    claim_child(Path(member["folder"]), batch_id, member["id"])
            if state.get("due_at") and _time(state["due_at"]) > datetime.now(UTC):
                return copy.deepcopy(state)
            settings = Settings(**state["settings"])
            def validate_member(member, *, references=True):
                target = Path(member["folder"])
                key = "prepared_fingerprint" if references else "prepared_core_fingerprint"
                if member.get(key) != _fingerprint(target, include_references=references):
                    raise ValueError("Prepared member changed; review and prepare a new revision")
            if not state.get("managed_session", {}).get("name"):
                try:
                    for member in state["members"]:
                        validate_member(member)
                except ReferenceUnavailable as exc:
                    # No pod exists yet: an unmounted linked folder is retried, not failed.
                    state.update(status="waiting", waiting_reason=f"Linked folder unavailable: {exc}",
                                 next_check_at=(datetime.now(UTC) + timedelta(minutes=state["retry_minutes"])).isoformat())
                    self._save(folder, state)
                    return copy.deepcopy(state)
                except (ValueError, OSError) as exc:
                    state.update(status="failed", waiting_reason=str(exc), next_check_at=None)
                    self._save(folder, state)
                    raise
            approved = state["gpu_choices"]
            settings.runpod_gpu_tiers = [item["gpu_id"] for item in approved]
            state["settings"] = settings.public_dict()
            if not state.get("managed_session", {}).get("name"):
                try:
                    choice, reason = select_gpu(approved, _catalog(settings), state["priority"])
                except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
                    state.update(status="waiting", waiting_reason=f"GPU availability check failed: {exc}",
                                 next_check_at=(datetime.now(UTC) + timedelta(minutes=state["retry_minutes"])).isoformat())
                    self._save(folder, state)
                    return copy.deepcopy(state)
                if choice is None:
                    state.update(status="waiting", waiting_reason=reason,
                                 next_check_at=(datetime.now(UTC) + timedelta(minutes=state["retry_minutes"])).isoformat())
                    self._save(folder, state)
                    return copy.deepcopy(state)
                approved_choice = next(a for a in approved if a["gpu_id"] == choice["gpu_id"])
                state["resource_choice"] = {**choice,
                                            "hourly_usd": approved_choice["max_hourly"],
                                            "cpu_hourly_usd": 0.0}
            state.update(status="running", waiting_reason=None, next_check_at=None)
            self._save(folder, state)
            def persist():
                self._save(folder, state)
            context = {"folder": folder, "state": state, "persist": persist, "batch": True}
            session = None
            try:
                if (state.get("managed_session", {}).get("downloads_verified")
                        and all(m.get("status") == "complete" for m in state["members"])):
                    receipts = []
                    for member in state["members"]:
                        target = Path(member["folder"])
                        child = self._read(target)
                        if child.get("sequence"):
                            for variation in child["sequence"]["children"]:
                                sub = target / variation["folder"]
                                receipts.extend(_verified_receipts(sub, self._read(sub)))
                        else:
                            receipts.extend(_verified_receipts(target, child))
                    if not receipts:
                        raise ValueError("Acknowledged batch has no verified local downloads")
                    state["status"] = "complete"
                    persist()
                    _release_members(state)
                    return copy.deepcopy(state)
                # Establish exactly one durable allocation intent before any child can submit.
                session = session_for(settings, folder, state, persist, on_event, cancel, detach or Event())
                try:
                    endpoint = session.connect(state.get("resource_choice"))
                except ValueError as exc:
                    if state.get("managed_session", {}).get("name"):
                        raise
                    state.update(status="waiting", waiting_reason=str(exc),
                                 next_check_at=(datetime.now(UTC) + timedelta(minutes=state["retry_minutes"])).isoformat())
                    persist()
                    return copy.deepcopy(state)
                def run_member(member, phase, position):
                    target = Path(member["folder"])
                    with run_lock(target):
                        child = self._read(target)
                        if child.get("batch_parent") != batch_id or child["run_id"] != member["id"]:
                            raise ValueError("Batch member identity changed")
                        submitted = bool(child.get("prompt_id") or child.get("submission_started"))
                        for variation in child.get("sequence", {}).get("children", []):
                            sub = target / variation["folder"]
                            if (sub / "state.json").is_file():
                                progress = self._read(sub)
                                submitted |= bool(progress.get("prompt_id") or progress.get("submission_started"))
                        validate_member(member, references=not submitted)
                        child_settings = Settings(**child["settings"])
                        if child_settings.quantity > 1:
                            context["validate_pending"] = lambda current=member: validate_member(current)
                            result = run_sequence(child_settings, child["concept"], target, child, on_event, cancel,
                                                  detach, None, batch=context, batch_phase=phase)
                        else:
                            result = _run_locked(child_settings, child["concept"], target, child, on_event, cancel,
                                                 detach, sequence=context, batch_phase=phase)
                    if phase == "submit":
                        member["status"] = "queued"
                        persist()
                        on_event({"stage": "Queued", "message":
                                  f"Queued {position}/{len(state['members'])} concepts · ComfyUI renders sequentially; downloads overlap queued work",
                                  "progress": None})
                    elif phase == "run":
                        member.update(status="complete", result=result)
                        persist()

                for phase in ("stage", "submit"):
                    if phase == "stage":
                        on_event({"stage": "Staging", "message": "Preparing all references before queueing renders", "progress": None})
                    else:
                        on_event({"stage": "Queueing", "message": "Adding individual renders to ComfyUI", "progress": None})
                    for position, member in enumerate(list(state["members"]), 1):
                        run_member(member, phase, position)
                index = 0
                while True:
                    # Drain points: before each member's run step, and once more before sealing.
                    self._drain(state, folder, persist, endpoint, run_member, on_event, live_schema)
                    if index >= len(state["members"]):
                        break
                    run_member(state["members"][index], "run", index + 1)
                    index += 1
                state["sealed_at"] = now()  # Later requests reopen during keep-warm, or are rejected.
                persist()
                receipts, completed_runs = [], []
                for member in state["members"]:
                    target = Path(member["folder"])
                    child = self._read(target)
                    if child.get("sequence"):
                        for variation in child["sequence"]["children"]:
                            child_folder = target / variation["folder"]
                            progress = self._read(child_folder)
                            receipts.extend(_verified_receipts(child_folder, progress))
                            completed_runs.append(progress)
                    else:
                        receipts.extend(_verified_receipts(target, child))
                        completed_runs.append(child)
                from .resources import record_run
                state["render_count"] = len(completed_runs)
                measured = [run.get("execution_seconds") for run in completed_runs]
                state["generation_seconds"] = sum(measured) if all(type(x) in (int, float) for x in measured) else None
                if state.get("reopen"):
                    # Cost history for a reopened batch records only the reopened portion's renders.
                    since = state["reopen"]["at"]
                    portion = []
                    for member in state["members"]:
                        if (member.get("added_at") or "") >= since:
                            target = Path(member["folder"])
                            child = self._read(target)
                            children = child.get("sequence", {}).get("children", [])
                            portion.extend(self._read(target / item["folder"]) for item in children)
                            if not children:
                                portion.append(child)
                    seconds = [run.get("execution_seconds") for run in portion]
                    state["portion_render_count"] = len(portion)
                    state["portion_generation_seconds"] = (sum(seconds) if all(type(x) in (int, float) for x in seconds)
                                                           else None)
                persist()
                session.complete(receipts)
                state["status"] = "complete"
                persist()
                _release_members(state)
                record_run(folder, state)
                return copy.deepcopy(state)
            except Exception as exc:
                failure = exc
                if not cancel.is_set() and not (detach and detach.is_set()) and session:
                    from .cost_safety import record_owner_failure
                    if record_owner_failure(state, settings, str(exc), "batch"):
                        persist()
                    # Already accepted prompts continue in ComfyUI after a later stage,
                    # submission, or download failure. Collect what can be verified
                    # without issuing any new prompt, then keep the original failure
                    # as the reason for explicit review.
                    recovery_errors = []
                    def recovery_event(event):
                        on_event({**event, "folder": str(folder)})
                    for member in state["members"]:
                        if cancel.is_set() or (detach and detach.is_set()):
                            break
                        target = Path(member["folder"])
                        try:
                            child = self._read(target)
                            jobs = [target / item["folder"] for item in child.get("sequence", {}).get("children", [])] or [target]
                        except Exception as recovery_error:  # noqa: BLE001 -- preserve first failure
                            recovery_errors.append(str(recovery_error))
                            continue
                        for job_folder in jobs:
                            if cancel.is_set() or (detach and detach.is_set()):
                                break
                            try:
                                with run_lock(job_folder):
                                    job = self._read(job_folder)
                                    if not (job.get("prompt_id") or job.get("submission_started")):
                                        continue
                                    result = _run_locked(Settings(**job["settings"]), child["concept"], job_folder,
                                                         job, recovery_event, cancel, detach, sequence=context)
                                    recovery_event({"stage": "Saved", "message": str(result["final_path"]),
                                                    "progress": None, "completed_result": result})
                            except Exception as recovery_error:  # noqa: BLE001 -- preserve first failure
                                recovery_errors.append(str(recovery_error))
                    if recovery_errors:
                        state["recovery_download_errors"] = recovery_errors
                        persist()
                    if (not cancel.is_set() and not (detach and detach.is_set())
                            and state.get("managed_session", {}).get("owner_failure")):
                        # Every submitted job was tried; failure shutdown may now count down.
                        state["managed_session"]["recovery_completed_at"] = now()
                        persist()
                if detach and detach.is_set():
                    state["status"] = "detached"
                elif cancel.is_set():
                    state["status"] = "cancelled"
                    if session and state.get("managed_session", {}).get("sid"):
                        try:
                            session.shutdown()
                        except Exception as shutdown_error:  # noqa: BLE001 -- never claim confirmed retirement
                            state["status"] = "failed"
                            failure = RuntimeError(f"Shutdown could not be confirmed: {shutdown_error}")
                else:
                    state["status"] = "failed"
                state["waiting_reason"] = str(failure)
                persist()
                if failure is not exc:
                    raise failure from exc
                raise
            finally:
                if session:
                    session.close()


    # Phase 4: additions inbox, sealing/reopen, warm-pod handoff

    def inbox(self, batch_id) -> Path:
        return self._folder(batch_id) / "inbox"

    def _settle(self, path: Path, outcome: str, request: dict, reason: str | None = None) -> None:
        """Move a request to inbox/accepted or inbox/rejected (never deletes the run)."""
        target = path.parent / outcome
        target.mkdir(exist_ok=True)
        record = {**request, "outcome": outcome, "settled_at": now()}
        if reason:
            record["reason"] = reason
        atomic_json(target / path.name, record)
        path.unlink(missing_ok=True)

    def _reject(self, path: Path, request: dict, reason: str) -> None:
        """Release the run (pending_batch cleared, reason recorded) and file the request as rejected."""
        folder = Path(request.get("folder", ""))
        try:
            with run_lock(folder):
                child = read_json(folder / "state.json")
                if child.get("pending_batch") == request.get("batch_id") and child.get("run_id") == request.get("run_id"):
                    child.pop("pending_batch", None)
                    child["batch_rejection"] = {"batch_id": request.get("batch_id"), "reason": reason, "at": now()}
                    atomic_json(folder / "state.json", child)
        except (OSError, ValueError, KeyError, RuntimeError):
            pass  # The request is still filed with its reason; the run is untouched.
        self._settle(path, "rejected", request, reason)

    def _drain(self, state, folder, persist, endpoint, run_member, on_event, live_schema=None):
        """Accept inbox requests into this running batch (the only membership writer)."""
        from .additions import accept_request
        inbox = folder / "inbox"
        if not inbox.is_dir():
            return
        info = None
        for path in sorted(inbox.glob("*.json")):
            try:
                request = read_json(path)
            except (OSError, ValueError) as error:
                self._settle(path, "rejected", {"file": path.name}, f"Unreadable request: {error}")
                continue
            existing = next((m for m in state["members"] if m["id"] == request.get("run_id")), None)
            if existing is not None:
                # Idempotent: a duplicate, or a crash after acceptance was recorded.
                from .additions import claim_child
                claim_child(Path(existing["folder"]), state["id"], existing["id"])
                self._settle(path, "accepted", request, "duplicate" if existing.get("added_at") is None else None)
                continue
            if info is None:
                info = live_schema(endpoint) if live_schema else _live_object_info(endpoint)
            try:
                member = accept_request(state, request, info)
            except ValueError as error:
                self._reject(path, request, str(error))
                on_event({"stage": "Addition rejected", "message": str(error), "progress": None})
                continue
            state["members"].append(member)
            persist()  # Durable acceptance before the child claim and any submission.
            from .additions import claim_child
            claim_child(Path(member["folder"]), state["id"], member["id"])
            position = len(state["members"])
            run_member(member, "stage", position)
            run_member(member, "submit", position)
            self._settle(path, "accepted", request)
            on_event({"stage": "Addition accepted", "message": f"Added {member['concept'][:60]} to the running batch",
                      "progress": None})

    def service_inbox(self, batch_id):
        """Requests for a closed batch: reopen during keep-warm, else reject ("start a new batch")."""
        folder = self._folder(batch_id)
        inbox = folder / "inbox"
        if not inbox.is_dir() or not any(inbox.glob("*.json")):
            return None
        try:
            with run_lock(folder):
                state = self._read(folder)
                if state["status"] in ("armed", "waiting", "running", "draft", "preparing", "prepared", "paused",
                                       "failed", "detached"):
                    return None  # The owner drains it (or it is not running yet).
                reason = reopen_refusal(state)
                if reason is None:
                    self._reopen(folder, state, "inbox request")
                    return "reopened"
                for path in sorted(inbox.glob("*.json")):
                    try:
                        request = read_json(path)
                    except (OSError, ValueError):
                        request = {"file": path.name}
                    self._reject(path, request, reason + " Start a new batch for this concept.")
                return "rejected"
        except RuntimeError as exc:
            if "already running in another simpleGen window" in str(exc):
                return None
            raise

    def _reopen(self, folder, state, reason):
        saved = state["managed_session"]
        try:
            from .resources import record_run
            record_run(folder, state, until=now())  # Close the portion billed so far.
        except (OSError, ValueError) as error:
            saved.setdefault("alerts", []).append(f"cost_history_write_failed: {type(error).__name__}")
        state["reopen"] = {"at": now(), "reason": reason}
        state.setdefault("reopen_history", []).append(state["reopen"])
        for key in ("downloads_verified", "acknowledged_manifest", "idle_since", "idle_remaining_seconds"):
            saved.pop(key, None)
        state.pop("sealed_at", None)
        state.update(status="running", next_check_at=now(), waiting_reason=None)
        self._save(folder, state)  # Jobs and receipts are kept; completion re-verifies everything.

    def reopen(self, batch_id, reason="add to warm pod"):
        """Explicit "Add to warm pod": reopen a completed batch whose pod is in its keep-warm window."""
        folder = self._folder(batch_id)
        with run_lock(folder):
            state = self._read(folder)
            refusal = reopen_refusal(state)
            if refusal:
                raise ValueError(refusal + " Start a new batch instead.")
            self._reopen(folder, state, reason)
            return copy.deepcopy(state)

    def try_handoff(self, batch_id, *, live_schema=None, stores=None):
        """Let a due batch adopt another batch's warm pod instead of creating one.

        Returns the source batch id, or None when no compatible warm pod exists.
        """
        from .additions import handoff_candidates, perform_handoff
        target = self._folder(batch_id)
        state = self._read(target)
        if (state.get("managed_session", {}).get("name") or state["status"] not in ("armed", "waiting")
                or len(state.get("gpu_choices", [])) != 2):
            return None
        if state.get("due_at") and _time(state["due_at"]) > datetime.now(UTC):
            return None
        for source in handoff_candidates(stores or registered_stores() or [self], exclude=target):
            if perform_handoff(Path(source["folder"]), target, live_schema=live_schema):
                return source["id"]
        return None

    def recover_handoffs(self):
        from .additions import recover_handoff
        for row in self.list():
            if row.get("managed_session", {}).get("handoff_intent"):
                recover_handoff(Path(row["folder"]))


_RELEASABLE_STATUSES = ("prepared", "armed", "waiting", "paused", "failed", "detached")


def retired_without_work(saved: dict) -> bool:
    """A session whose pod is confirmed gone and that never held a job: it owns nothing.

    Reads only the session record (safe for the UI thread). Handoffs are excluded:
    their ownership moves between batches through their own recovery path.
    """
    return bool(saved.get("name") and saved.get("retired") and saved.get("billing_confirmed_stopped")
                and not saved.get("jobs") and not saved.get("unresolved_jobs")
                and not saved.get("handoff_intent") and not saved.get("handed_off"))


def owns_gpu(state: dict) -> bool:
    saved = state.get("managed_session") or {}
    return bool(saved.get("create_intent") or saved.get("name")) and not retired_without_work(saved)


def _member_submitted(member: dict) -> bool:
    """Whether any render of this member reached submission; unreadable state counts as submitted."""
    target = Path(member["folder"])
    try:
        child = read_json(target / "state.json")
        states = [child]
        for variation in child.get("sequence", {}).get("children", []):
            path = target / variation["folder"] / "state.json"
            if path.is_file():
                states.append(read_json(path))
    except (OSError, ValueError, KeyError, TypeError):
        return True
    return any(s.get("prompt_id") or s.get("submission_started") for s in states)


def release_retired_session(folder: Path, state: dict) -> bool:
    """Archive a retired, never-used GPU session so the batch can select compute again.

    A pod that never became ready can be retired (by the startup policy or by
    the user terminating it in RunPod) before any job was bound. Its record
    still carries ``create_intent`` and a session name, which blocked every
    batch action and made Resume reconnect to the vanished pod forever. Only a
    session that is retired with billing confirmed stopped, never held a job,
    and whose members never reached submission is released; anything else
    keeps the existing review path. The record moves to ``retired_sessions``
    (append-only) after its cost row is written. The caller saves the state.
    """
    saved = state.get("managed_session") or {}
    if (state.get("status") not in _RELEASABLE_STATUSES or not retired_without_work(saved)
            or any(_member_submitted(m) for m in state.get("members", []))):
        return False
    try:
        from .resources import record_run
        record_run(folder, state)  # Idempotent: enriches this session's existing row.
    except (OSError, ValueError) as error:
        saved.setdefault("alerts", []).append(f"cost_history_write_failed: {type(error).__name__}")
    from .global_registry import remove
    remove(Path(folder))
    state.setdefault("retired_sessions", []).append({**copy.deepcopy(saved), "released_at": now()})
    state["managed_session"] = {}
    return True


def reopen_refusal(state: dict) -> str | None:
    """Why a completed batch cannot take more work on its warm pod, or None."""
    saved = state.get("managed_session") or {}
    if saved.get("retired") or saved.get("billing_confirmed_stopped"):
        return "This batch's GPU is retired."
    if saved.get("delete_intent"):
        return "This batch's GPU is shutting down."
    if saved.get("handed_off") or saved.get("handoff_intent"):
        return "This batch's GPU was handed to another batch."
    if state.get("status") != "complete" or not saved.get("downloads_verified") or not saved.get("pod_id"):
        return "This batch has no warm GPU to add to."
    return None


def _live_object_info(endpoint):
    """The live pod's /object_info (not the cache)."""
    from . import engine
    comfy = engine.Comfy(endpoint)
    try:
        return comfy.object_info()
    finally:
        close = getattr(comfy, "close", None)
        if close:
            close()


def _register_root(output_folder):
    index = user_dir() / "batches" / "roots.json"
    index.parent.mkdir(parents=True, exist_ok=True)
    with index.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            rows = json.loads(index.read_text()) if index.is_file() else []
            if not isinstance(rows, list) or any(not isinstance(row, str) for row in rows):
                raise ValueError("Batch roots index is invalid")
            path = str(Path(output_folder).expanduser().resolve())
            if path not in rows:
                rows.append(path)
                atomic_json(index, rows)
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def registered_stores():
    """Find scheduled batches even after the user changes the current output folder."""
    index = user_dir() / "batches" / "roots.json"
    if not index.is_file():
        return []
    rows = json.loads(index.read_text())
    if not isinstance(rows, list) or any(not isinstance(row, str) for row in rows):
        raise ValueError("Batch roots index is invalid")
    return [BatchStore(row, register=False) for row in rows if Path(row).is_dir()]
