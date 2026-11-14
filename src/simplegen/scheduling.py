"""Durable local-only scheduling. No agent or GPU work occurs while pending."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import time
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from .settings import Settings, atomic_json, user_dir

PENDING = {"scheduled", "waiting", "overdue"}
VISIBLE = PENDING | {"running", "attention", "paused", "repaired"}
RETRY_SECONDS = 30 * 60


def fingerprint(record: dict) -> str:
    return hashlib.sha256(json.dumps(record, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def pristine(record: dict, state: dict) -> bool:
    if record.get("prepared"):
        # A prepared project (e.g. a repair revision) carries accepted work and
        # must still be untouched since scheduling.
        return (not record.get("started") and state.get("status") == "scheduled"
                and not state.get("schedule", {}).get("released")
                and not state.get("submission_started") and not state.get("prompt_id")
                and bool(state.get("workflow_acceptance")) and not state.get("sequence")
                and state.get("run_id") == record.get("id"))
    return (not record.get("started") and state.get("status") == "scheduled"
            and not state.get("schedule", {}).get("released")
            and not state.get("submission_started") and not state.get("prompt_id")
            and not state.get("workflow_edit_attempts") and not state.get("workflow_acceptance")
            and not state.get("sequence"))


def durable_bytes(path: Path, data: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    parent = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(parent)
    finally:
        os.close(parent)


def atomic_bytes(path: Path, data: bytes) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    parent = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(parent)
    finally:
        os.close(parent)


def system_zone():
    name = os.environ.get("TZ", "").removeprefix(":")
    if name and not name.startswith("/"):
        return ZoneInfo(name)
    with Path(name or "/etc/localtime").open("rb") as handle:
        return ZoneInfo.from_file(handle, key=name or "system local time")


def resolve_local(value: datetime, *, zone=None) -> datetime:
    """Reject DST gaps/folds instead of silently choosing a different wall time."""
    zone = zone or system_zone()
    candidates = set()
    for fold in (0, 1):
        instant = value.replace(tzinfo=zone, fold=fold).astimezone(UTC)
        if instant.astimezone(zone).replace(tzinfo=None) == value:
            candidates.add(instant)
    if len(candidates) != 1:
        raise ValueError("That local time is missing or ambiguous due to daylight saving. Choose another time.")
    return candidates.pop()


def display_time(instant: str) -> str:
    return datetime.fromisoformat(instant).astimezone(system_zone()).strftime("%m/%d/%y %H:%M")


def local_queue_empty(settings: Settings) -> bool:
    from .comfy import Comfy
    if settings.cloud or settings.managed_runpod:
        raise ValueError("Scheduling is available for Local only.")
    comfy = Comfy(settings)
    try:
        queue = comfy.request("GET", "/queue", timeout=5).json()
        if not all(isinstance(queue.get(k), list) for k in ("queue_running", "queue_pending")):
            raise ValueError("Cannot verify that the local ComfyUI queue is empty.")
        return not queue["queue_running"] and not queue["queue_pending"]
    finally:
        comfy.close()


class Scheduler:
    def __init__(self, root: Path | None = None, *, clock=time.time):
        self.root = root or user_dir() / "schedules"
        self.root.mkdir(parents=True, exist_ok=True)
        self.clock = clock
        self.last_tick = clock()
        self.suspended: set[str] = set()
        self._lock = (self.root / ".owner.lock").open("a")
        try:
            fcntl.flock(self._lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.close()
        if self.owner:
            self.recover()

    @contextmanager
    def suspend(self, identity: str):
        self.suspended.add(identity)
        try:
            yield
        finally:
            self.suspended.discard(identity)

    @property
    def owner(self):
        return self._lock is not None

    def close(self):
        if self._lock is not None:
            self._lock.close()
            self._lock = None

    def _require_owner(self):
        if not self.owner:
            raise ValueError("Schedules are managed by another open simpleGen window.")

    def records(self):
        result = []
        for path in self.root.glob("*.json"):
            value = json.loads(path.read_text(encoding="utf-8"))
            if value.get("schema_version") != 1 or value.get("id") != path.stem:
                raise ValueError(f"Unsupported schedule record: {path.name}")
            result.append(value)
        return sorted(result, key=lambda item: (item["due_at"], item["id"]))

    def get(self, identity):
        return next(r for r in self.records() if r["id"] == identity)

    def _save(self, record):
        self._require_owner()
        atomic_json(self.root / f"{record['id']}.json", record)

    def add(self, settings: Settings, concept: str, due: datetime, fork_seed: dict | None = None, *,
            preflight: dict, override_unverified: bool = False):
        """Publish a schedule only after a passing (or explicitly overridden) preflight."""
        from .preflight import FAIL, PASS, UNVERIFIED, receipt, summary
        self._require_owner()
        if settings.cloud or settings.managed_runpod:
            raise ValueError("Scheduling is available for Local only.")
        if due.tzinfo is None or due.timestamp() <= self.clock():
            raise ValueError("Choose a future date and time.")
        status = preflight.get("status") if isinstance(preflight, dict) else None
        if status == FAIL:
            raise ValueError(f"Preflight failed: {summary(preflight)}")
        if status == UNVERIFIED and not override_unverified:
            raise ValueError(f"Preflight could not verify this request: {summary(preflight)}")
        if status not in (PASS, UNVERIFIED):
            raise ValueError("Scheduling requires a preflight result.")
        override = status == UNVERIFIED
        from .editing import digest, read_json
        from .engine import _template, create_run, fork_uses_snapshot
        settings.validate(require_folders=False)
        seeded = fork_uses_snapshot(settings, fork_seed)
        source = (Path(fork_seed["folder"]) / "template.original.json") if seeded else _template(settings)
        template = read_json(source)
        folder, state = create_run(settings, concept, fork_seed)
        # A failed save still leaves a project that cannot accidentally run early.
        state["schedule"] = {"due_at": due.astimezone(UTC).isoformat(), "released": False}
        state["status"] = "scheduled"
        state["preflight"] = receipt(preflight, override_unverified=override)
        atomic_json(folder / "state.json", state)
        if not state.get("template_sha256"):
            atomic_json(folder / "template.original.json", template)
        state.update(template_path=state.get("template_path") or str(source),
                     template_sha256=digest(folder / "template.original.json"))
        atomic_json(folder / "state.json", state)
        record = {"schema_version": 1, "id": state["run_id"], "folder": str(folder),
                  "concept": concept, "due_at": state["schedule"]["due_at"],
                  "selected_local": due.astimezone(system_zone()).isoformat(),
                  "status": "scheduled", "started": False,
                  "next_retry_at": state["schedule"]["due_at"], "retry_count": 0,
                  "preflight": {"status": status, "checked_at": preflight.get("checked_at"),
                                "override_unverified": override,
                                "reason": "" if status == PASS else summary(preflight)}}
        (folder / "report.txt").write_text(
            f"SCHEDULED (LOCAL)\n{display_time(record['due_at'])}\nKeep simpleGen open and this computer awake.\n",
            encoding="utf-8")
        self._save(record)  # Publish only after the project and snapshot are durable.
        return record

    def add_prepared(self, folder: Path, due: datetime, *, preflight: dict, override_unverified: bool = False,
                     source_record: str | None = None):
        """Schedule an already-prepared project (accepted work, never submitted).

        No new run is created and no agent runs at dispatch; the same
        preflight re-check as ``add`` applies when it becomes due.
        """
        from .editing import digest
        from .engine import run_lock
        from .preflight import FAIL, PASS, UNVERIFIED, receipt, summary
        from .repair import refusal
        self._require_owner()
        if due.tzinfo is None or due.timestamp() <= self.clock():
            raise ValueError("Choose a future date and time.")
        status = preflight.get("status") if isinstance(preflight, dict) else None
        if status == FAIL:
            raise ValueError(f"Preflight failed: {summary(preflight)}")
        if status == UNVERIFIED and not override_unverified:
            raise ValueError(f"Preflight could not verify this request: {summary(preflight)}")
        if status not in (PASS, UNVERIFIED):
            raise ValueError("Scheduling requires a preflight result.")
        override = status == UNVERIFIED
        folder = Path(folder).expanduser().resolve()
        with run_lock(folder):
            state = json.loads((folder / "state.json").read_text(encoding="utf-8"))
            settings = Settings(**state["settings"])
            if settings.cloud or settings.managed_runpod:
                raise ValueError("Scheduling is available for Local only.")
            reason = refusal(folder, state)
            if reason:
                raise ValueError(reason)
            prior = [r for r in self.records() if r["id"] == state["run_id"]]
            if state.get("schedule") or any(r["status"] != "removed" for r in prior):
                raise ValueError("This project is already scheduled.")
            if preflight.get("template_sha256") not in (None, digest(folder / "template.original.json")):
                raise ValueError("Preflight was taken for a different workflow snapshot. Check again.")
            due_at = due.astimezone(UTC).isoformat()
            state["schedule"] = {"due_at": due_at, "released": False}
            state["status"] = "scheduled"
            state["preflight"] = receipt(preflight, override_unverified=override)
            atomic_json(folder / "state.json", state)
            record = {"schema_version": 1, "id": state["run_id"], "folder": str(folder),
                      "concept": state["concept"], "due_at": due_at,
                      "selected_local": due.astimezone(system_zone()).isoformat(),
                      "status": "scheduled", "started": False, "prepared": True,
                      "next_retry_at": due_at, "retry_count": 0,
                      "preflight": {"status": status, "checked_at": preflight.get("checked_at"),
                                    "override_unverified": override,
                                    "reason": "" if status == PASS else summary(preflight)}}
            (folder / "report.txt").write_text(
                f"SCHEDULED (LOCAL)\n{display_time(due_at)}\nKeep simpleGen open and this computer awake.\n",
                encoding="utf-8")
            for removed in prior:  # Keep the removed record before its id is reused.
                history = self.root / "history"
                history.mkdir(exist_ok=True)
                atomic_json(history / f"{removed['id']}-{uuid.uuid4().hex[:8]}.json", removed)
            self._save(record)
        if source_record:
            source = self.get(source_record)
            if source.get("status") == "repaired" and source.get("repaired_to", {}).get("run_id") == record["id"]:
                source["repaired_to"]["record_id"] = record["id"]
                self._save(source)
        return record

    def mark_repaired(self, identity: str, folder: Path, run_id: str):
        """Link a needs-attention record to its repair revision; it can no longer dispatch."""
        self._require_owner()
        record = self.get(identity)
        if record["status"] == "repaired" or record.get("repaired_to"):
            raise ValueError("This scheduled request was already repaired.")
        if record["status"] != "attention" or record.get("edit_pending"):
            raise ValueError("Only a scheduled request that needs attention can be repaired.")
        revision = json.loads((Path(folder) / "state.json").read_text(encoding="utf-8"))
        origin = revision.get("repaired_from", {})
        if (revision.get("run_id") != run_id
                or Path(origin.get("folder", "")).resolve() != Path(record["folder"]).resolve()):
            raise ValueError("That revision was not repaired from this scheduled request.")
        record.update(status="repaired", note=f"Repaired → {Path(folder).name}",
                      repaired_to={"run_id": run_id, "folder": str(folder),
                                   "at": datetime.now(UTC).isoformat()})
        self._save(record)
        return record

    def recover(self):
        for record in self.records():
            if record.get("edit_pending"):
                try:
                    self._finish_edit(record)
                except (OSError, ValueError, KeyError, TypeError):
                    record["status"] = "attention"
                    record["note"] = "Interrupted edit needs review; generation was not dispatched"
                    self._save(record)
                continue
            if record["status"] in {"running", "attention"}:
                try:
                    state = json.loads((Path(record["folder"]) / "state.json").read_text())
                    status = "complete" if state.get("status") == "complete" and state.get("run_id") == record["id"] else "attention"
                except (OSError, ValueError):
                    status = "attention"
                if status != record["status"]:
                    record["status"] = status
                    self._save(record)
            elif record["status"] in PENDING and datetime.fromisoformat(record["due_at"]).timestamp() <= self.clock():
                record["status"] = "overdue"
                self._save(record)

    def tick(self):
        if not self.owner:
            return []
        current = self.clock()
        gap = current - self.last_tick > 60
        self.last_tick = current
        due = []
        for record in self.records():
            if (record["status"] not in PENDING or record.get("edit_pending")
                    or record["id"] in self.suspended):
                continue
            if datetime.fromisoformat(record["due_at"]).timestamp() > current:
                continue
            if not record.get("started"):
                try:
                    state = json.loads((Path(record["folder"]) / "state.json").read_text())
                    safe = pristine(record, state)
                except (OSError, ValueError, KeyError, TypeError):
                    safe = False
                if not safe:
                    record["status"] = "attention"
                    record["note"] = "Saved request changed or failed; review before Resume"
                    self._save(record)
                    continue
            status = "overdue" if gap else "waiting"
            if record["status"] != status:
                record["status"] = status
                self._save(record)
            if ((not record.get("started") or record.get("resume_requested"))
                    and datetime.fromisoformat(record.get("next_retry_at", record["due_at"])).timestamp() <= current):
                due.append(record)
        return due

    def note(self, identity, message):
        record = self.get(identity)
        if record.get("note") != message:
            record["note"] = message
            self._save(record)

    def defer(self, identity, message):
        """Record one failed availability attempt and its next eligible instant."""
        record = self.get(identity)
        if record["status"] not in PENDING or record.get("started") or record.get("edit_pending"):
            return
        if datetime.fromisoformat(record.get("next_retry_at", record["due_at"])).timestamp() > self.clock():
            return
        record["status"] = "waiting"
        record["note"] = message
        record["next_retry_at"] = datetime.fromtimestamp(self.clock() + RETRY_SECONDS, UTC).isoformat()
        record["retry_count"] = record.get("retry_count", 0) + 1
        self._save(record)

    def change(self, identity, action, due=None):
        self._require_owner()
        self.tick()
        record = self.get(identity)
        if record.get("edit_pending"):
            raise ValueError("This scheduled edit is being recovered; reopen the request.")
        if record["status"] not in VISIBLE or record["status"] == "running":
            raise ValueError("This schedule is already running or no longer pending.")
        if record["status"] == "repaired" and action != "remove":
            raise ValueError("This request was repaired; run or schedule its repaired revision instead.")
        from .engine import run_lock
        folder = Path(record["folder"])
        with run_lock(folder):
            record = self.get(identity)
            if record.get("edit_pending"):
                raise ValueError("This scheduled edit is being recovered; reopen the request.")
            state = json.loads((folder / "state.json").read_text())
            if action == "remove":
                record["status"] = "removed"
                if (not record["started"] and state.get("schedule")
                        and not state["schedule"].get("released") and state.get("status") == "scheduled"
                        and not state.get("prompt_id") and not state.get("submission_started")):
                    # Release the never-started project for Resume or a new schedule;
                    # its schedule data moves to history. State first: a crash
                    # before the record save leaves a record that tick() flags.
                    state.setdefault("schedule_history", []).append(
                        {**state.pop("schedule"), "record_id": record["id"], "prior_status": state["status"],
                         "removed_at": datetime.fromtimestamp(self.clock(), UTC).isoformat()})
                    state["status"] = "validated" if state.get("workflow_acceptance") else "created"
                    atomic_json(folder / "state.json", state)
            elif action in ("now", "reschedule"):
                if action == "reschedule" and record["started"]:
                    raise ValueError("This run has already started. Use Resume to reconnect.")
                instant = datetime.fromtimestamp(self.clock(), UTC) if action == "now" else due
                if instant is None or instant.tzinfo is None or (action == "reschedule" and instant.timestamp() <= self.clock()):
                    raise ValueError("Choose a future date and time.")
                record.update(due_at=instant.astimezone(UTC).isoformat(), status="scheduled", note="")
                record["next_retry_at"] = record["due_at"]
                record["resume_requested"] = bool(record["started"] and action == "now")
                state["schedule"].update(due_at=record["due_at"], released=False)
                self._save(record)  # The schedule record owns the selected start instant.
                atomic_json(folder / "state.json", state)
            else:
                raise ValueError("Unknown schedule action.")
            self._save(record)

    def claim(self, identity, preflight: dict | None = None):
        """Reserve a due request. A fresh dispatch needs a preflight taken now:
        ``fail`` moves it to attention without any agent work; ``unverified``
        follows the 30-minute blocked-retry path."""
        self._require_owner()
        record = self.get(identity)
        if (record["status"] not in PENDING or record.get("edit_pending") or identity in self.suspended
                or datetime.fromisoformat(record["due_at"]).timestamp() > self.clock()
                or datetime.fromisoformat(record.get("next_retry_at", record["due_at"])).timestamp() > self.clock()):
            return None
        from .engine import run_lock
        folder = Path(record["folder"])
        with run_lock(folder):
            record = self.get(identity)
            if (record["status"] not in PENDING or record.get("edit_pending") or identity in self.suspended
                    or datetime.fromisoformat(record["due_at"]).timestamp() > self.clock()
                    or datetime.fromisoformat(record.get("next_retry_at", record["due_at"])).timestamp() > self.clock()):
                return None
            state = json.loads((folder / "state.json").read_text())
            if record.get("started") and record.get("resume_requested") and record["status"] in PENDING:
                state["schedule"].update(due_at=record["due_at"], released=True)
                record.update(status="running", note="", resume_requested=False)
                self._save(record)
                atomic_json(folder / "state.json", state)
                return folder
            if not pristine(record, state):
                record["status"] = "attention"
                record["note"] = "Saved request changed or started; review before Resume"
                self._save(record)
                return None
            settings = Settings(**state["settings"])
            if settings.cloud or settings.managed_runpod:
                raise ValueError("Scheduling is available for Local only.")
            from .preflight import FAIL, PASS, receipt, summary
            status = preflight.get("status") if isinstance(preflight, dict) else None
            if status != PASS:
                reason = summary(preflight) if status else "Preflight did not run"
                record["preflight"] = {"status": status or "unverified", "checked_at": (preflight or {}).get("checked_at"),
                                       "override_unverified": False, "reason": reason, "at": "dispatch"}
                if status == FAIL:
                    record.update(status="attention", note=f"Preflight failed: {reason}")
                    self._save(record)
                    return None
                record.update(status="waiting", note=f"Preflight unverified: {reason}",
                              next_retry_at=datetime.fromtimestamp(self.clock() + RETRY_SECONDS, UTC).isoformat(),
                              retry_count=record.get("retry_count", 0) + 1)
                self._save(record)
                return None
            record["preflight"] = {"status": PASS, "checked_at": preflight.get("checked_at"),
                                   "override_unverified": False, "reason": "", "at": "dispatch"}
            state["preflight_dispatch"] = receipt(preflight)
            record.update(status="running", started=True, note="")
            self._save(record)  # Dispatch reservation precedes release and the worker.
            state["schedule"].update(due_at=record["due_at"], released=True)
            atomic_json(folder / "state.json", state)
        return folder

    def edit(self, identity, expected: str, settings: Settings, concept: str, due: datetime):
        """Commit an append-only pending revision; never edit a started project."""
        self._require_owner()
        from .editing import digest, read_json
        from .engine import _template, run_lock
        record = self.get(identity)
        folder = Path(record["folder"])
        with run_lock(folder):
            record = self.get(identity)
            state_path, template_path = folder / "state.json", folder / "template.original.json"
            old_state = state_path.read_bytes()
            old_template = template_path.read_bytes()
            state = json.loads(old_state)
            if fingerprint(record) != expected or record["status"] not in PENDING or not pristine(record, state):
                raise ValueError("This scheduled request changed or started while editing. Reopen it.")
            if not concept.strip():
                raise ValueError("Enter a concept first.")
            if settings.cloud or settings.managed_runpod:
                raise ValueError("Scheduling is available for Local only.")
            settings.validate()
            if due.tzinfo is None:
                raise ValueError("Choose a valid local date and time.")
            due_at = due.astimezone(UTC).isoformat()
            if due_at != record["due_at"] and due.timestamp() <= self.clock():
                raise ValueError("Choose a future date and time.")
            same_workflow = (settings.template_file == state["settings"].get("template_file")
                             and settings.template_folder == state["settings"].get("template_folder"))
            source = Path(state["template_path"]) if same_workflow else _template(settings)
            new_template = None if same_workflow else read_json(source)
            revision = uuid.uuid4().hex
            directory = folder / "schedule-revisions" / revision
            directory.mkdir(parents=True, exist_ok=False)
            durable_bytes(directory / "prior-state.json", old_state)
            durable_bytes(directory / "prior-template.json", old_template)
            atomic_json(directory / "prior-record.json", record)
            new_state = dict(state)
            new_state.update(concept=concept, settings=settings.public_dict(),
                             template_path=str(source))
            new_state["schedule"] = {**state["schedule"], "due_at": due_at, "released": False}
            atomic_json(directory / "state.json", new_state)
            if same_workflow:
                durable_bytes(directory / "template.json", old_template)
            else:
                atomic_json(directory / "template.json", new_template)
            new_state["template_sha256"] = digest(directory / "template.json")
            atomic_json(directory / "state.json", new_state)
            updated = dict(record)
            updated.update(concept=concept, due_at=due_at,
                           selected_local=due.astimezone(system_zone()).isoformat(),
                           revision=revision, note="")
            if due_at != record["due_at"]:
                updated["next_retry_at"] = due_at
                updated["status"] = "scheduled"
            atomic_json(directory / "record.json", updated)
            manifest = {name: digest(directory / name) for name in
                        ("state.json", "template.json", "record.json", "prior-state.json",
                         "prior-template.json", "prior-record.json")}
            atomic_json(directory / "manifest.json", manifest)
            record["edit_pending"] = revision
            self._save(record)
            self._finish_edit(record, locked=True)
        return self.get(identity)

    def _finish_edit(self, record, *, locked=False):
        from .editing import digest, read_json
        from .engine import run_lock
        folder = Path(record["folder"])
        revision = record["edit_pending"]
        directory = folder / "schedule-revisions" / revision
        def finish():
            manifest = read_json(directory / "manifest.json")
            required = {"state.json", "template.json", "record.json", "prior-state.json",
                        "prior-template.json", "prior-record.json"}
            if set(manifest) != required or not all(isinstance(v, str) and len(v) == 64
                                                     for v in manifest.values()):
                raise ValueError("Pending schedule edit manifest is incomplete")
            for name, expected in manifest.items():
                if digest(directory / name) != expected:
                    raise ValueError("Pending schedule edit revision changed")
            state = read_json(directory / "state.json")
            read_json(directory / "template.json")
            updated = read_json(directory / "record.json")
            prior = read_json(directory / "prior-record.json")
            marker_base = {key: value for key, value in record.items() if key != "edit_pending"}
            if (marker_base != prior or updated.get("id") != record.get("id")
                    or updated.get("folder") != record.get("folder")
                    or updated.get("revision") != revision or prior.get("id") != record.get("id")
                    or state.get("run_id") != record.get("id") or state.get("status") != "scheduled"
                    or state.get("settings", {}).get("cloud") or state.get("schedule", {}).get("released")
                    or state.get("concept") != updated.get("concept")
                    or state.get("schedule", {}).get("due_at") != updated.get("due_at")
                    or state.get("template_sha256") != digest(directory / "template.json")):
                raise ValueError("Pending schedule edit revision is inconsistent")
            state_hash = digest(folder / "state.json")
            template_hash = digest(folder / "template.original.json")
            if state_hash not in {manifest["prior-state.json"], manifest["state.json"]} or template_hash not in {
                    manifest["prior-template.json"], manifest["template.json"]}:
                raise ValueError("Project changed while a schedule edit was pending")
            atomic_bytes(folder / "template.original.json", (directory / "template.json").read_bytes())
            atomic_bytes(folder / "state.json", (directory / "state.json").read_bytes())
            updated.pop("edit_pending", None)
            self._save(updated)
        if locked:
            finish()
        else:
            with run_lock(folder):
                finish()

    def finish(self, identity, success):
        record = self.get(identity)
        record["status"] = "complete" if success else "attention"
        self._save(record)

    def reconcile_completed(self, folder: Path) -> bool:
        """Clear a stale attention ledger after an explicitly resumed saved run succeeds."""
        if not self.owner:
            return False
        folder = Path(folder).resolve()
        for record in self.records():
            if Path(record["folder"]).resolve() != folder or record["status"] != "attention":
                continue
            state = json.loads((folder / "state.json").read_text())
            if state.get("run_id") != record["id"] or state.get("status") != "complete":
                return False
            record["status"] = "complete"
            record["note"] = ""
            self._save(record)
            return True
        return False
