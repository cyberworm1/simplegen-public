"""Managed RunPod cost safety: owner failures, collected outputs, alerts, schema drift.

Pure desktop code; no provider calls. The lifecycle decisions that use these
helpers live in ``GlobalSession.status()``.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

import httpx

from .settings import atomic_json, user_dir

log = logging.getLogger(__name__)
KINDS = ("run", "batch", "startup")
VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}
EVENTS = {
    "owner_failure": "Generation failed while the GPU is billing",
    "failure_countdown": "Failure shutdown countdown started",
    "failure_retired": "GPU retired by the failure shutdown policy",
    "outputs_not_collected": "Finished outputs are not downloaded; the GPU keeps billing until you resume",
    "unresolved_jobs": "Some jobs have no known outcome; the GPU is kept and billing until you resolve them",
    "startup_timeout": "GPU never became ready within the startup timeout",
    "delete_uncertain": "GPU shutdown could not be confirmed; the pod may still be billing",
    "spend_cap_exceeded": "Actual spend passed the batch spend cap; renders continue and are never cancelled automatically",
    "handoff_orphaned": "A warm-pod handoff did not complete; ownership was restored to the original batch",
}


def _now() -> str:
    return datetime.now(UTC).isoformat()


def owned(state: dict) -> dict | None:
    """The desktop-owned Global Volume session record, or None (legacy/none)."""
    saved = state.get("managed_session") or {}
    if saved.get("backend") != "global_desktop" or not saved.get("name"):
        return None
    return saved


# Owner failure


def record_owner_failure(state: dict, settings, reason: str, kind: str) -> bool:
    """Persistable marker that the owning run/batch failed (not detach, not cancel).

    The caller persists state before releasing its lock. The first failure is
    kept (a startup timeout stays ``startup``); Resume clears it.
    """
    if kind not in KINDS:
        raise ValueError(f"Unknown owner failure kind: {kind}")
    saved = owned(state)
    if saved is None or saved.get("retired") or saved.get("owner_failure"):
        return False
    saved["owner_failure"] = {"at": _now(), "reason": str(reason)[:500], "kind": kind}
    saved.pop("recovery_completed_at", None)
    saved["failure_idle_since"] = None
    alert(state, settings, "owner_failure" if kind != "startup" else "startup_timeout")
    return True


def clear_owner_failure(state: dict) -> bool:
    """Resume: forget the failure and stop any failure countdown before new work."""
    saved = state.get("managed_session") or {}
    changed = False
    for key in ("owner_failure", "recovery_completed_at", "failure_idle_since",
                "failure_remaining_seconds", "failure_eligible", "attention", "uncollected_outputs", "unresolved_jobs"):
        if saved.get(key) is not None:
            changed = True
        saved.pop(key, None)
    return changed


# Collected outputs


def _read(folder: Path) -> dict:
    return json.loads((folder / "state.json").read_text(encoding="utf-8"))


def owner_receipts(folder: Path, state: dict) -> list[dict]:
    """Verified local receipts for every job folder the owner can see.

    A batch gathers its members (and their variations); a one-off run its own
    folder or its variations. A folder whose downloads fail verification
    contributes nothing, so its outputs count as not collected.
    """
    from .engine import _verified_receipts
    folder = Path(folder)
    targets: list[Path] = []
    try:
        if state.get("members"):
            for member in state["members"]:
                target = Path(member["folder"])
                child = _read(target)
                children = child.get("sequence", {}).get("children", [])
                targets.extend([target / c["folder"] for c in children] or [target])
        else:
            children = state.get("sequence", {}).get("children", [])
            targets.extend([folder / c["folder"] for c in children] or [folder])
    except (OSError, ValueError, KeyError, TypeError):
        return []
    receipts = []
    for target in targets:
        try:
            receipts.extend(_verified_receipts(target, target_state(target, folder, state)))
        except (OSError, ValueError, RuntimeError, KeyError, TypeError):
            continue
    return receipts


def target_state(target: Path, folder: Path, state: dict) -> dict:
    return state if Path(target) == Path(folder) else _read(target)


_PROBED: dict[tuple, bool] = {}


def _verified_file(item: dict) -> bool:
    from .engine import probe_video
    path = Path(item["path"])
    try:
        size = path.stat().st_size
    except OSError:
        return False
    if item.get("size_bytes", 0) <= 0 or size != item["size_bytes"]:
        return False
    if PurePosixPath(item["filename"]).suffix.lower() not in VIDEO_SUFFIXES:
        return True
    key = (str(path), size, path.stat().st_mtime_ns)
    if key not in _PROBED:
        try:
            probe_video(path)
            _PROBED[key] = True
        except (OSError, RuntimeError, ValueError):
            return False
    return _PROBED[key]


def uncollected(jobs: dict, receipts: list[dict]) -> list[dict]:
    """Outputs of successful jobs without a verified local receipt (same checks as ``complete()``)."""
    found = set()
    for item in receipts:
        try:
            if _verified_file(item):
                found.add((item.get("prompt_id"), item["filename"], item.get("subfolder", ""),
                           item.get("type", "output")))
        except (KeyError, TypeError):
            continue
    missing = []
    for job in jobs.values():
        if job.get("status") != "success":
            continue
        if not job.get("outputs"):
            # Success without observed outputs is unresolved, never "fully collected".
            missing.append({"prompt_id": job.get("prompt_id"), "filename": None,
                            "note": "no outputs observed"})
            continue
        for output in job["outputs"]:
            key = (output["prompt_id"], output["filename"], output.get("subfolder", ""),
                   output.get("type", "output"))
            if key not in found:
                missing.append({"prompt_id": output["prompt_id"], "filename": output["filename"]})
    return missing


# Alerts


def _post(url: str, text: str) -> None:
    response = httpx.post(url, content=text.encode("utf-8"), timeout=5,
                          headers={"Content-Type": "text/plain; charset=utf-8"})
    response.raise_for_status()


def _elapsed(saved: dict) -> str:
    started = saved.get("session_started_at")
    try:
        seconds = int((datetime.now(UTC) - datetime.fromisoformat(started)).total_seconds())
    except (TypeError, ValueError):
        return "unknown"
    return f"{seconds // 3600}h {seconds % 3600 // 60:02d}m"


def display_name(state: dict) -> str:
    if state.get("members") is not None:
        return f"Batch {str(state.get('id', state.get('run_id', '')))[:8]}"
    line = (state.get("concept") or "").strip().splitlines()
    return f"Run: {line[0][:60]}" if line else f"Run {str(state.get('run_id', ''))[:8]}"


def message(state: dict, event: str, detail: str = "") -> str:
    """Plain text with no credentials, endpoints or file paths."""
    saved = state.get("managed_session") or {}
    rate = saved.get("actual_hourly", saved.get("quoted_hourly"))
    rate_text = f"${rate:.2f}/h" if isinstance(rate, (int, float)) else "rate unknown"
    parts = [f"simpleGen · {display_name(state)}", EVENTS[event],
             f"GPU {saved.get('gpu_id') or 'unknown'} at {rate_text}, billing {_elapsed(saved)}"]
    if detail:
        parts.append(detail)
    return "\n".join(parts)


def webhook_url() -> str:
    """The current global alert webhook; never read from a project snapshot."""
    from .settings import Settings
    try:
        return (Settings.load().alert_webhook_url or "").strip()
    except (OSError, ValueError, TypeError):
        return ""


def alert(state: dict, settings, event: str, detail: str = "", *, key: str | None = None) -> bool:
    """Record (and send once) an alert. Delivery failure never affects lifecycle.

    ``settings`` is the session's snapshot; the webhook comes from global settings.
    """
    saved = state.setdefault("managed_session", {})
    failure = (saved.get("owner_failure") or {}).get("at", "")
    key = key or f"{event}:{failure}"
    sent = saved.setdefault("alerts_sent", [])
    if key in sent:
        return False
    sent.append(key)
    text = message(state, event, detail)
    saved.setdefault("alert_log", []).append({"key": key, "event": event, "message": text, "at": _now()})
    url = webhook_url()
    if url:
        try:
            _post(url, text)
        except Exception as error:  # noqa: BLE001 -- alert delivery is best effort
            log.warning("Alert webhook failed: %s", type(error).__name__)
            errors = saved.setdefault("alert_errors", [])
            errors.append({"key": key, "error": type(error).__name__, "at": _now()})
            del errors[:-10]
    return True


def session_alerts(saved: dict) -> list[dict]:
    """Alerts of one session with keys unique across sessions (for desktop notifications)."""
    sid = saved.get("sid") or saved.get("name") or ""
    return [{"key": f"{sid}:{entry['key']}", "message": entry["message"]}
            for entry in saved.get("alert_log", []) if isinstance(entry, dict) and "message" in entry]


def session_notices(saved: dict) -> list[str]:
    """Banner lines: failure state, countdown, uncollected outputs, schema drift."""
    lines = []
    failure = saved.get("owner_failure")
    if failure and not saved.get("billing_confirmed_stopped"):
        lines.append(f"Failed ({failure.get('kind')}) while the GPU is billing. Resume, or it shuts down "
                     "after the failure grace period once every finished output is downloaded.")
    remaining = saved.get("failure_remaining_seconds")
    if remaining is not None and not saved.get("billing_confirmed_stopped"):
        lines.append(f"Failure shutdown in {remaining // 60}m {remaining % 60:02d}s.")
    if saved.get("unresolved_jobs"):
        lines.append(f"{len(saved['unresolved_jobs'])} job(s) have no known outcome; the GPU is kept and "
                     "billing until you resolve them (Resume reconnects).")
    if saved.get("attention") == "outputs_not_collected":
        lines.append(f"{len(saved.get('uncollected_outputs', []))} finished output(s) are not downloaded; "
                     "the GPU is kept until you resume or retry downloads.")
    if saved.get("delete_uncertain_at") and not saved.get("billing_confirmed_stopped"):
        lines.append("GPU shutdown could not be confirmed; retrying. Check RunPod if this persists.")
    if (saved.get("schema_drift") or {}).get("drift"):
        lines.append(drift_notice(saved["schema_drift"]))
    return lines


# Schema drift


def _model_choices(schema: dict) -> dict:
    from .preflight import _MODEL_INPUTS, _choices
    found = {}
    for kind, info in schema.items():
        if not isinstance(info, dict):
            continue
        for section in ("required", "optional"):
            for name, spec in (info.get("input", {}).get(section) or {}).items():
                if name in _MODEL_INPUTS and isinstance(spec, list) and spec:
                    choices = _choices(spec)
                    if choices is not None:
                        found[(kind, name)] = {c for c in choices if isinstance(c, str)}
    return found


def compare_schemas(live: dict, cached: dict) -> dict:
    from .preflight import schema_hash
    live_models, cached_models = _model_choices(live), _model_choices(cached)
    changed = []
    for kind, name in sorted(set(live_models) | set(cached_models)):
        new, old = live_models.get((kind, name), set()), cached_models.get((kind, name), set())
        if new != old:
            changed.append({"class_type": kind, "input": name,
                            "added": sorted(new - old)[:20], "removed": sorted(old - new)[:20]})
    live_hash, cached_hash = schema_hash(live), schema_hash(cached)
    return {"drift": live_hash != cached_hash, "live_sha256": live_hash, "cached_sha256": cached_hash,
            "added_node_types": sorted(set(live) - set(cached))[:50],
            "removed_node_types": sorted(set(cached) - set(live))[:50],
            "changed_model_choices": changed[:50]}


def pointer_path() -> Path:
    return user_dir() / "runpod" / "latest-live-schema.json"


def check_schema_drift(folder: Path, saved: dict, settings, live: dict) -> dict | None:
    """Once per session: snapshot the live schema and compare it with the cache. Never edits the cache."""
    sid = saved.get("sid")
    if not sid or (saved.get("schema_drift") or {}).get("sid") == sid:
        return None
    snapshot = Path(folder) / f"live-object-info-{sid}.json"
    atomic_json(snapshot, live)
    try:
        raw = json.loads(Path(settings.runpod_schema_file).expanduser().read_text(encoding="utf-8"))
        cached = raw.get("object_info", raw) if isinstance(raw, dict) else {}
        record = compare_schemas(live, cached)
    except (OSError, ValueError, TypeError) as error:
        record = {"drift": True, "cache_error": type(error).__name__}
    record.update(sid=sid, snapshot=str(snapshot), checked_at=_now())
    saved["schema_drift"] = record
    if record["drift"]:
        pointer = pointer_path()
        pointer.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(pointer, {"snapshot": str(snapshot), "sid": sid, "checked_at": record["checked_at"],
                              "live_sha256": record.get("live_sha256")})
    return record


def drift_notice(record: dict) -> str:
    if record.get("cache_error"):
        return "Live RunPod node schema saved; the configured cache could not be read to compare."
    added, removed = record.get("added_node_types", []), record.get("removed_node_types", [])
    models = record.get("changed_model_choices", [])
    return (f"Live RunPod node schema differs from the cached schema: {len(added)} node type(s) added, "
            f"{len(removed)} removed, {len(models)} loader model list(s) changed. "
            "Settings → Use latest live schema adopts it (batches will need preparing again).")


def adopt_live_schema(settings, snapshot: Path | None = None) -> dict:
    """Explicit user action: keep the old cache under a timestamped name, then swap atomically."""
    if snapshot is None:
        pointer = pointer_path()
        if not pointer.is_file():
            raise ValueError("No live RunPod schema has been saved yet.")
        snapshot = Path(json.loads(pointer.read_text(encoding="utf-8"))["snapshot"])
    live = json.loads(Path(snapshot).read_text(encoding="utf-8"))
    if not isinstance(live, dict) or not live or not all(
            isinstance(v, dict) and isinstance(v.get("input"), dict) for v in live.values()):
        raise ValueError("Saved live schema is empty or incompatible.")
    if not settings.runpod_schema_file:
        raise ValueError("No cached RunPod schema file is configured.")
    cache = Path(settings.runpod_schema_file).expanduser()
    backup = None
    if cache.exists():
        stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        backup = cache.with_name(f"{cache.stem}.{stamp}{cache.suffix}")
        shutil.copy2(cache, backup)
    temporary = cache.with_name(cache.name + ".adopt.tmp")
    temporary.write_text(json.dumps(live, ensure_ascii=False), encoding="utf-8")
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, cache)
    return {"installed": str(cache), "backup": str(backup) if backup else None}


# Phase 4: spend projection for batches (additions and warm-pod reuse)


def billed_hours(saved: dict, now: datetime | None = None) -> float | None:
    """Hours this owner has billed: since adoption for an adopted pod, else since the pod started."""
    since = saved.get("adopted_at") or saved.get("session_started_at")
    try:
        start = datetime.fromisoformat(since)
    except (TypeError, ValueError):
        return None
    return max(0.0, ((now or datetime.now(UTC)) - start).total_seconds() / 3600)


def _rate(state: dict) -> float | None:
    saved = state.get("managed_session") or {}
    for value in (saved.get("actual_hourly"), saved.get("quoted_hourly"),
                  (state.get("resource_choice") or {}).get("hourly_usd")):
        if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0:
            return float(value)
    return None


def released_usd(state: dict) -> float:
    """Billed by sessions this batch retired and released (``batches.release_retired_session``).

    A lower bound: a session missing its times or rate adds nothing rather than a guess.
    """
    total = 0.0
    for saved in state.get("retired_sessions") or []:
        rate = saved.get("actual_hourly") or saved.get("quoted_hourly")
        try:
            start = datetime.fromisoformat(saved.get("adopted_at") or saved["session_started_at"])
            end = datetime.fromisoformat(saved.get("session_completed_at") or saved["released_at"])
        except (KeyError, TypeError, ValueError):
            continue
        if isinstance(rate, (int, float)) and not isinstance(rate, bool) and rate > 0:
            total += max(0.0, (end - start).total_seconds() / 3600) * rate
    return total


def spent_usd(state: dict, now: datetime | None = None) -> float | None:
    hours, rate = billed_hours(state.get("managed_session") or {}, now), _rate(state)
    current = None if hours is None or rate is None else hours * rate
    earlier = released_usd(state)
    if current is None:
        return earlier or None
    return current + earlier


def _job_states(state: dict) -> list[dict]:
    jobs = []
    for member in state.get("members", []):
        try:
            target = Path(member["folder"])
            child = _read(target)
            children = child.get("sequence", {}).get("children", [])
            if children:
                for item in children:
                    sub = target / item["folder"] / "state.json"
                    jobs.append(json.loads(sub.read_text(encoding="utf-8")) if sub.is_file() else {})
            else:
                jobs.extend({} for _ in range(max(0, child.get("settings", {}).get("quantity", 1) - 1)))
                jobs.append(child)
        except (OSError, ValueError, KeyError, TypeError):
            jobs.append({})
    return jobs


def batch_projection(state: dict, extra_renders: int = 0, now: datetime | None = None) -> dict:
    """Billed so far + remaining renders × this session's measured average + the keep-warm window.

    The average comes only from completed renders' ``execution_seconds``; before
    any completes it is unknown (None), never guessed.
    """
    saved = state.get("managed_session") or {}
    rate = _rate(state)
    jobs = _job_states(state)
    measured = [j["execution_seconds"] for j in jobs if j.get("status") == "complete"
                and isinstance(j.get("execution_seconds"), (int, float))]
    average = sum(measured) / len(measured) if measured else None
    remaining = sum(1 for j in jobs if j.get("status") != "complete")
    minutes = (saved.get("frozen_profile") or {}).get("keep_warm_minutes") or 5
    billed = spent_usd(state, now) or 0.0

    def cost(renders: int) -> float | None:
        return None if average is None or rate is None else renders * average / 3600 * rate
    keep_warm = minutes / 60 * rate if rate is not None else None
    remaining_usd, extra_usd = cost(remaining), cost(extra_renders)
    known = [v for v in (billed, keep_warm, remaining_usd, extra_usd) if v is not None]
    total = (billed + keep_warm + remaining_usd + (extra_usd or 0.0)
             if None not in (keep_warm, remaining_usd) and (extra_usd is not None or not extra_renders) else None)
    return {"hourly_usd": rate, "billed_usd": round(billed, 2), "average_render_seconds": average,
            "remaining_renders": remaining, "remaining_usd": remaining_usd, "keep_warm_usd": keep_warm,
            "extra_renders": extra_renders, "extra_usd": extra_usd, "total_usd": total,
            "lower_bound_usd": round(sum(known), 2), "cap_usd": state.get("max_total_usd")}


def projection_text(projection: dict) -> str:
    def money(value):
        return "unknown" if value is None else f"${value:.2f}"
    remaining = ("render time unknown" if projection["average_render_seconds"] is None
                 else money(projection["remaining_usd"]))
    text = (f"billed so far {money(projection['billed_usd'])} at {money(projection['hourly_usd'])}/h; "
            f"{projection['remaining_renders']} render(s) remaining {remaining}; "
            f"keep-warm {money(projection['keep_warm_usd'])}")
    if projection["extra_renders"]:
        text += f"; this addition {money(projection['extra_usd'])}"
    text += f"; projected total {money(projection['total_usd'])}"
    if projection.get("cap_usd") is not None:
        text += f" (cap ${projection['cap_usd']:.2f})"
    return text


def confirmation_text(projection: dict) -> str:
    """What the user confirms before queueing an addition. With a cap set and an
    incomplete projection, say so explicitly (never refused, never guessed)."""
    text = "Projected spend with this addition: " + projection_text(projection)
    if projection.get("cap_usd") is not None and projection.get("total_usd") is None:
        text = (f"A spend cap of ${projection['cap_usd']:.2f} is set, but the total cannot be projected yet: "
                f"render time unknown (no render has finished). Billed so far: ${projection['billed_usd']:.2f}.\n\n"
                + text)
    return text


def over_cap(projection: dict) -> bool:
    """Reject only on known numbers: the known lower bound already exceeds the cap."""
    cap = projection.get("cap_usd")
    return cap is not None and projection["lower_bound_usd"] > cap
