"""Durable local list of desktop-owned Global Volume sessions to observe."""

from __future__ import annotations

import fcntl
import json
from pathlib import Path

from .settings import atomic_json, user_dir


def _registry_path() -> Path:
    return user_dir() / "runpod" / "global-sessions.json"


def _change(project: Path | None = None, *, remove: bool = False) -> list[str]:
    target = _registry_path()
    target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    lock = target.with_suffix(".lock")
    with lock.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            rows = json.loads(target.read_text()) if target.exists() else []
            if not isinstance(rows, list) or any(not isinstance(row, str) for row in rows):
                raise ValueError("Global session index is invalid")
            if project is not None:
                path = str(project.expanduser().resolve())
                rows = [row for row in rows if row != path]
                if not remove:
                    rows.append(path)
                atomic_json(target, rows)
            return rows
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def register(project: Path) -> None:
    _change(project)


def remove(project: Path) -> None:
    _change(project, remove=True)


def listed() -> list[Path]:
    return [Path(row) for row in _change()]


def poll_registered(skip=None) -> dict:
    """Read-only status polls can trigger only the already authorized idle cleanup.

    ``skip`` is a path or a collection of paths polled elsewhere (the selected
    project, or the folder a running batch worker owns).
    """
    from .engine import cloud_action
    if skip is None:
        skip = ()
    elif isinstance(skip, (str, Path)):
        skip = (skip,)
    skipped = {Path(path).expanduser().resolve() for path in skip if path}

    projects = listed()
    from .cost_safety import session_alerts, session_notices

    summary = {"active": len(projects), "polled": 0, "needs_attention": [], "alerts": [], "notices": []}
    for project in projects:
        if project.expanduser().resolve() in skipped:
            continue
        try:
            state = cloud_action(project, "status")
            summary["polled"] += 1
            summary["alerts"].extend(session_alerts(state))
            summary["notices"].extend(f"{project.name}: {line}" for line in session_notices(state))
            if state.get("billing_confirmed_stopped") is True:
                remove(project)
        except RuntimeError as exc:
            if "already running in another simpleGen window" not in str(exc):
                summary["needs_attention"].append(str(project))
        except (OSError, ValueError, KeyError, TypeError):
            # Provider or state failure keeps the session visible and indexed.
            summary["needs_attention"].append(str(project))
    summary["active"] = len(listed())
    return summary
