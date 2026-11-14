"""Local, content-addressed cache of the references a run actually uses (Phase 6).

Accepted agent work snapshots every reference into
``user_dir()/reference-cache/blobs/<sha256><ext>`` and records it in the run's
``references.json``; later steps (uploads, batch fingerprints, saved-run
preflight) read the verified blobs, so the linked project folder does not need
to be mounted after acceptance. Originals are only ever read.

Each run holding blobs has a pin (``pins/<run_id>.json``). Pins are released
when a run, batch or schedule completes successfully; garbage collection
removes unpinned blobs under a cache lock.
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import logging
import os
import re
import time
import uuid
from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from pathlib import Path

from .settings import atomic_json, user_dir

MANIFEST = "references.json"
VERSION = 1
TEMP_PREFIX = ".tmp-"
STALE_TEMP_SECONDS = 24 * 3600
_BLOB = re.compile(r"^[0-9a-f]{64}(\.[a-z0-9]{1,10})?$")
_log = logging.getLogger(__name__)


class ReferenceCacheError(RuntimeError):
    """A reference cannot be captured right now; nothing was accepted, retrying is safe."""


class IntegrityError(RuntimeError):
    """A cached blob is corrupt and its original no longer matches the recorded hash."""


def root() -> Path:
    return user_dir() / "reference-cache"


def _blobs() -> Path:
    return root() / "blobs"


def _pins() -> Path:
    return root() / "pins"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@contextlib.contextmanager
def cache_lock():
    root().mkdir(parents=True, exist_ok=True)
    with (root() / ".lock").open("a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _fsync_dir(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _extension(path: Path) -> str:
    suffix = path.suffix.lower()
    return suffix if re.fullmatch(r"\.[a-z0-9]{1,10}", suffix) else ""


def _copy_to_temp(source: Path) -> tuple[Path, str, int]:
    """Copy ``source`` to a temporary blob, fsync it and verify the written bytes."""
    _blobs().mkdir(parents=True, exist_ok=True)
    temp = _blobs() / f"{TEMP_PREFIX}{uuid.uuid4().hex}"
    digest, size = hashlib.sha256(), 0
    try:
        with source.open("rb") as reader, temp.open("xb") as writer:
            for chunk in iter(lambda: reader.read(1024 * 1024), b""):
                digest.update(chunk)
                writer.write(chunk)
                size += len(chunk)
            writer.flush()
            os.fsync(writer.fileno())
        if file_sha256(temp) != digest.hexdigest():
            raise OSError("written copy does not match the original")
    except BaseException:
        temp.unlink(missing_ok=True)
        raise
    return temp, digest.hexdigest(), size


def blob_path(entry: dict) -> Path:
    name = str(entry.get("blob", ""))
    if not _BLOB.match(name) or not name.startswith(str(entry.get("sha256", ""))):
        raise IntegrityError(f"Reference manifest has an invalid blob name: {name!r}")
    return _blobs() / name


def verified(entry: dict) -> bool:
    try:
        path = blob_path(entry)
        return path.is_file() and not path.is_symlink() and file_sha256(path) == entry["sha256"]
    except (OSError, IntegrityError, KeyError):
        return False


def _place(temp: Path, sha: str, extension: str) -> str:
    """Under the cache lock: publish ``temp`` as the blob, reusing a verified existing one."""
    name = sha + extension
    target = _blobs() / name
    if target.is_file() and not target.is_symlink() and file_sha256(target) == sha:
        temp.unlink(missing_ok=True)
    else:
        os.replace(temp, target)  # Also replaces a corrupt blob with verified bytes.
        _fsync_dir(_blobs())
    return name


def _write_pin(run_id: str, blobs: Iterable[str], folder: Path | None) -> None:
    names = sorted(set(blobs))
    pin = _pins() / f"{run_id}.json"
    if not names:
        pin.unlink(missing_ok=True)
        return
    atomic_json(pin, {"run_id": run_id, "folder": str(folder) if folder else None,
                      "blobs": names, "pinned_at": datetime.now(UTC).isoformat()})


def _normal(value: str, roots: list[Path]) -> str:
    path = Path(value).expanduser()
    if not path.is_absolute() and roots:
        path = roots[0] / path
    return os.path.normpath(str(path))


# Manifests

def load(folder: Path) -> dict | None:
    path = Path(folder) / MANIFEST
    if not path.is_file():
        return None
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if (not isinstance(manifest, dict) or not isinstance(manifest.get("references"), list)
            or not all(isinstance(e, dict) and isinstance(e.get("original"), str)
                       and isinstance(e.get("sha256"), str) for e in manifest["references"])):
        return None
    return manifest


def manifest_for(folder: Path, state: dict | None = None) -> dict | None:
    """The run's manifest; a variation child uses its sequence parent's."""
    folder = Path(folder)
    manifest = load(folder)
    if manifest is None and state and state.get("sequence_parent"):
        manifest = load((folder / state["sequence_parent"]).resolve())
    return manifest


def find(manifest: dict | None, value: str, roots: list[Path]) -> dict | None:
    if not manifest:
        return None
    normal = _normal(value, roots)
    for entry in manifest["references"]:
        if (value == entry["original"] or value in entry.get("requested", [])
                or normal == os.path.normpath(entry["original"])):
            return entry
    return None


def _used_lines(folder: Path) -> list[str]:
    path = Path(folder) / "used_references.txt"
    if not path.is_file():
        return []
    return [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def complete(folder: Path, state: dict | None = None, *, verify: bool = True) -> bool:
    """True when a manifest covers every saved reference and (optionally) each blob verifies."""
    folder = Path(folder)
    if state is None:
        try:
            state = json.loads((folder / "state.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False
    if not state.get("workflow_acceptance"):
        return False
    manifest = manifest_for(folder, state)
    if manifest is None:
        return False
    if any(find(manifest, line, []) is None for line in _used_lines(folder)):
        return False
    return not verify or all(verified(entry) for entry in manifest["references"])


# Capture

def snapshot(run_id: str, values: Iterable[str], roots: list[Path], resolve: Callable[[str], Path], *,
             reuse: dict | None = None, folder: Path | None = None) -> dict:
    """Store every referenced file and pin it for ``run_id``; return the manifest (not yet written).

    ``resolve`` maps a requested value to its original file (raising if it is
    missing or outside the linked folders). Entries of ``reuse`` whose blobs
    verify are kept without touching the originals.
    """
    groups: dict[str, dict] = {}
    pending: list[tuple[str, Path]] = []
    for value in values:
        entry = find(reuse, value, roots)
        if entry is not None and verified(entry):
            group = groups.setdefault(entry["original"], {**entry, "requested": []})
        else:
            try:
                original = resolve(value)
            except (OSError, ValueError) as error:
                raise ReferenceCacheError(
                    f"Reference cannot be read right now: {value} ({error}). Nothing was accepted; make the "
                    "linked project folder available and retry; saved agent work is reused.") from None
            group = groups.get(str(original))
            if group is None:
                group = groups[str(original)] = {"original": str(original), "requested": []}
                pending.append((str(original), original))
        if value not in group["requested"]:
            group["requested"].append(value)
    temps: list[Path] = []
    try:
        stored = {}
        for key, original in pending:
            try:
                temp, sha, size = _copy_to_temp(original)
            except OSError as error:
                raise ReferenceCacheError(
                    f"Reference cannot be read right now: {original} ({type(error).__name__}). Nothing was "
                    "accepted; make the linked project folder available and retry; saved agent work is "
                    "reused.") from None
            temps.append(temp)
            stored[key] = (temp, sha, size, _extension(original))
        captured_at = datetime.now(UTC).isoformat()
        with cache_lock():
            for key, (temp, sha, size, extension) in stored.items():
                groups[key].update(sha256=sha, size=size, blob=_place(temp, sha, extension),
                                   captured_at=captured_at)
            # The pin is written in the same critical section as the blobs, so GC never sees them unpinned.
            _write_pin(run_id, (g["blob"] for g in groups.values()), folder)
    finally:
        for temp in temps:
            temp.unlink(missing_ok=True)
    return {"version": VERSION, "run_id": run_id,
            "references": [{key: group[key] for key in
                            ("original", "requested", "sha256", "size", "blob", "captured_at")}
                           for group in groups.values()]}


def capture(folder: Path, run_id: str, values: Iterable[str], roots: list[Path],
            resolve: Callable[[str], Path], *, reuse: dict | None = None) -> dict:
    """Snapshot and pin, then write ``<folder>/references.json`` atomically."""
    folder = Path(folder)
    manifest = snapshot(run_id, values, roots, resolve, reuse=reuse, folder=folder)
    atomic_json(folder / MANIFEST, manifest)
    return manifest


def run_values(prepared: dict, plan: dict) -> list[str]:
    """Every LoadImage slot value and every used_references entry of accepted work."""
    values = [slot["value"] for slot in prepared.get("reference_slots", []) if isinstance(slot.get("value"), str)]
    values.extend(p for p in plan.get("used_references", []) if isinstance(p, str))
    return values


def settings_roots(settings) -> list[Path]:
    return settings.reference_roots()


def capture_run(folder: Path, run_id: str, settings, prepared: dict, plan: dict,
                *, reuse: dict | None = None) -> dict:
    """Acceptance-time capture, resolving each value like the engine does."""
    from . import engine
    roots = settings_roots(settings)
    return capture(folder, run_id, run_values(prepared, plan), roots,
                   lambda value: engine._inside(value, roots), reuse=reuse)


def cache_saved(folder: Path) -> dict:
    """"Cache references now" for an accepted run saved before the cache existed."""
    from . import engine
    folder = Path(folder).expanduser().resolve()
    with engine.run_lock(folder):
        state = json.loads((folder / "state.json").read_text(encoding="utf-8"))
        if not state.get("workflow_acceptance"):
            raise ReferenceCacheError("Only accepted agent work has references to cache.")
        if state.get("batch_parent") or state.get("sequence_parent"):
            raise ReferenceCacheError("Batch members and variations are cached with their owner.")
        if state.get("prompt_id") or state.get("submission_started"):
            raise ReferenceCacheError("This run was already submitted; its references were uploaded.")
        from .settings import Settings
        roots = settings_roots(Settings(**state["settings"]))
        plan = json.loads((folder / "plan.json").read_text(encoding="utf-8"))
        values = _used_lines(folder) + [p for p in plan.get("used_references", []) if isinstance(p, str)]
        references = plan.get("references")
        if isinstance(references, dict):
            values.extend(v for v in references.values() if isinstance(v, str))
        elif isinstance(references, list):
            values.extend(r["path"] for r in references if isinstance(r, dict) and isinstance(r.get("path"), str))
        return capture(folder, state["run_id"], values, roots, lambda value: engine._inside(value, roots),
                       reuse=load(folder))


# Use

def localize(entry: dict, report: Callable[[str], None] | None = None) -> Path:
    """The verified file to read for ``entry``: its blob, or a still-matching original.

    A corrupt or missing blob is reported and never used. The original is used
    only when its hash matches the recorded sha256 (the blob is then restored).
    """
    try:
        path = blob_path(entry)
    except IntegrityError:
        path = None
    if path is not None and verified(entry):
        return path
    name = Path(entry["original"]).name
    message = f"Cached reference {name} is missing or corrupted (sha256 mismatch); it will not be used."
    _log.warning(message)
    if report:
        report(message)
    original = Path(entry["original"])
    try:
        matches = original.is_file() and file_sha256(original) == entry["sha256"]
    except OSError:
        matches = False
    if not matches:
        raise IntegrityError(
            f"Cached reference {name} is corrupted and the original is unavailable or has changed since "
            "acceptance. Nothing was uploaded; make the unchanged original available, or start a new run.")
    with contextlib.suppress(OSError, IntegrityError):
        temp, sha, _ = _copy_to_temp(original)
        if sha != entry["sha256"]:
            temp.unlink(missing_ok=True)
        else:
            with cache_lock():
                _place(temp, sha, "" if path is None else path.name[64:])
    return original


# Pins and cleanup

def pin_existing(run_id: str, manifest: dict, folder: Path | None = None) -> None:
    with cache_lock():
        _write_pin(run_id, (e["blob"] for e in manifest["references"]), folder)


def pinned(run_id: str) -> bool:
    return (_pins() / f"{run_id}.json").is_file()


def release(run_ids: Iterable[str], *, collect: bool = True) -> None:
    """Drop the pins of successfully completed runs, then collect unpinned blobs."""
    ids = [r for r in run_ids if r and re.fullmatch(r"[A-Za-z0-9_.-]+", r)]
    if not ids or not root().is_dir():
        return
    with cache_lock():
        for run_id in ids:
            (_pins() / f"{run_id}.json").unlink(missing_ok=True)
    if collect:
        gc()


def release_quietly(run_ids: Iterable[str]) -> None:
    """Release after success; a cache problem must never fail completed work."""
    try:
        release(run_ids)
    except OSError as error:
        _log.warning("Reference cache release failed: %s", error)


def gc(*, stale_seconds: float = STALE_TEMP_SECONDS) -> dict:
    """Delete blobs with no pin, and abandoned temporary copies, under the cache lock."""
    if not _blobs().is_dir():
        return {"removed": 0, "bytes": 0}
    removed = freed = 0
    with cache_lock():
        keep: set[str] = set()
        if _pins().is_dir():
            for pin in _pins().glob("*.json"):
                try:
                    keep.update(json.loads(pin.read_text(encoding="utf-8"))["blobs"])
                except (OSError, ValueError, KeyError, TypeError):
                    _log.warning("Unreadable reference pin %s; skipping cleanup.", pin.name)
                    return {"removed": 0, "bytes": 0, "skipped": pin.name}
        now = time.time()
        for path in _blobs().iterdir():
            try:
                if path.name.startswith(TEMP_PREFIX):
                    if now - path.stat().st_mtime < stale_seconds:
                        continue
                elif path.name in keep or not _BLOB.match(path.name):
                    continue
                size = path.stat().st_size
                path.unlink()
            except OSError:
                continue
            removed += 1
            freed += size
    return {"removed": removed, "bytes": freed}


def gc_quietly() -> None:
    """App-start collection; a cache problem never blocks the app."""
    try:
        gc()
    except OSError as error:
        _log.warning("Reference cache cleanup failed: %s", error)


def usage() -> dict:
    """Total and unpinned cache size in bytes (for Settings)."""
    total = unused = 0
    keep: set[str] = set()
    if _pins().is_dir():
        for pin in _pins().glob("*.json"):
            with contextlib.suppress(OSError, ValueError, KeyError, TypeError):
                keep.update(json.loads(pin.read_text(encoding="utf-8"))["blobs"])
    if _blobs().is_dir():
        for path in _blobs().iterdir():
            with contextlib.suppress(OSError):
                size = path.stat().st_size
                total += size
                if path.name not in keep:
                    unused += size
    return {"bytes": total, "unused_bytes": unused}
