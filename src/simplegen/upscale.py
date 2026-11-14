"""4K/60 upscale: service client and durable local job records (Phase 8).

A separate upscale service (not part of this repository) does the GPU work;
its HTTP API is the one ``Client`` below calls. This module owns the desktop side:

- one JSON record per upscale attempt under ``user_dir()/upscale/jobs``, written
  before the upload (the job id is chosen here, so repeating the upload never
  creates a second job);
- ``advance`` moves a record one step: upload, poll, download, verify, then
  delete the host copy. Unreachable hosts only delay; nothing is retried after
  a failure without the owner choosing Retry;
- verified outputs land in ``<run>/final/4k60/`` with ``manifest.json``. The
  run's own state, receipts and final cut are never changed.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import stat
import subprocess
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import httpx

from .settings import Settings, atomic_json, user_dir

log = logging.getLogger(__name__)

API = "/v1"
FPS = 60
UHD = (3840, 2160)
FRAMINGS = (
    ("long_side", "Long side 3840"),
    ("fit_uhd", "Fit inside 3840 × 2160"),
    ("fill_16x9", "Fill 16:9 (crop)"),
    ("pad_16x9", "Pad 16:9"),
)
# Upscale methods ranked by a side-by-side quality comparison, fastest first.
METHODS = (
    ("rtxvsr+rife", "RTX Video Super Resolution", "fastest"),
    ("remacri+rife", "Remacri", "fast"),
    ("seedvr2-3b+rife", "SeedVR2 3B", "slow"),
)
DEFAULT_METHOD = "remacri+rife"  # "fast"; also the upscale host's default preset.
OUTPUT_DIR = Path("final") / "4k60"


def method_label(preset: str, host_label: str = "") -> str:
    """Display name for an upscale preset, with the speed note for the preferred methods."""
    for key, label, speed in METHODS:
        if key == preset:
            return f"{label} ({speed})"
    return host_label or preset


ACTIVE = ("pending", "remote", "downloading", "cancelling")
FLAGGED = ("failed", "cancelled")
REMOTE_TERMINAL = ("done", "failed", "cancelled")
CHUNK = 1024 * 1024


class UpscaleError(RuntimeError):
    """The service refused or failed a request; the message is user-facing."""


class Unreachable(UpscaleError):
    """The service could not be contacted. Callers wait; this is never a job failure."""


def now() -> str:
    return datetime.now(UTC).isoformat()


# Framing (mirrors the service; the service is authoritative for output size)


def _even(value: float) -> int:
    return max(2, round(value / 2) * 2)


def target_size(width: int, height: int, framing: str) -> tuple[int, int]:
    if width <= 0 or height <= 0:
        raise ValueError("Source dimensions must be positive.")
    if framing == "long_side":
        scale = 3840 / max(width, height)
        return _even(width * scale), _even(height * scale)
    if framing == "fit_uhd":
        scale = min(UHD[0] / width, UHD[1] / height)
        return _even(width * scale), _even(height * scale)
    if framing in ("fill_16x9", "pad_16x9"):
        return UHD
    raise ValueError("Choose a supported 4K framing.")


def framing_for(settings: Settings) -> str:
    override = (settings.upscale_framing_by_aspect or {}).get(settings.aspect_ratio or "")
    return override or settings.upscale_framing


# Token (never in settings, run state or logs)


def token_path() -> Path:
    return user_dir() / "upscale" / "token"


def load_token() -> str:
    value = os.environ.get("SIMPLEGEN_UPSCALE_TOKEN", "").strip()
    if value:
        return value
    path = token_path()
    try:
        if path.is_symlink() or stat.S_IMODE(path.stat().st_mode) & 0o077:
            raise UpscaleError("The upscale token file must be private (mode 600).")
        return path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return ""


def save_token(value: str) -> None:
    path = token_path()
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(value.strip() + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


# HTTP client


class Client:
    def __init__(self, url: str, token: str, *, transport: httpx.BaseTransport | None = None,
                 timeout: float = 10.0):
        if not url.strip():
            raise UpscaleError("No upscale service is configured. Add its URL in Settings.")
        self.http = httpx.Client(base_url=url.rstrip("/") + API, transport=transport, timeout=timeout,
                                 headers={"Authorization": f"Bearer {token}"} if token else {})

    @classmethod
    def from_settings(cls, settings: Settings, **kwargs) -> Client:
        return cls(settings.upscale_url, load_token(), **kwargs)

    def close(self) -> None:
        self.http.close()

    def _request(self, method: str, path: str, **kwargs) -> httpx.Response:
        try:
            response = self.http.request(method, path, **kwargs)
        except httpx.TransportError as error:
            raise Unreachable(f"Upscale service is unreachable ({type(error).__name__}).") from None
        if response.status_code == 401:
            raise UpscaleError("The upscale service rejected the token. Check it in Settings.")
        return response

    @staticmethod
    def _error(response: httpx.Response, action: str) -> UpscaleError:
        try:
            detail = response.json().get("detail") or response.json().get("error") or ""
        except (ValueError, AttributeError):
            detail = ""
        detail = detail if isinstance(detail, str) else json.dumps(detail)[:300]
        return UpscaleError(f"{action} failed (HTTP {response.status_code}){': ' + detail if detail else ''}.")

    def health(self) -> dict:
        response = self._request("GET", "/health")
        if response.status_code != 200:
            raise self._error(response, "Health check")
        return response.json()

    def job(self, job_id: str) -> dict | None:
        response = self._request("GET", f"/jobs/{job_id}")
        if response.status_code == 404:
            return None
        if response.status_code != 200:
            raise self._error(response, "Job status")
        return response.json()

    def put(self, job_id: str, source: Path, *, sha256: str, preset: str, framing: str,
            progress: Callable[[int, int], None] | None = None) -> dict:
        total = source.stat().st_size

        def body():
            sent = 0
            with source.open("rb") as handle:
                while chunk := handle.read(CHUNK):
                    sent += len(chunk)
                    if progress:
                        progress(sent, total)
                    yield chunk

        params = {"preset": preset, "framing": framing, "sha256": sha256, "name": source.name}
        response = self._request("PUT", f"/jobs/{job_id}", params=params, content=body(),
                                 headers={"Content-Type": "application/octet-stream",
                                          "Content-Length": str(total)},
                                 timeout=httpx.Timeout(10.0, write=600.0, read=600.0))
        if response.status_code in (200, 201):
            return response.json()
        if response.status_code == 507:
            raise UpscaleError("The upscale host does not have enough free disk for this video.")
        raise self._error(response, "Upload")

    def download(self, job_id: str, target: Path,
                 progress: Callable[[int, int], None] | None = None) -> tuple[str, int]:
        """Stream the result to ``target`` through a partial file; return (sha256, bytes)."""
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_name(target.name + ".part")
        digest, size = hashlib.sha256(), 0
        try:
            with self.http.stream("GET", f"/jobs/{job_id}/result",
                                  timeout=httpx.Timeout(10.0, read=600.0)) as response:
                if response.status_code != 200:
                    response.read()
                    raise self._error(response, "Download")
                expected = response.headers.get("X-Output-SHA256", "").lower()
                total = int(response.headers.get("Content-Length") or 0)
                with partial.open("wb") as handle:
                    for chunk in response.iter_bytes(CHUNK):
                        handle.write(chunk)
                        digest.update(chunk)
                        size += len(chunk)
                        if progress:
                            progress(size, total)
                    handle.flush()
                    os.fsync(handle.fileno())
        except httpx.TransportError as error:
            partial.unlink(missing_ok=True)
            raise Unreachable(f"Upscale download interrupted ({type(error).__name__}).") from None
        actual = digest.hexdigest()
        if expected and expected != actual:
            partial.unlink(missing_ok=True)
            raise UpscaleError("The downloaded 4K60 video does not match the host's checksum.")
        partial.replace(target)
        return actual, size

    def cancel(self, job_id: str) -> dict | None:
        response = self._request("POST", f"/jobs/{job_id}/cancel")
        if response.status_code == 404:
            return None
        if response.status_code != 200:
            raise self._error(response, "Cancel")
        return response.json()

    def delete(self, job_id: str) -> bool:
        """True once the host no longer holds the job; False while it is still running."""
        response = self._request("DELETE", f"/jobs/{job_id}")
        if response.status_code in (204, 404):
            return True
        if response.status_code == 409:
            return False
        raise self._error(response, "Host clean-up")


# Media checks


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def _rate(text: str | None) -> float:
    try:
        numerator, _, denominator = (text or "0/0").partition("/")
        return float(numerator) / float(denominator or 1)
    except (ValueError, ZeroDivisionError):
        return 0.0


def probe(path: Path) -> dict:
    from .agents import executable
    completed = subprocess.run(
        [executable("ffprobe"), "-v", "error", "-show_entries",
         "format=duration:stream=codec_type,codec_name,width,height,avg_frame_rate,r_frame_rate",
         "-of", "json", str(path)],
        capture_output=True, text=True, timeout=60, check=False)
    if completed.returncode:
        raise UpscaleError(f"{path.name} could not be inspected by ffprobe.")
    data = json.loads(completed.stdout)
    streams = data.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    if not video:
        raise UpscaleError(f"{path.name} has no video stream.")
    return {"width": int(video.get("width") or 0), "height": int(video.get("height") or 0),
            "fps": _rate(video.get("avg_frame_rate")) or _rate(video.get("r_frame_rate")),
            "codec": video.get("codec_name"), "duration": float(data.get("format", {}).get("duration") or 0),
            "has_audio": any(s.get("codec_type") == "audio" for s in streams),
            "size_bytes": path.stat().st_size}


def verify_output(source: dict, output: dict, framing: str) -> list[str]:
    """Problems with a downloaded result; empty means it is accepted."""
    problems = []
    expected = target_size(source["width"], source["height"], framing)
    if (output["width"], output["height"]) != expected:
        problems.append(f"size is {output['width']}×{output['height']}, expected {expected[0]}×{expected[1]}")
    if abs(output["fps"] - FPS) > 0.01:
        problems.append(f"frame rate is {output['fps']:.3f}, expected {FPS}")
    if abs(output["duration"] - source["duration"]) > 1.5 / FPS + 0.05:
        problems.append(f"duration is {output['duration']:.3f}s, source is {source['duration']:.3f}s")
    if source["has_audio"] and not output["has_audio"]:
        problems.append("audio is missing")
    return problems


# Records


def jobs_dir() -> Path:
    return user_dir() / "upscale" / "jobs"


def _path(job_id: str) -> Path:
    return jobs_dir() / f"{job_id}.json"


def save(record: dict) -> dict:
    record["updated_at"] = now()
    atomic_json(_path(record["id"]), record)
    return record


def load(job_id: str) -> dict:
    return json.loads(_path(job_id).read_text(encoding="utf-8"))


def records() -> list[dict]:
    found = []
    folder = jobs_dir()
    if not folder.is_dir():
        return found
    for path in sorted(folder.glob("*.json")):
        try:
            found.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            log.warning("Unreadable upscale record %s", path.name)
    return sorted(found, key=lambda r: r.get("created_at", ""))


def run_folder_for(final: Path) -> Path:
    """The run folder that owns a final cut (``<run>/final/<file>``)."""
    folder = final.resolve().parent.parent
    if not (folder / "state.json").is_file():
        raise ValueError("This video is not a saved simpleGen final cut.")
    return folder


def records_for(source: Path) -> list[dict]:
    source = source.resolve()
    return [r for r in records() if Path(r["run_folder"], r["source"]).resolve() == source]


def create(final: Path, *, framing: str, preset: str = "", origin: str = "manual",
           retry_of: str | None = None) -> dict:
    """Write the intent record. No network: the upload happens in ``advance``."""
    final = final.resolve()
    folder = run_folder_for(final)
    if framing not in dict(FRAMINGS):
        raise ValueError("Choose a supported 4K framing.")
    if any(r["status"] in ACTIVE for r in records_for(final)):
        raise ValueError("This video already has an upscale in progress.")
    state = json.loads((folder / "state.json").read_text(encoding="utf-8"))
    return save({
        "schema": 1, "id": str(uuid.uuid4()), "created_at": now(), "origin": origin, "retry_of": retry_of,
        "run_folder": str(folder), "run_id": state.get("run_id"), "concept": (state.get("concept") or "")[:200],
        "source": str(final.relative_to(folder)), "source_sha256": sha256_file(final),
        "source_bytes": final.stat().st_size, "framing": framing, "preset": preset,
        "status": "pending", "remote": None, "output": None, "error": None,
        "host_cleared": False, "alerted": False,
    })


def enqueue_if_requested(folder: Path, state: dict) -> dict | None:
    """Engine hook after a final cut is verified. Never raises: the render is already complete."""
    try:
        settings = Settings(**state["settings"])
        if not settings.upscale or not (state.get("result") or {}).get("final_path"):
            return None
        final = (Path(folder) / state["result"]["final_path"]).resolve()
        existing = [r for r in records_for(final) if r["status"] not in ("dismissed",)]
        if existing:
            return existing[-1]
        return create(final, framing=framing_for(settings), preset=settings.upscale_preset, origin="auto")
    except Exception as error:  # noqa: BLE001 -- the render result must never depend on upscaling
        log.warning("Upscale request not recorded: %s", error)
        return None


def _source(record: dict) -> Path:
    return Path(record["run_folder"]) / record["source"]


def _output_name(record: dict) -> Path:
    stem = Path(record["source"]).stem
    suffix = "" if record["framing"] == "long_side" else f"_{record['framing']}"
    return OUTPUT_DIR / f"{stem}_4k60{suffix}.mp4"


def _fail(record: dict, message: str) -> dict:
    record.update(status="failed", error=message, failed_at=now())
    return save(record)


def advance(record: dict, client: Client, *, health: dict | None = None,
            progress: Callable[[dict], None] | None = None) -> dict:
    """Move one record forward as far as it can go now. Idempotent and crash-safe."""
    job_id = record["id"]
    if _cancel_marker(job_id).exists():
        if record["status"] == "pending":
            record.update(status="cancelled", cancelled_at=now(), host_cleared=True)
        elif record["status"] in ("remote", "downloading"):
            record["status"] = "cancelling"
        _cancel_marker(job_id).unlink(missing_ok=True)
        save(record)
    if record["status"] == "pending":
        source = _source(record)
        if not source.is_file() or sha256_file(source) != record["source_sha256"]:
            return _fail(record, "The final cut changed or is missing; it was not uploaded.")
        preset = record["preset"] or ((health or client.health()).get("default_preset") or "")
        if not preset:
            record["note"] = "Waiting for a default preset on the upscale host."
            return save(record)
        remote = client.job(job_id)
        if remote is None:
            def uploaded(sent, total):
                if progress:
                    progress({"id": job_id, "stage": "Uploading", "fraction": sent / total if total else None})
            try:
                remote = client.put(job_id, source, sha256=record["source_sha256"], preset=preset,
                                    framing=record["framing"], progress=uploaded)
            except Unreachable:
                raise
            except UpscaleError as error:
                return _fail(record, str(error))
        record.update(status="remote", preset=preset, remote=remote, note=None, submitted_at=now())
        save(record)
    if record["status"] == "cancelling":
        remote = client.cancel(job_id)
        if remote is None or remote["state"] in REMOTE_TERMINAL:
            record.update(status="cancelled", remote=remote, cancelled_at=now())
            record["host_cleared"] = client.delete(job_id)
        return save(record)
    if record["status"] == "remote":
        remote = client.job(job_id)
        if remote is None:
            return _fail(record, "The upscale host no longer has this job.")
        record["remote"] = remote
        if remote["state"] in ("failed", "cancelled"):
            return _fail(record, remote.get("error") or f"The upscale host reports the job {remote['state']}.")
        if remote["state"] != "done":
            return save(record)
        record["status"] = "downloading"
        save(record)
    if record["status"] == "downloading":
        folder = Path(record["run_folder"])
        target = folder / _output_name(record)

        def downloaded(size, total):
            if progress:
                progress({"id": job_id, "stage": "Downloading", "fraction": size / total if total else None})
        try:
            sha, size = client.download(job_id, target, downloaded)
        except Unreachable:
            raise
        except UpscaleError as error:
            # The host copy stays until the owner retries or dismisses.
            return _fail(record, str(error))
        expected = ((record.get("remote") or {}).get("output") or {}).get("sha256")
        problems = [] if not expected or expected == sha else ["checksum differs from the host's record"]
        try:
            source_meta, output_meta = probe(_source(record)), probe(target)
            problems += verify_output(source_meta, output_meta, record["framing"])
        except UpscaleError as error:
            problems.append(str(error))
        if problems:
            target.unlink(missing_ok=True)
            # The host copy stays until the owner retries or dismisses.
            return _fail(record, "The 4K60 video failed verification: " + "; ".join(problems) + ".")
        record.update(status="complete", completed_at=now(), error=None, output={
            "path": str(target.relative_to(folder)), "sha256": sha, "bytes": size, "metadata": output_meta})
        _write_manifest(record)
        save(record)
    if record["status"] == "complete" and not record["host_cleared"]:
        record["host_cleared"] = client.delete(job_id)
        save(record)
    if record["status"] == "dismissed" and not record["host_cleared"]:
        remote = client.job(job_id)
        if remote is not None and remote["state"] not in REMOTE_TERMINAL:
            client.cancel(job_id)
            return save(record)
        record["host_cleared"] = client.delete(job_id)
        save(record)
    return record


def _write_manifest(record: dict) -> None:
    folder = Path(record["run_folder"])
    path = folder / OUTPUT_DIR / "manifest.json"
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        manifest = {"schema": 1, "outputs": []}
    manifest["outputs"].append({
        "job_id": record["id"], "source": record["source"], "source_sha256": record["source_sha256"],
        "output": record["output"]["path"], "sha256": record["output"]["sha256"],
        "bytes": record["output"]["bytes"], "preset": record["preset"], "framing": record["framing"],
        "metadata": record["output"]["metadata"], "completed_at": record["completed_at"]})
    atomic_json(path, manifest)


def upscaled_version(final: Path) -> Path | None:
    """The newest verified 4K60 output of a final cut, if any."""
    final = final.resolve()
    try:
        folder = run_folder_for(final)
        manifest = json.loads((folder / OUTPUT_DIR / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    source = str(final.relative_to(folder))
    for entry in reversed(manifest.get("outputs", [])):
        path = folder / entry["output"]
        if entry.get("source") == source and path.is_file():
            return path
    return None


def upscaled_metadata(output: Path) -> dict | None:
    output = output.resolve()
    folder = output.parent.parent.parent
    try:
        manifest = json.loads((folder / OUTPUT_DIR / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    for entry in reversed(manifest.get("outputs", [])):
        if (folder / entry["output"]).resolve() == output:
            return entry.get("metadata")
    return None


# Owner actions


def _cancel_marker(job_id: str) -> Path:
    return jobs_dir() / f"{job_id}.cancel"


def request_cancel(job_id: str) -> None:
    """Ask the background tick to cancel; it owns active records, so the UI never rewrites them."""
    if load(job_id)["status"] not in ("pending", "remote", "downloading"):
        raise ValueError("This upscale is not running.")
    _cancel_marker(job_id).write_text(now() + "\n", encoding="utf-8")


def dismiss(job_id: str) -> dict:
    record = load(job_id)
    if record["status"] not in FLAGGED:
        raise ValueError("Only a failed or cancelled upscale can be dismissed.")
    record.update(status="dismissed", dismissed_at=now())
    return save(record)


def retry(job_id: str) -> dict:
    """The owner's explicit retry: dismiss the flagged record, start a new one."""
    old = load(job_id)
    if old["status"] not in FLAGGED:
        raise ValueError("Only a failed or cancelled upscale can be retried.")
    dismiss(job_id)
    return create(_source(old), framing=old["framing"], preset=old["preset"], origin="retry", retry_of=job_id)


# Background tick (called off the UI thread)


def describe(record: dict) -> str:
    remote = record.get("remote") or {}
    status = record["status"]
    if status == "pending":
        return record.get("note") or "4K60 upscale waiting to upload."
    if status == "remote":
        state = remote.get("state", "queued")
        if state == "running":
            eta = remote.get("eta_seconds")
            eta_text = f", about {_duration(eta)} left" if isinstance(eta, (int, float)) else ""
            return f"4K60 upscale {remote.get('stage') or 'running'} {100 * (remote.get('progress') or 0):.0f}%{eta_text}."
        if state == "waiting_gpu":
            return "4K60 upscale waiting for the host GPU to be free."
        return "4K60 upscale queued on the host."
    if status == "downloading":
        return "Downloading and verifying the 4K60 video."
    if status == "cancelling":
        return "Cancelling the 4K60 upscale."
    if status == "complete":
        return "4K60 version ready."
    if status == "failed":
        return f"4K60 upscale failed: {record.get('error')} Retry or dismiss."
    if status == "cancelled":
        return "4K60 upscale cancelled. Retry or dismiss."
    return "4K60 upscale dismissed."


def _duration(seconds: float) -> str:
    seconds = int(seconds)
    return f"{seconds // 3600}h {seconds % 3600 // 60:02d}m" if seconds >= 3600 else f"{max(1, seconds // 60)} min"


def estimate_seconds(health: dict, preset: str, duration: float) -> float | None:
    preset = preset or health.get("default_preset") or ""
    for entry in health.get("presets", []):
        if entry.get("id") == preset and isinstance(entry.get("seconds_per_output_frame"), (int, float)):
            return entry["seconds_per_output_frame"] * duration * FPS
    return None


def needs_work(record: dict) -> bool:
    return record["status"] in ACTIVE or (
        record["status"] in ("complete", "dismissed") and not record.get("host_cleared"))


def tick(settings: Settings, *, client: Client | None = None,
         progress: Callable[[dict], None] | None = None) -> dict:
    """Advance every unfinished record once. Returns records, new alerts and a host note."""
    pending = [r for r in records() if needs_work(r)]
    result = {"records": [], "alerts": [], "note": ""}
    if pending and settings.upscale_url:
        own = client is None
        try:
            client = client or Client.from_settings(settings)
            health = None
            for record in pending:
                before = record["status"]
                try:
                    if record["status"] == "pending" and health is None:
                        health = client.health()
                    advance(record, client, health=health, progress=progress)
                except Unreachable as error:
                    result["note"] = str(error)
                    break
                except UpscaleError as error:
                    if record["status"] in ACTIVE and record["status"] != "pending":
                        result["note"] = str(error)
                    else:
                        _fail(record, str(error))
                if record["status"] == "failed" and before != "failed":
                    _notify(record, result)
        except UpscaleError as error:
            result["note"] = str(error)
        finally:
            if own and client is not None:
                client.close()
    elif pending:
        result["note"] = "4K60 upscales are waiting: no upscale service is configured."
    result["records"] = records()
    return result


def _notify(record: dict, result: dict) -> None:
    if record.get("alerted"):
        return
    text = f"simpleGen · 4K60 upscale failed\n{record.get('concept', '')[:60]}\n{record.get('error')}"
    result["alerts"].append({"key": f"upscale:{record['id']}", "message": text})
    from .cost_safety import _post, webhook_url
    url = webhook_url()
    if url:
        try:
            _post(url, text)
        except Exception as error:  # noqa: BLE001 -- alert delivery is best effort
            log.warning("Upscale alert webhook failed: %s", type(error).__name__)
    record["alerted"] = True
    save(record)


def host_summary(health: dict) -> str:
    gpu = health.get("gpu") or {}
    presets = [p for p in health.get("presets", []) if p.get("ready")]
    jobs = health.get("jobs", [])
    held = sum(int(j.get("bytes") or 0) for j in jobs)
    default = health.get("default_preset") or "not chosen yet"
    return (f"{gpu.get('name', 'GPU unknown')} · {int(gpu.get('vram_free_mb') or 0) / 1024:.1f} GB VRAM free · "
            f"{int(health.get('disk_free_bytes') or 0) / 1024**3:.0f} GB disk free\n"
            f"Ready presets: {', '.join(p['id'] for p in presets) or 'none'} (default: {default})\n"
            f"Jobs held on host: {len(jobs)} ({held / 1024**3:.2f} GB)")


def clearable_host_jobs(health: dict) -> list[str]:
    """Host jobs no unfinished local record needs (and not running)."""
    keep = {r["id"] for r in records() if r["status"] in ACTIVE or r["status"] in FLAGGED}
    return [j["id"] for j in health.get("jobs", [])
            if j["id"] not in keep and j.get("state") not in ("running", "waiting_gpu", "queued")]


def clear_host(settings: Settings, job_ids: list[str], *, client: Client | None = None) -> int:
    own = client is None
    client = client or Client.from_settings(settings)
    try:
        cleared = 0
        for job_id in job_ids:
            if client.delete(job_id):
                cleared += 1
        return cleared
    finally:
        if own:
            client.close()


# Preflight (advisory: the render never waits for the upscale host)


def preflight_check(settings: Settings, *, client: Client | None = None) -> dict:
    def check(status: str, message: str, **details) -> dict:
        return {"id": "upscale_service", "status": status, "message": message, "details": details,
                "advisory": True}

    if not settings.upscale_url.strip():
        return check("fail", "4K60 is on but no upscale service is set in Settings. The render will run; "
                             "its upscale waits until a service is added.")
    own = client is None
    try:
        client = client or Client.from_settings(settings, timeout=5.0)
        health = client.health()
    except Unreachable as error:
        return check("unverified", f"{error} The render will run; its 4K60 upscale waits for the host.")
    except UpscaleError as error:
        return check("fail", f"{error} The render will run; its 4K60 upscale waits until this is fixed.")
    finally:
        if own and client is not None:
            client.close()
    preset = settings.upscale_preset or health.get("default_preset") or ""
    ready = {p.get("id"): p for p in health.get("presets", []) if p.get("ready")}
    if not preset:
        return check("unverified", "The upscale host has no default preset yet; the 4K60 upscale will wait.")
    if preset not in ready:
        return check("fail", f"Upscale preset '{preset}' is not ready on the host.", ready=sorted(ready))
    estimate = estimate_seconds(health, preset, settings.duration)
    eta = f" Estimated {_duration(estimate)} per video." if estimate else ""
    return check("pass", f"Upscale service ready ({preset}).{eta}", preset=preset)
