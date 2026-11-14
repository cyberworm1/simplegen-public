"""ComfyUI transport: scoped jobs, real events and verified local downloads."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import time
import uuid
from collections.abc import Callable
from pathlib import Path, PurePosixPath
from threading import Event
from urllib.parse import urlencode

import httpx
import websocket

from .agents import Cancelled
from .settings import Settings

VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}


class ComfyRequestError(RuntimeError):
    """A failed request, classified for read-only monitoring retries."""

    def __init__(self, message: str, *, retryable: bool):
        super().__init__(message)
        self.retryable = retryable


class Comfy:
    def __init__(self, settings: Settings, client: httpx.Client | None = None):
        self.settings = settings
        settings.validate(require_folders=False)
        self.url = settings.comfy_url.rstrip("/")
        token = os.environ.get(settings.token_env, "") if settings.token_env else ""
        self.headers = {"User-Agent": "LocalCloudH3/0.1"}
        if token:
            self.headers["Authorization"] = f"Bearer {token}"
        if settings.browser_login_file:
            try:
                login = json.loads(
                    Path(settings.browser_login_file).expanduser().read_text(encoding="utf-8")
                )
                username, password = login["username"], login["password"]
                if (
                    not isinstance(username, str)
                    or not isinstance(password, str)
                    or not username
                    or not password
                    or ":" in username
                ):
                    raise ValueError
            except (OSError, ValueError, KeyError, TypeError):
                raise ValueError(
                    "Cannot read browser login. Choose a JSON file containing username and password in Settings."
                ) from None
            encoded = base64.b64encode(f"{username}:{password}".encode()).decode("ascii")
            self.headers["Authorization"] = f"Basic {encoded}"
        self.client = client or httpx.Client(
            timeout=httpx.Timeout(30, connect=10), headers=self.headers, follow_redirects=False
        )
        self.client.headers.update(self.headers)

    def close(self) -> None:
        self.client.close()

    def request(self, method: str, path: str, **kwargs):
        try:
            response = self.client.request(method, self.url + path, **kwargs)
            response.raise_for_status()
            return response
        except httpx.HTTPStatusError as error:
            detail = error.response.text[:3000]
            status = error.response.status_code
            raise ComfyRequestError(
                f"ComfyUI {path}: HTTP {status}: {detail}",
                retryable=status in (408, 429) or 500 <= status < 600,
            ) from None
        except httpx.HTTPError as error:
            raise ComfyRequestError(
                f"ComfyUI {path}: connection failed ({type(error).__name__}).",
                retryable=True,
            ) from None

    def object_info(self) -> dict:
        return self.request("GET", "/object_info", timeout=90).json()

    def history(self, prompt_id: str) -> dict | None:
        return self.request("GET", f"/history/{prompt_id}").json().get(prompt_id)

    def queue(self) -> dict:
        return self.request("GET", "/queue").json()

    def find_submission(self, run_id: str) -> str | None:
        # Reconcile uncertain POST without ever resubmitting it.
        matches: set[str] = set()
        queue = self.queue()
        items = queue.get("queue_running", []) + queue.get("queue_pending", [])
        history = self.request("GET", "/history", params={"max_items": 1000}).json()
        items += [entry.get("prompt", []) for entry in history.values()]
        for item in items:
            if (
                len(item) > 3
                and isinstance(item[3], dict)
                and item[3].get("simplegen_run_id") == run_id
            ):
                matches.add(str(item[1]))
        if len(matches) > 1:
            raise RuntimeError("Multiple ComfyUI jobs match this run. Inspect its saved report.")
        return next(iter(matches), None)

    def submit(self, graph: dict, run_id: str, client_id: str) -> str:
        response = self.request(
            "POST",
            "/prompt",
            json={
                "prompt": graph,
                "client_id": client_id,
                "extra_data": {"simplegen_run_id": run_id},
            },
        ).json()
        if response.get("node_errors") or not response.get("prompt_id"):
            raise RuntimeError("ComfyUI rejected the workflow: " + json.dumps(response)[:4000])
        return str(response["prompt_id"])

    def upload(self, source: Path, run_id: str, cancel: Event) -> str:
        if cancel.is_set():
            raise Cancelled("Cancelled before uploading references.")
        # Distinct source paths with the same basename cannot collide.
        digest = hashlib.sha256(str(source.resolve()).encode()).hexdigest()[:12]
        name = f"{digest}_{source.name}"
        with source.open("rb") as handle:
            response = self.request(
                "POST",
                "/upload/image",
                timeout=180,
                data={"subfolder": f"simplegen/{run_id}", "overwrite": "false"},
                files={"image": (name, handle, "application/octet-stream")},
            ).json()
        if not response.get("name"):
            raise RuntimeError(f"ComfyUI did not accept reference {source.name}.")
        return str(PurePosixPath(response.get("subfolder", "")) / response["name"])

    def cancel_job(self, prompt_id: str) -> str:
        self.request("POST", "/queue", json={"delete": [prompt_id]})
        stats = self.request("GET", "/system_stats").json()
        version = stats.get("system", {}).get("comfyui_version", "")
        match = re.match(r"(\d+)\.(\d+)", version)
        # Older servers may silently treat a targeted interrupt as a global one.
        if match and tuple(map(int, match.groups())) >= (0, 36):
            self.request("POST", "/interrupt", json={"prompt_id": prompt_id})
            return (
                "Cancellation requested for this ComfyUI job. Cloud instance billing can continue."
            )
        return (
            "Removed this job if queued; monitoring stopped. This server has no verified targeted "
            "interrupt support: a running job may continue. Cloud instance billing can continue."
        )

    def monitor(
        self,
        prompt_id: str,
        client_id: str,
        graph: dict,
        cancel: Event,
        on_event: Callable[[dict], None],
        *,
        detach: Event | None = None,
        cancel_remote: bool = True,
    ) -> dict:
        started = time.monotonic()
        last_poll = 0.0
        last_retry = -10.0
        socket = None
        completed: set[str] = set()
        current = None
        status_failures = 0
        last_queue_check = started
        try:
            while True:
                if detach is not None and detach.is_set():
                    from .lifecycle import Detached
                    raise Detached("Detached from saved generation; Resume reconnects without resubmitting.")
                if cancel.is_set():
                    if not cancel_remote:
                        raise Cancelled("Manual shutdown requested.")
                    try:
                        message = self.cancel_job(prompt_id)
                    except RuntimeError:
                        message = "Monitoring cancelled; server cancellation could not be confirmed. GPU billing may continue."
                    raise Cancelled(message)
                now = time.monotonic()
                if now - started > self.settings.generation_timeout:
                    raise TimeoutError(
                        "Monitoring time limit reached. The server job may still run; Resume reconnects without resubmitting."
                    )
                if now - last_poll >= 2:
                    last_poll = now
                    missing = False
                    try:
                        entry = self.history(prompt_id)
                        if not entry and now - last_queue_check > 20:
                            queue = self.queue()
                            queued = queue.get("queue_running", []) + queue.get("queue_pending", [])
                            if not any(
                                len(item) > 1 and str(item[1]) == prompt_id for item in queued
                            ):
                                # A job can finish between history and queue reads.
                                entry = self.history(prompt_id)
                                missing = entry is None
                            # Advance only after every reconciliation read succeeds.
                            # Otherwise the next poll retries the failed queue/history
                            # phase rather than resetting its failure budget.
                            last_queue_check = now
                    except RuntimeError as error:
                        if cancel.is_set():
                            continue
                        if isinstance(error, ComfyRequestError) and not error.retryable:
                            raise
                        status_failures += 1
                        if status_failures >= 5:
                            raise RuntimeError(
                                "Connection lost. Job ID is saved; use Resume to reconnect. Server generation may continue."
                            ) from None
                        entry = None
                        on_event(
                            {
                                "stage": "Generating",
                                "message": f"Status connection interrupted · retrying ({status_failures}/5); existing job remains saved",
                                "progress": None,
                            }
                        )
                    else:
                        status_failures = 0
                        if missing:
                            raise RuntimeError(
                                "Saved job is absent from ComfyUI queue and history. It may have been cancelled or lost on restart. No new render was submitted."
                            )
                    if entry:
                        status = entry.get("status", {})
                        if status.get("status_str") == "error":
                            raise RuntimeError(
                                "Generation failed: "
                                + json.dumps(status.get("messages", []))[:4000]
                            )
                        if status.get("completed") or status.get("status_str") == "success":
                            return entry
                if cancel.is_set():
                    continue
                if socket is None and now - last_retry > 10:
                    last_retry = now
                    ws_url = ("wss" if self.url.startswith("https") else "ws") + self.url[
                        self.url.index(":") :
                    ]
                    try:
                        socket = websocket.create_connection(
                            ws_url + "/ws?" + urlencode({"clientId": client_id}),
                            timeout=1,
                            header=self.headers,
                            suppress_origin=True,
                        )
                    except (OSError, websocket.WebSocketException):
                        on_event(
                            {
                                "stage": "Generating",
                                "message": "Waiting for ComfyUI · live steps unavailable, checking saved job status",
                                "progress": None,
                            }
                        )
                if socket is None:
                    cancel.wait(0.25)
                    continue
                try:
                    raw = socket.recv()
                    if not raw:
                        socket.close()
                        socket = None
                        continue
                    if not isinstance(raw, str):
                        continue
                    event = json.loads(raw)
                except websocket.WebSocketTimeoutException:
                    continue
                except (OSError, ValueError, websocket.WebSocketException):
                    socket.close()
                    socket = None
                    continue
                data = event.get("data", {})
                if data.get("prompt_id") != prompt_id:
                    continue
                kind = event.get("type")
                if kind in ("execution_error", "execution_interrupted"):
                    raise RuntimeError(
                        "ComfyUI stopped: " + str(data.get("exception_message", kind))
                    )
                if kind == "execution_cached":
                    completed.update(str(n) for n in data.get("nodes", []))
                if kind == "executing":
                    if current:
                        completed.add(current)
                    current = str(data["node"]) if data.get("node") is not None else None
                if kind == "executed" and data.get("node") is not None:
                    completed.add(str(data["node"]))
                if kind in ("executing", "executed", "progress", "execution_cached"):
                    message = f"{len(completed)} / {len(graph)} workflow nodes complete"
                    fraction = 0.0
                    if kind == "progress" and data.get("max", 0) > 0:
                        fraction = min(1, max(0, data.get("value", 0) / data["max"]))
                        message += f" · step {data.get('value', 0)}/{data['max']}"
                    progress = min(99.0, 100 * (len(completed) + fraction) / max(1, len(graph)))
                    on_event(
                        {
                            "stage": "Generating",
                            "message": message + f" · {int(now - started)}s",
                            "progress": progress,
                        }
                    )
        finally:
            if socket:
                socket.close()

    def download(
        self,
        descriptor: dict,
        target: Path,
        cancel: Event,
        on_bytes: Callable[[int, int | None], None] | None = None,
    ) -> dict:
        if cancel.is_set():
            raise Cancelled("Download paused; use Resume to finish saving the package.")
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_suffix(target.suffix + ".part")
        from .global_session import is_global
        size_only = is_global(self.settings)
        digest = None if size_only else hashlib.sha256()
        size = 0
        params = {
            key: descriptor.get(key, default)
            for key, default in (("filename", ""), ("subfolder", ""), ("type", "output"))
        }
        try:
            with self.client.stream(
                "GET", self.url + "/view", params=params, timeout=60
            ) as response:
                response.raise_for_status()
                total = (
                    int(response.headers["content-length"])
                    if response.headers.get("content-length", "").isdigit()
                    else None
                )
                with partial.open("wb") as handle:
                    for chunk in response.iter_bytes(1024 * 1024):
                        if cancel.is_set():
                            raise Cancelled(
                                "Download paused; use Resume to finish saving the package."
                            )
                        handle.write(chunk)
                        if digest is not None:
                            digest.update(chunk)
                        size += len(chunk)
                        if on_bytes:
                            on_bytes(size, total)
                    handle.flush()
                    os.fsync(handle.fileno())
                if size == 0 or (size_only and total is None) or (total is not None and size != total):
                    raise RuntimeError("Incomplete download; use Resume to retry it.")
                partial.replace(target)
                directory_fd = os.open(target.parent, os.O_RDONLY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
        except httpx.HTTPError:
            raise RuntimeError("Output download interrupted; use Resume to retry it.") from None
        if size_only:
            return {"path": str(target), "size_bytes": size, "verification": "size"}
        return {"path": str(target), "size_bytes": size, "sha256": digest.hexdigest()}


def output_files(entry: dict) -> list[dict]:
    files: list[dict] = []
    seen: set[tuple] = set()

    def walk(value, node_id):
        if isinstance(value, dict):
            if isinstance(value.get("filename"), str):
                name = value["filename"]
                subfolder = value.get("subfolder", "")
                if (
                    not name
                    or PurePosixPath(name).name != name
                    or "\\" in name
                    or ".." in PurePosixPath(subfolder).parts
                    or PurePosixPath(subfolder).is_absolute()
                ):
                    raise ValueError("Server returned an unsafe output path.")
                if value.get("type", "output") not in ("output", "temp"):
                    return
                key = (node_id, name, subfolder, value.get("type", "output"))
                if key not in seen:
                    seen.add(key)
                    files.append(
                        {
                            "node_id": node_id,
                            "filename": name,
                            "subfolder": subfolder,
                            "type": value.get("type", "output"),
                        }
                    )
            else:
                for nested in value.values():
                    walk(nested, node_id)
        elif isinstance(value, list):
            for nested in value:
                walk(nested, node_id)

    for node_id, value in entry.get("outputs", {}).items():
        walk(value, str(node_id))
    return files


def new_id() -> str:
    return uuid.uuid4().hex
