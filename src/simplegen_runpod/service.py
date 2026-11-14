"""Session API and monitor on the same disposable CPU. One process only."""

from __future__ import annotations

import hmac
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, ConfigDict, StrictBool, StrictInt

from .control import Conflict
from .provider import ProviderError


class Body(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Receipt(Body):
    id: str
    sha256: str
    size_bytes: StrictInt


class Completion(Body):
    downloads_verified: StrictBool
    manifest_sha256: str
    receipts: list[Receipt]


class Prompt(Body):
    prompt_id: str


class Cancel(Body):
    request_id: str


def create_app(controller, *, background=True):
    stop = threading.Event()

    def monitor():
        while not stop.is_set():
            try:
                controller.tick()
            except Exception:  # noqa: BLE001 -- keep monitoring alive without leaking provider secrets
                # Do not disclose provider payloads or secrets. Failure NEVER cancels compute.
                print(
                    "manager_tick_failed; compute_retained; manual_attention_may_be_required",
                    flush=True,
                )
            stop.wait(controller.config.poll_seconds)

    @asynccontextmanager
    async def lifespan(app):
        thread = None
        if background:
            thread = threading.Thread(
                target=monitor, daemon=True, name="session-monitor"
            )
            thread.start()
        yield
        stop.set()
        if thread:
            thread.join(timeout=1)
        # Process exit, SIGTERM and client disconnect are not cancellation requests.

    app = FastAPI(
        title="simpleGen disposable session manager",
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )

    @app.middleware("http")
    async def limit_body(request: Request, call_next):
        from starlette.responses import JSONResponse

        if request.headers.get("transfer-encoding"):
            return JSONResponse({"error": "streaming request bodies unsupported"}, 413)
        try:
            length = int(request.headers.get("content-length", "0"))
        except ValueError:
            return JSONResponse({"error": "invalid length"}, 400)
        if not 0 <= length <= 1_000_000:
            return JSONResponse({"error": "request too large"}, 413)
        return await call_next(request)

    def auth(sid, authorization):
        if not hmac.compare_digest(authorization or "", "Bearer " + controller.secret):
            raise HTTPException(401, "Unauthorized")
        if sid != controller.sid:
            raise HTTPException(404, "Unknown session")

    def invoke(fn, *args):
        try:
            return fn(*args)
        except KeyError:
            raise HTTPException(404, "Session or job not found") from None
        except Conflict as exc:
            raise HTTPException(409, str(exc)) from None
        except ValueError:
            raise HTTPException(400, "Invalid request") from None
        except (ProviderError, RuntimeError, OSError):
            raise HTTPException(
                503, "Manager state/provider unavailable; retain session and reconcile"
            ) from None

    @app.get("/health")
    def health():
        if controller.clock() - controller.last_tick > 60:
            raise HTTPException(
                503, "Manager monitoring needs attention; compute is retained"
            )
        return {"monitor": "running"}

    @app.put("/v2/sessions/{sid}")
    def start(sid: str, authorization: str | None = Header(default=None)):
        auth(sid, authorization)
        return invoke(controller.start, sid)

    @app.get("/v2/sessions/{sid}")
    def status(sid: str, authorization: str | None = Header(default=None)):
        auth(sid, authorization)
        return invoke(controller.status, sid)

    @app.put("/v2/sessions/{sid}/jobs/{job_id}")
    def begin(sid: str, job_id: str, authorization: str | None = Header(default=None)):
        auth(sid, authorization)
        return invoke(controller.begin_job, sid, job_id)

    @app.put("/v2/sessions/{sid}/jobs/{job_id}/prompt")
    def bind(
        sid: str,
        job_id: str,
        body: Prompt,
        authorization: str | None = Header(default=None),
    ):
        auth(sid, authorization)
        return invoke(controller.bind_prompt, sid, job_id, body.prompt_id)

    @app.post("/v2/sessions/{sid}/complete")
    def complete(
        sid: str, body: Completion, authorization: str | None = Header(default=None)
    ):
        auth(sid, authorization)
        return invoke(
            controller.seal,
            sid,
            body.downloads_verified,
            body.manifest_sha256,
            [r.model_dump() for r in body.receipts],
        )

    @app.post("/v2/sessions/{sid}/cancel")
    def cancel(
        sid: str, body: Cancel, authorization: str | None = Header(default=None)
    ):
        auth(sid, authorization)
        return invoke(controller.cancel, sid, body.request_id)

    return app
