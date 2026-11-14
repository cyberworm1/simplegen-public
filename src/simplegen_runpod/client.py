"""Blocking desktop integration; run off the Qt UI thread. No retries of prompt POST."""

from __future__ import annotations

from urllib.parse import urlsplit

import httpx

from .control import session_id
from .provider import identifier


class Client:
    def __init__(self, url: str, token: str, client: httpx.Client | None = None):
        parts = urlsplit(url)
        _ = parts.port  # Reject malformed/out-of-range ports before any request.
        if (
            parts.scheme != "https"
            or not parts.hostname
            or parts.username
            or parts.password
            or parts.query
            or parts.fragment
            or parts.path not in ("", "/")
        ):
            raise ValueError("Use an HTTPS manager origin without embedded credentials")
        self.url = url.rstrip("/")
        self.headers = {"Authorization": f"Bearer {token}"}
        self.client = client or httpx.Client(timeout=90, follow_redirects=False)

    def close(self):
        self.client.close()

    def _call(self, method, sid, suffix="", body=None):
        try:
            response = self.client.request(
                method,
                self.url + "/v2/sessions/" + session_id(sid) + suffix,
                headers=self.headers,
                json=body,
                follow_redirects=False,
            )
            response.raise_for_status()
            return response.json()
        except (httpx.HTTPError, ValueError):
            raise RuntimeError(
                "Manager unavailable or request rejected; reconnect to the same session"
            ) from None

    def start(self, sid):
        return self._call("PUT", sid)

    def status(self, sid):
        return self._call("GET", sid)

    def begin_job(self, sid, job_id):
        return self._call("PUT", sid, "/jobs/" + identifier(job_id))

    def bind_prompt(self, sid, job_id, prompt_id):
        return self._call(
            "PUT",
            sid,
            "/jobs/" + identifier(job_id) + "/prompt",
            {"prompt_id": prompt_id},
        )

    def complete(self, sid, *, downloads_verified, manifest_sha256, receipts):
        return self._call(
            "POST",
            sid,
            "/complete",
            {
                "downloads_verified": downloads_verified,
                "manifest_sha256": manifest_sha256,
                "receipts": receipts,
            },
        )

    def cancel(self, sid, *, request_id):
        return self._call("POST", sid, "/cancel", {"request_id": request_id})
