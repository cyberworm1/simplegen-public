"""Read-only ComfyUI observations used inside the disposable session manager."""

from __future__ import annotations

import httpx

from .outputs import video_outputs
from .provider import identifier


class ComfyMonitor:
    def __init__(
        self,
        pod_id: str,
        username: str,
        password: str,
        required_nodes: list[str],
        client: httpx.Client | None = None,
        extra_known=None,
    ):
        """``extra_known``: prompt ids that are not this session's jobs but are expected
        on the pod (inherited at a warm-pod adoption). Default: none, as before."""
        self.base = f"https://{identifier(pod_id)}-8189.proxy.runpod.net"
        self.client = client or httpx.Client(timeout=10, follow_redirects=False)
        self.client.headers["User-Agent"] = "LocalCloudH3/0.1"
        self.auth = httpx.BasicAuth(username, password)
        self.required_nodes = required_nodes
        self.extra_known = frozenset(str(x) for x in (extra_known or ()))
        self.checked = False

    def read(self, path: str):
        response = self.client.get(self.base + path, auth=self.auth)
        response.raise_for_status()
        return response.json()

    def poll(self, jobs: dict) -> dict:
        observation = {
            "ready": False,
            "healthy": False,
            "auth_verified": self.checked,
            "queue_empty": None,
            "jobs": {},
            "outputs": {},
            "fatal": False,
        }
        try:
            if not self.checked:
                # Validate the gateway really rejects unauthenticated API requests.
                unauth = self.client.get(self.base + "/system_stats", auth=None)
                if unauth.status_code not in (401, 403):
                    if unauth.status_code == 200:
                        observation["fatal"] = True
                    return observation
                stats = self.read("/system_stats")
                if not any(d.get("type") == "cuda" for d in stats.get("devices", [])):
                    observation["fatal"] = True
                    return observation
                nodes = self.read("/object_info")
                if not isinstance(nodes, dict) or any(
                    n not in nodes for n in self.required_nodes
                ):
                    observation["fatal"] = True
                    return observation
                self.checked = True
            queue = self.read("/queue")
            if not all(
                isinstance(queue.get(k), list)
                for k in ("queue_running", "queue_pending")
            ):
                return observation
            entries = queue["queue_running"] + queue["queue_pending"]
            if any(not isinstance(x, list) or len(x) < 2 for x in entries):
                return observation
            queued = {str(x[1]) for x in entries}
            if all(job["prompt_id"] for job in jobs.values()):
                known = {job["prompt_id"] for job in jobs.values()} | self.extra_known
                histories = self.read("/history?max_items=1000")
                if not isinstance(histories, dict):
                    return observation
                observation["unregistered_jobs"] = bool(
                    (queued | set(histories)) - known
                )
            for job in jobs.values():
                pid = job["prompt_id"]
                if not pid:
                    continue
                history = self.read("/history/" + identifier(pid))
                entry = history.get(pid, {})
                status = entry.get("status", {})
                if status.get("status_str") == "error":
                    result = "error"
                elif (
                    status.get("completed") is True
                    and status.get("status_str") == "success"
                ):
                    result = "success"
                else:
                    result = "running" if pid in queued else "missing"
                observation["jobs"][pid] = result
                if result == "success":
                    observation["outputs"][pid] = video_outputs(pid, entry)
            # Read queue again AFTER histories so idle evidence is as recent as possible.
            queue = self.read("/queue")
            if not all(
                isinstance(queue.get(k), list)
                for k in ("queue_running", "queue_pending")
            ):
                return observation
            observation.update(
                ready=True,
                healthy=True,
                auth_verified=True,
                queue_empty=not queue["queue_running"] and not queue["queue_pending"],
            )
        except (httpx.HTTPError, ValueError, TypeError, KeyError, AttributeError):
            # Unknown state never counts as idle. Next scheduled poll retries all reads.
            pass
        return observation
