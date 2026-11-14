"""Narrow RunPod REST v2 adapter: no volume update/delete, stop/start, or reset methods.

Callers keep reading REST v1 field names (desiredStatus, costPerHr,
networkVolumeId, machine.dataCenterId). v2 data is converted to that view at
this boundary, so saved sessions and their callers are unchanged.
Spec: https://api.runpod.io/v2/openapi.json
"""

from __future__ import annotations

import math
import re
import time
from typing import Any
from urllib.parse import urlencode

import httpx

GPU_STOCK_RANK = {"LOW": 1, "MEDIUM": 2, "HIGH": 3}
BASE_URL = "https://api.runpod.io/v2"
PAGE_LIMIT = 100
MAX_RETRY_AFTER_SECONDS = 30


def available_gpu_centers(centers: object) -> list[dict[str, str]]:
    """Unique available locations, with North America first for display only."""
    by_id: dict[str, dict[str, str]] = {}
    for center in centers if isinstance(centers, list) else []:
        if not isinstance(center, dict):
            continue
        location = center.get("id")
        level = center.get("availability")
        if (not isinstance(location, str) or not location.strip()
                or level not in GPU_STOCK_RANK):
            continue
        location = location.strip()
        if GPU_STOCK_RANK[level] > GPU_STOCK_RANK.get(by_id.get(location, {}).get("availability"), 0):
            by_id[location] = {"id": location, "availability": level}
    return sorted(by_id.values(), key=lambda c: (
        not c["id"].startswith(("US-", "CA-")), -GPU_STOCK_RANK[c["availability"]], c["id"]))


class ProviderError(RuntimeError):
    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


def finite_positive(value: Any) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError("Expected a finite positive number")
    return number


def identifier(value: str) -> str:
    if not re.fullmatch(r"[a-zA-Z0-9_-]{1,100}", value):
        raise ValueError("Invalid provider identifier")
    return value


def gpu_is_stopped(pod: dict) -> bool:
    """Recognize the explicit stop state; unknown state is never a stop.

    pod_from_v2 maps v2 status EXITED to desiredStatus EXITED. EXITED records
    the provider's stopped state, not who requested it or proof of deletion.
    ERROR, unknown values, missing runtime/ports and startup failures prove
    nothing. GET 404 is handled separately by RunPod.get; TERMINATED is not
    absence proof.
    """
    return isinstance(pod, dict) and pod.get("desiredStatus") == "EXITED"


# v1 desiredStatus was the requested state, so a starting pod read RUNNING.
# ERROR and any unknown status keep their own value: never stopped, never running.
_V1_STATUS = {"PROVISIONING": "RUNNING", "STARTING": "RUNNING", "RUNNING": "RUNNING",
              "EXITED": "EXITED", "TERMINATED": "TERMINATED"}


def pod_from_v2(pod: Any) -> dict:
    """Add the v1 names callers read; every v2 field is kept unchanged.

    v2 reports cost 0.0 for EXITED/TERMINATED pods and may omit it; a zero or
    missing cost is unknown, never free, so costPerHr is left out.
    """
    if not isinstance(pod, dict) or not isinstance(pod.get("id"), str):
        raise ProviderError("Unexpected RunPod pod schema")
    view = dict(pod)
    status = pod.get("status")
    view["desiredStatus"] = _V1_STATUS.get(status, status if isinstance(status, str) else "UNKNOWN")
    view.pop("costPerHr", None)
    cost = pod.get("cost")
    if (type(cost) in (int, float) and math.isfinite(cost) and cost > 0):
        view["costPerHr"] = float(cost)
    mounts = pod.get("mounts")
    network = mounts.get("network") if isinstance(mounts, dict) else None
    if isinstance(network, list) and network and isinstance(network[0], dict):
        volume = network[0].get("volumeId")
        if isinstance(volume, str):
            view["networkVolumeId"] = volume
    if isinstance(pod.get("dataCenterId"), str):
        view["machine"] = {"dataCenterId": pod["dataCenterId"]}
    return view


def volume_from_v2(volume: Any) -> dict:
    """v2 names the volume region dataCenter; callers read v1's dataCenterId."""
    if not isinstance(volume, dict):
        raise ProviderError("Unexpected RunPod network-volume schema")
    view = dict(volume)
    if isinstance(volume.get("dataCenter"), str):
        view["dataCenterId"] = volume["dataCenter"]
    return view


def create_body_from_v1(payload: dict) -> dict:
    """Translate the saved v1-shaped create payload into a v2 CreatePodRequest.

    Saved session policies keep their v1 names. A field without a known v2
    meaning is refused before any request, so nothing is created.
    """
    body: dict[str, Any] = {}
    rest = dict(payload)
    rename = {"name": "name", "imageName": "image", "dockerStartCmd": "cmd",
              "containerDiskInGb": "disk", "ports": "ports", "env": "env",
              "cloudType": "cloud", "dataCenterIds": "dataCenterIds"}
    for old, new in rename.items():
        if old in rest:
            body[new] = rest.pop(old)
    # v2 pods are on-demand, unlocked and have no public-IP option.
    for flag in ("interruptible", "locked", "supportPublicIp"):
        if rest.pop(flag, False) is not False:
            raise ProviderError(f"RunPod v2 cannot honour {flag}; nothing was created", 422)
    if rest.pop("volumeInGb", 0) != 0:
        raise ProviderError("Host-local pod volumes are not used; nothing was created", 422)
    volume, path = rest.pop("networkVolumeId", None), rest.pop("volumeMountPath", None)
    if volume is not None:
        if not isinstance(path, str) or not path:
            raise ProviderError("A network volume needs a mount path; nothing was created", 422)
        body["mounts"] = {"network": [{"volumeId": volume, "path": path}]}
    kind = rest.pop("computeType", None)
    if kind == "GPU":
        ids = rest.pop("gpuTypeIds", None)
        if not isinstance(ids, list) or len(ids) != 1:
            raise ProviderError("Choose exactly one GPU type; nothing was created", 422)
        gpu = {"id": ids[0], "count": rest.pop("gpuCount", 1)}
        for old, new in (("minRAMPerGPU", "minRamPerGpu"), ("minVCPUPerGPU", "minVcpuCountPerGpu"),
                         ("allowedCudaVersions", "allowedCudaVersions"),
                         ("minCudaVersion", "minCudaVersion")):
            if old in rest:
                gpu[new] = rest.pop(old)
        body["gpu"] = gpu
    elif kind == "CPU":
        ids = rest.pop("cpuFlavorIds", None)
        if not isinstance(ids, list) or len(ids) != 1:
            raise ProviderError("Choose exactly one CPU flavor; nothing was created", 422)
        body["cpu"] = {"id": ids[0], "vcpuCount": rest.pop("vcpuCount", None)}
    else:
        raise ProviderError("Pod compute type must be GPU or CPU; nothing was created", 422)
    if rest:
        raise ProviderError("Unsupported pod settings for RunPod v2: "
                            + ", ".join(sorted(rest)) + "; nothing was created", 422)
    return body


def retry_after(response: httpx.Response | None, attempt: int) -> float:
    """Honour Retry-After on a 429, capped so a read never stalls the app for long."""
    value = response.headers.get("Retry-After") if response is not None else None
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return 0.5 * 2**attempt
    return min(max(seconds, 0.0), MAX_RETRY_AFTER_SECONDS)


def all_pods(get_page) -> list[dict]:
    """Read every page of GET /v2/pods; a missed page could hide a pod we pay for."""
    pods: list[dict] = []
    cursor, seen = None, set()
    while True:
        page = get_page({"limit": PAGE_LIMIT, **({"cursor": cursor} if cursor else {})})
        items = page.get("pods") if isinstance(page, dict) else None
        pagination = page.get("pagination") if isinstance(page, dict) else None
        if (not isinstance(items, list) or not isinstance(pagination, dict)
                or type(pagination.get("hasNextPage")) is not bool):
            raise ProviderError("Unexpected RunPod pod-list schema")
        for item in items:
            pod = pod_from_v2(item)
            if not isinstance(pod.get("name"), str):
                raise ProviderError("Unexpected RunPod pod-list schema")
            pods.append(pod)
        if not pagination["hasNextPage"]:
            return pods
        cursor = pagination.get("nextCursor")
        if not isinstance(cursor, str) or not cursor or cursor in seen:
            raise ProviderError("Unexpected RunPod pod-list pagination")
        seen.add(cursor)


class RunPod:
    def __init__(self, api_key: str, client: httpx.Client | None = None):
        self.client = client or httpx.Client(timeout=20, follow_redirects=False)
        self.headers = {"Authorization": f"Bearer {api_key}"}

    def _request(self, method: str, path: str, body=None, *, catalog=False):
        # Every call is REST v2 now; ``catalog`` is accepted for existing callers.
        # Only reads retry automatically. In particular POST is attempted once.
        attempts = 3 if method == "GET" else 1
        for attempt in range(attempts):
            response = None
            try:
                response = self.client.request(
                    method, BASE_URL + path, headers=self.headers, json=body
                )
                if response.status_code >= 300:
                    raise ProviderError(
                        f"RunPod HTTP {response.status_code}", response.status_code
                    )
                return response.json() if response.content else None
            except (httpx.HTTPError, ValueError, ProviderError) as exc:
                code = exc.status if isinstance(exc, ProviderError) else None
                if attempt + 1 < attempts and (
                    code is None or code in (408, 429) or code >= 500
                ):
                    time.sleep(retry_after(response if code == 429 else None, attempt))
                    continue
                # Never surface response bodies, payloads, credentials or URLs.
                raise ProviderError(
                    f"RunPod request failed ({code or 'transport/JSON'})", code
                ) from None

    def create(self, payload: dict) -> dict:
        """One POST of a v1-shaped payload. Errors keep v1's meaning for callers:
        4xx other than 408/429 means nothing was created; anything else is uncertain."""
        result = self._request("POST", "/pods", create_body_from_v1(payload))
        # Without an id the caller records the create as uncertain and reconciles.
        if isinstance(result, dict) and isinstance(result.get("id"), str):
            return pod_from_v2(result)
        return result

    def get(self, pod_id: str) -> dict | None:
        try:
            result = self._request("GET", "/pods/" + identifier(pod_id))
            if not isinstance(result, dict) or result.get("id") != pod_id:
                raise ProviderError("Unexpected RunPod pod-detail schema")
            return pod_from_v2(result)
        except ProviderError as exc:
            if exc.status == 404:
                return None
            raise

    def list(self) -> list[dict]:
        return all_pods(lambda params: self._request("GET", "/pods?" + urlencode(params)))

    def delete(self, pod_id: str) -> None:
        try:
            self._request("DELETE", "/pods/" + identifier(pod_id))
        except ProviderError as exc:
            if exc.status != 404:
                raise

    def volume(self, volume_id: str) -> dict:
        return volume_from_v2(self._request("GET", "/network-volumes/" + identifier(volume_id)))

    def quote(self, payload: dict) -> float:
        """Fresh catalog compute rate, NOT a capacity reservation or complete bill."""
        if payload["computeType"] == "CPU":
            catalog = self._request("GET", "/catalog/cpus", catalog=True)
            flavor = next(
                x for x in catalog["cpus"] if x["id"] == payload["cpuFlavorIds"][0]
            )
            return (
                finite_positive(flavor["price"]["securePerVcpu"]) * payload["vcpuCount"]
            )
        catalog = self._request(
            "GET",
            "/catalog/gpus?include=AVAILABILITY&product=POD&cloud=SECURE",
            catalog=True,
        )
        gpu = next(x for x in catalog["gpus"] if x["id"] == payload["gpuTypeIds"][0])
        return finite_positive(gpu["price"]["secure"]) * payload["gpuCount"]
