"""Global Volume GPU deployment contract; no lifecycle calls on import.

The caller must durably save a unique owned name and creation intent before
calling deploy. A transport failure has an uncertain outcome, never a retry.

Create stays on GraphQL: REST v2 has no Global Volume (object-store) mount, and
GraphQL works until its January 2027 sunset. Find, get and delete use REST v2.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import urlencode

import httpx

from .provider import BASE_URL, ProviderError, all_pods, finite_positive, identifier, pod_from_v2

MOUNT_PATH = "/workspace-global"
GRAPHQL_URL = "https://api.runpod.io/graphql"
DEPLOY_MUTATION = (
    "mutation($input: PodFindAndDeployOnDemandInput) { "
    "podFindAndDeployOnDemand(input: $input) { id name costPerHr } }"
)


class CreateUncertain(ProviderError):
    """A deployment may exist; reconcile the exact owned name before proceeding."""


@dataclass(frozen=True)
class GlobalGPUProfile:
    # The user's own RunPod template (ComfyUI with the H3 nodes) and Global Volume (models).
    template_id: str = ""
    volume_id: str = ""
    mount_path: str = MOUNT_PATH
    container_disk_gb: int = 200
    min_cuda_version: str = "13.0"
    cloud_type: str = "SECURE"
    port: str = "8189/http"

    def validate(self) -> None:
        for value in (self.template_id, self.volume_id):
            if not isinstance(value, str) or not re.fullmatch(r"[a-zA-Z0-9_-]{1,100}", value):
                raise ValueError("Global GPU profile needs your RunPod template ID and Global Volume ID")
        if (
            self.mount_path != MOUNT_PATH
            # Saved sessions can still carry the earlier 150 GB profile.
            # Their frozen deployment intent must remain recoverable.
            or self.container_disk_gb not in (150, 200)
            or self.min_cuda_version != "13.0"
            or self.cloud_type != "SECURE"
            or self.port != "8189/http"
        ):
            raise ValueError("Global GPU profile does not match the supported deployment settings")

    def deployment_input(self, *, name: str, gpu_type_id: str,
                         data_center_id: str | None = None) -> dict:
        self.validate()
        if not isinstance(gpu_type_id, str) or not gpu_type_id.strip() or len(gpu_type_id) > 200:
            raise ValueError("Select a valid GPU type")
        payload = {
            "name": identifier(name),
            "templateId": self.template_id,
            "gpuTypeId": gpu_type_id,
            "gpuCount": 1,
            "minCudaVersion": self.min_cuda_version,
            "volumeMounts": [{
                "volumeId": self.volume_id,
                "volumeType": "OBJECT_STORE_VOLUME",
                "mountPath": self.mount_path,
            }],
            "containerDiskInGb": self.container_disk_gb,
            "volumeInGb": 0,
            "cloudType": self.cloud_type,
        }
        if data_center_id is not None:
            payload["dataCenterId"] = identifier(data_center_id)
        return payload


def load_global_profile(path) -> GlobalGPUProfile:
    """Validate a separate desktop-only profile; never reinterpret schema-v2 state."""
    import json
    from pathlib import Path

    raw = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if raw.get("mode") != "global_desktop" or raw.get("schema_version") != 1:
        raise ValueError("Expected a global desktop RunPod profile")
    policy = {key: value for key, value in raw.items() if key not in
              ("mode", "schema_version", "gpu_tiers", "gpu_max_hourly")}
    profile = GlobalGPUProfile(**policy)
    profile.validate()
    if profile.container_disk_gb != 200:
        raise ValueError("New Global GPU profiles require a 200 GB container disk")
    tiers = raw.get("gpu_tiers")
    if not isinstance(tiers, list) or not tiers or any(
        not isinstance(tier, str) or not tier.strip() for tier in tiers
    ):
        raise ValueError("Global profile needs at least one GPU tier")
    finite_positive(raw.get("gpu_max_hourly"))
    return profile


class GlobalGPUProvider:
    def __init__(self, api_key: str, client: httpx.Client | None = None):
        if not isinstance(api_key, str) or not api_key.strip():
            raise ValueError("RunPod API key is required")
        self.client = client or httpx.Client(timeout=httpx.Timeout(30, connect=10), follow_redirects=False)
        self.headers = {"Authorization": "Bearer " + api_key, "User-Agent": "Mozilla/5.0"}

    def deploy(self, profile: GlobalGPUProfile, *, name: str, gpu_type_id: str,
               data_center_id: str | None = None) -> dict:
        """Attempt one paid create; caller must already have persisted intent."""
        payload = profile.deployment_input(name=name, gpu_type_id=gpu_type_id,
                                           data_center_id=data_center_id)
        try:
            response = self.client.post(
                GRAPHQL_URL,
                headers=self.headers,
                json={"query": DEPLOY_MUTATION, "variables": {"input": payload}},
            )
        except httpx.HTTPError:
            raise CreateUncertain("RunPod create outcome is uncertain; reconcile the owned name") from None
        if response.status_code in (401, 403):
            raise ProviderError("RunPod rejected the API credentials", response.status_code)
        if response.status_code in (408, 429) or response.status_code >= 500:
            raise CreateUncertain("RunPod create outcome is uncertain; reconcile the owned name", response.status_code)
        if response.status_code >= 300:
            raise ProviderError("RunPod rejected the GPU deployment", response.status_code)
        try:
            document = response.json()
            if document.get("errors"):
                raise ProviderError("RunPod rejected the GPU deployment schema")
            pod = document["data"]["podFindAndDeployOnDemand"]
            result = {
                "id": identifier(pod["id"]),
                "name": identifier(pod["name"]),
                "costPerHr": finite_positive(pod["costPerHr"]),
            }
        except ProviderError:
            raise
        except (ValueError, TypeError, KeyError, AttributeError):
            raise CreateUncertain("RunPod create response is uncertain; reconcile the owned name") from None
        if result["name"] != name:
            raise CreateUncertain("RunPod returned a different pod name; reconcile the owned name")
        return result

    def reconcile_exact_name(self, name: str) -> dict | None:
        """Read only: find exactly one pod created under a persisted owned name.

        Every page is read. costPerHr is present only when RunPod reports a
        positive cost; a stopped pod's 0.0 is unknown, not free.
        """
        identifier(name)
        try:
            matches = [pod for pod in all_pods(self._page) if pod.get("name") == name]
        except (httpx.HTTPError, ProviderError, ValueError, TypeError, AttributeError):
            raise ProviderError("RunPod owned-pod reconciliation failed") from None
        if len(matches) > 1:
            raise ProviderError("Multiple pods match the owned name; manual reconciliation required")
        if not matches:
            return None
        try:
            result = {"id": identifier(matches[0]["id"]), "name": name}
            if "costPerHr" in matches[0]:
                result["costPerHr"] = finite_positive(matches[0]["costPerHr"])
        except (ValueError, TypeError, KeyError):
            raise ProviderError("RunPod owned-pod reconciliation failed") from None
        return result

    def _page(self, params: dict) -> dict:
        response = self.client.get(BASE_URL + "/pods?" + urlencode(params), headers=self.headers)
        response.raise_for_status()
        return response.json()

    def get(self, pod_id: str) -> dict | None:
        try:
            response = self.client.get(
                BASE_URL + "/pods/" + identifier(pod_id),
                headers=self.headers,
            )
            if response.status_code == 404:
                return None
            response.raise_for_status()
            pod = response.json()
            if not isinstance(pod, dict) or pod.get("id") != pod_id:
                raise ValueError
            return pod_from_v2(pod)
        except (httpx.HTTPError, ValueError, TypeError, ProviderError):
            raise ProviderError("RunPod pod status is unavailable") from None

    def delete(self, pod_id: str) -> None:
        try:
            response = self.client.delete(
                BASE_URL + "/pods/" + identifier(pod_id),
                headers=self.headers,
            )
            if response.status_code != 404:
                response.raise_for_status()
        except httpx.HTTPError:
            raise ProviderError("RunPod pod termination is unconfirmed") from None
