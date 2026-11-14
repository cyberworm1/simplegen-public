"""RunPod v2 adapter. Never adopts manual pods or retries an uncertain create.

The caller persists its run ID and ownership token before calling create, then
persists the returned pod before inspecting allocation_valid. The provider has
no create-time price ceiling or lifetime deadline in its current v2 schema.
"""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import quote

import httpx


class RunPodUnavailable(RuntimeError):
    pass


class RunPodAuthentication(RuntimeError):
    pass


class RequestUncertain(RuntimeError):
    """A create might have succeeded. Reconcile ownership; never blindly retry."""


def _identifier(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", value):
        raise ValueError("Invalid RunPod resource identifier.")
    return value


def _number(value, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"RunPod returned an invalid {label}.")  # noqa: TRY004 -- invalid provider data
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"RunPod returned an invalid {label}.")
    return float(value)


def _public_pod(pod: dict) -> dict:
    """Provider env can include passwords; return only identity markers."""
    fields = (
        "id", "name", "status", "actions", "image", "disk", "ports", "mounts", "gpu",
        "cloud", "dataCenterId", "cudaVersion", "cost", "createdAt", "startedAt", "cluster",
    )
    result = {key: pod[key] for key in fields if key in pod}
    env = pod.get("env") or {}
    if not isinstance(env, dict):
        env = {}
    result["env"] = {key: env[key] for key in ("SIMPLEGEN_RUN_ID", "SIMPLEGEN_OWNER_TOKEN")
                     if key in env}
    return result


class RunPod:
    def __init__(self, settings, client: httpx.Client | None = None):
        self.settings = settings
        self.client = client or httpx.Client(
            base_url="https://api.runpod.io/v2/", timeout=httpx.Timeout(30, connect=10),
            follow_redirects=False,
        )
        self._owns_client = client is None

    def close(self) -> None:
        if self._owns_client:
            self.client.close()

    def configuration(self) -> dict:
        """Read deployment metadata; never the browser/API credential contents."""
        root = Path(self.settings.runpod_config_folder).expanduser().resolve()
        if not self.settings.runpod_config_folder:
            raise ValueError("Choose the existing runpodConfig folder.")
        try:
            session = json.loads((root / "gpu-session.json").read_text(encoding="utf-8"))
            original = session["payload"]
            # Existing deployment snapshots use REST v1 field names.
            network = original.get("mounts", {}).get("network", [])
            volume = original.get("networkVolumeId") or network[0]["volumeId"]
            mount = original.get("volumeMountPath") or network[0]["path"]
            image = original.get("imageName") or original["image"]
            cmd = original.get("dockerStartCmd") or original["cmd"]
            gpu = original.get("gpu", {})
            gpu_id = gpu.get("id") or original["gpuTypeIds"][0]
            count = gpu.get("count", original.get("gpuCount", 1))
            disk = original.get("disk", original.get("containerDiskInGb", 20))
            regions = original["dataCenterIds"]
            ram = gpu.get("minRamPerGpu", original.get("minRAMPerGPU", 128))
            cloud = original.get("cloud", original.get("cloudType", "SECURE"))
            ports = original["ports"]
            if (cloud != "SECURE" or ports != ["8189/http"] or mount != "/workspace"
                    or not isinstance(image, str) or not image
                    or not isinstance(cmd, list) or not cmd
                    or not all(isinstance(part, str) and part for part in cmd)
                    or not isinstance(regions, list) or len(regions) != 1
                    or not isinstance(gpu_id, str) or not gpu_id
                    or any(type(n) is not int or n < 1 for n in (count, disk, ram))):
                raise ValueError
            _identifier(volume)
            _identifier(regions[0])
        except (OSError, ValueError, KeyError, IndexError, TypeError, AttributeError):
            raise ValueError("runpodConfig needs a valid secure-cloud GPU deployment payload "
                             "with /workspace and only authenticated port 8189/http.") from None
        login = root / "browser-login.json"
        if not login.is_file():
            raise ValueError("runpodConfig is missing browser-login.json.")
        return {
            "image": image, "cmd": cmd, "disk": disk, "ports": ports,
            "volume_id": volume, "data_center": regions[0], "mount_path": mount,
            "gpu_id": gpu_id, "gpu_count": count, "min_ram_per_gpu": ram,
            "min_vcpu_per_gpu": gpu.get("minVcpuCountPerGpu", original.get("minVCPUPerGPU")),
            "min_cuda_version": gpu.get("minCudaVersion", "13.0"),
            "browser_login_file": str(login), "cache_path": str(root / "gpu-object-info.json"),
        }

    def _key(self) -> str:
        value = os.environ.get(self.settings.runpod_token_env, "").strip()
        if not value and self.settings.runpod_credentials_file:
            try:
                data = json.loads(Path(self.settings.runpod_credentials_file).expanduser()
                                  .read_text(encoding="utf-8"))
                value = data.get("runpod_api_key", "")
            except (OSError, ValueError, TypeError, AttributeError):
                raise RunPodAuthentication("Cannot read the linked RunPod API credentials file.") from None
        if not isinstance(value, str) or not value.strip():
            raise RunPodAuthentication("Set the RunPod key environment variable or link its credentials file.")
        return value.strip()

    def _request(self, method: str, path: str, *, payload=None, params=None, missing=False):
        creating = method == "POST" and path == "pods"
        try:
            response = self.client.request(
                method, "https://api.runpod.io/v2/" + path,
                headers={"Authorization": "Bearer " + self._key()}, json=payload, params=params,
                follow_redirects=False,
            )
        except httpx.TransportError:
            if creating:
                raise RequestUncertain("RunPod creation response was lost. Reconcile the saved ownership "
                                       "before any further creation request.") from None
            raise RunPodUnavailable("RunPod could not be reached; no provider state was confirmed.") from None
        if response.status_code in (401, 403):
            raise RunPodAuthentication("RunPod rejected the API credentials or their permissions.")
        if response.status_code == 404 and missing:
            return None
        if creating and (response.status_code >= 500 or response.status_code == 408):
            raise RequestUncertain("RunPod creation outcome is uncertain; reconcile the saved ownership.")
        if not 200 <= response.status_code < 300:
            raise RunPodUnavailable(f"RunPod rejected the request (HTTP {response.status_code}).")
        if response.status_code == 204:
            return None
        try:
            result = response.json()
            if not isinstance(result, dict):
                raise ValueError  # noqa: TRY004 -- invalid provider response
            return result
        except ValueError:
            if creating:
                raise RequestUncertain("RunPod creation returned an unreadable response; reconcile ownership.") from None
            raise RunPodUnavailable("RunPod returned an unreadable response.") from None

    def options(self, gpu_count: int | None = None) -> list[dict]:
        config = self.configuration()
        count = config["gpu_count"] if gpu_count is None else gpu_count
        if type(count) is not int or count < 1:
            raise ValueError("GPU count must be a positive integer.")
        volume = self._request("GET", "network-volumes/" + config["volume_id"])
        if volume.get("id") != config["volume_id"] or volume.get("dataCenter") != config["data_center"]:
            raise ValueError("The existing RunPod volume does not match the configured data center.")
        catalog = self._request("GET", "catalog/gpus", params={
            "include": "AVAILABILITY", "product": "POD", "cloud": "SECURE", "count": count,
            "minCudaVersion": config["min_cuda_version"],
        })
        tiers = self.settings.runpod_gpu_tiers or [config["gpu_id"]]
        if not isinstance(tiers, list) or not all(isinstance(item, str) for item in tiers):
            raise ValueError("RunPod GPU tiers must be a list of GPU IDs.")
        options = []
        for gpu in catalog.get("gpus", []):
            if gpu.get("id") not in tiers or not gpu.get("secure"):
                continue
            region = next((dc for dc in gpu.get("dataCenters", [])
                           if dc.get("id") == config["data_center"]), None)
            if not region or region.get("availability") not in ("LOW", "MEDIUM", "HIGH"):
                continue
            rate = _number(gpu.get("price", {}).get("secure"), "GPU price") * count
            options.append({
                "gpu_id": gpu["id"], "gpu_count": count,
                "vram_gb": _number(gpu.get("memory"), "GPU VRAM"),
                "hourly_usd": rate, "availability": region["availability"],
                "data_center": config["data_center"], "volume_id": config["volume_id"],
                "quoted_at": datetime.now(UTC).isoformat(),
            })
        return sorted(options, key=lambda item: tiers.index(item["gpu_id"]))

    def _environment(self) -> dict:
        root = Path(self.settings.runpod_config_folder).expanduser()
        try:
            env = json.loads((root / "gpu-session.json").read_text())["payload"].get("env", {})
            if not isinstance(env, dict) or not all(
                isinstance(key, str) and isinstance(value, str) for key, value in env.items()
            ):
                raise ValueError
            return dict(env)
        except (OSError, ValueError, KeyError, TypeError):
            raise ValueError("The linked deployment environment is invalid.") from None

    @staticmethod
    def name(run_id: str) -> str:
        return "simplegen-" + _identifier(run_id)

    def verify_owned(self, pod: dict, run_id: str, owner_token: str) -> None:
        _identifier(owner_token)
        config = self.configuration()
        env = pod.get("env")
        mounts = pod.get("mounts")
        if (pod.get("name") != self.name(run_id)
                or not isinstance(env, dict) or not isinstance(mounts, dict)
                or env.get("SIMPLEGEN_RUN_ID") != run_id
                or env.get("SIMPLEGEN_OWNER_TOKEN") != owner_token
                or mounts.get("network") != [
                    {"volumeId": config["volume_id"], "path": config["mount_path"]}]
                or pod.get("cluster")):
            raise ValueError("Pod ownership or attached volume does not match this simpleGen run.")
        _identifier(pod.get("id"))

    def create(self, choice: dict, run_id: str, owner_token: str) -> dict:
        """Exactly one POST; persist intent before calling and receipt immediately after."""
        if not self.settings.runpod_managed:
            raise ValueError("Enable managed RunPod before creating a pod.")
        _identifier(owner_token)
        name = self.name(run_id)
        config = self.configuration()
        cap = _number(choice.get("hourly_usd"), "accepted hourly ceiling")
        current = next((item for item in self.options(choice.get("gpu_count"))
                        if all(item[key] == choice.get(key) for key in
                               ("gpu_id", "gpu_count", "volume_id", "data_center"))), None)
        if not current or current["hourly_usd"] > cap:
            raise RunPodUnavailable("Selected GPU availability or price changed. Refresh and choose again.")
        gpu = {"id": current["gpu_id"], "count": current["gpu_count"],
               "minRamPerGpu": config["min_ram_per_gpu"],
               "minCudaVersion": config["min_cuda_version"]}
        if config["min_vcpu_per_gpu"] is not None:
            gpu["minVcpuCountPerGpu"] = config["min_vcpu_per_gpu"]
        payload = {
            "name": name, "image": config["image"], "cmd": config["cmd"],
            "disk": config["disk"], "ports": config["ports"], "cloud": "SECURE", "gpu": gpu,
            "dataCenterIds": [config["data_center"]],
            "mounts": {"network": [{"volumeId": config["volume_id"], "path": config["mount_path"]}]},
            "env": {**self._environment(), "SIMPLEGEN_RUN_ID": run_id,
                    "SIMPLEGEN_OWNER_TOKEN": owner_token},
            "startSsh": False, "startJupyter": False,
        }
        response = self._request("POST", "pods", payload=payload)
        if not isinstance(response, dict):
            raise RequestUncertain("Created pod receipt was missing; reconcile saved ownership.")
        pod = _public_pod(response)
        if not isinstance(pod.get("id"), str):
            raise RequestUncertain("Created pod identifier was missing; reconcile saved ownership.")
        return self.validate_allocation(pod, choice, run_id, owner_token)

    def validate_allocation(self, pod: dict, choice: dict, run_id: str, owner_token: str) -> dict:
        """Check a new or reconciled pod; persist it before acting on a failed check."""
        config = self.configuration()
        pod = _public_pod(pod)
        cap = _number(choice.get("hourly_usd"), "accepted hourly ceiling")
        errors = []
        try:
            self.verify_owned(pod, run_id, owner_token)
            if _number(pod.get("cost"), "allocated hourly price") > cap:
                errors.append("Allocated hourly price exceeds the accepted ceiling.")
        except ValueError as exc:
            errors.append(str(exc))
        gpu = pod.get("gpu")
        if (not isinstance(gpu, dict) or gpu.get("id") != choice["gpu_id"]
                or gpu.get("count") != choice["gpu_count"]):
            errors.append("Allocated GPU differs from the selected GPU.")
        if pod.get("dataCenterId") != config["data_center"] or pod.get("cloud") != "SECURE":
            errors.append("Allocated region or cloud differs from the selected deployment.")
        pod["allocation_valid"] = not errors
        pod["allocation_error"] = " ".join(errors)
        return pod

    def get(self, pod_id: str) -> dict | None:
        pod = self._request("GET", "pods/" + quote(_identifier(pod_id), safe=""), missing=True)
        return _public_pod(pod) if pod is not None else None

    def find_owned(self, run_id: str, owner_token: str) -> dict | None:
        name = self.name(run_id)
        matches = []
        cursor = None
        seen = set()
        while True:
            page = self._request("GET", "pods", params={"limit": 100, **({"cursor": cursor} if cursor else {})})
            pods = page.get("pods")
            pagination = page.get("pagination")
            if (not isinstance(pods, list) or not all(isinstance(pod, dict) for pod in pods)
                    or not isinstance(pagination, dict)
                    or type(pagination.get("hasNextPage")) is not bool):
                raise RunPodUnavailable("RunPod returned invalid pagination; ownership remains uncertain.")
            for raw in pods:
                if raw.get("name") != name:
                    continue
                pod = _public_pod(raw)
                self.verify_owned(pod, run_id, owner_token)
                matches.append(pod)
            if not pagination.get("hasNextPage"):
                break
            cursor = pagination.get("nextCursor")
            if not isinstance(cursor, str) or not cursor or cursor in seen:
                raise RunPodUnavailable("RunPod returned invalid pagination; ownership remains uncertain.")
            seen.add(cursor)
        if len(matches) > 1:
            raise RequestUncertain("Multiple pods match this run; automatic adoption and termination are blocked.")
        return matches[0] if matches else None

    def terminate(self, pod_id: str, run_id: str, owner_token: str) -> bool:
        """Caller must supply a durably owned ID. True only after verified absence."""
        pod = self.get(pod_id)
        if pod is None:
            return True
        self.verify_owned(pod, run_id, owner_token)
        if pod.get("id") != pod_id:
            raise ValueError("RunPod returned a different pod; termination blocked.")
        self._request("DELETE", "pods/" + quote(_identifier(pod_id), safe=""), missing=True)
        return self.get(pod_id) is None

    def endpoint_settings(self, pod: dict):
        config = self.configuration()
        pod_id = _identifier(pod["id"])
        return replace(self.settings, comfy_url=f"https://{pod_id}-8189.proxy.runpod.net",
                       cloud=True, cloud_name="RunPod", token_env="",
                       browser_login_file=config["browser_login_file"])
