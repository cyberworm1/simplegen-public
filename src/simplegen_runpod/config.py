"""Versioned session policy. Failures alert; only downloads or manual action retire compute."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

from .provider import finite_positive, identifier


@dataclass(frozen=True)
class Config:
    volume_id: str
    region: str
    cpu_payload: dict
    gpu_payload: dict
    cpu_max_hourly: float
    gpu_max_hourly: float
    schema_version: int = 2
    idle_seconds: int = 300
    poll_seconds: int = 10
    startup_alert_seconds: int = 1200
    observation_alert_seconds: int = 600
    session_alert_seconds: int = 28800
    required_nodes: list[str] = field(default_factory=list)
    allow_provisioning: bool = False

    def validate(self) -> None:
        if self.schema_version != 2:
            raise ValueError(
                "Use schema_version 2 session policy; legacy deadline policies cannot be reused"
            )
        identifier(self.volume_id)
        if not re.fullmatch(r"[A-Z]+-[A-Z0-9]+-\d+", self.region):
            raise ValueError("Invalid region")
        if type(self.allow_provisioning) is not bool:
            raise ValueError("allow_provisioning must be a boolean")
        if self.idle_seconds != 300:
            raise ValueError("Verified downloads require five minutes of empty queues")
        for name in (
            "poll_seconds",
            "startup_alert_seconds",
            "observation_alert_seconds",
            "session_alert_seconds",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.poll_seconds > 30:
            raise ValueError("Poll at least every 30 seconds")
        if not isinstance(self.required_nodes, list) or any(
            not isinstance(n, str) or not n for n in self.required_nodes
        ):
            raise ValueError("required_nodes must be node class names")
        for name in ("cpu_max_hourly", "gpu_max_hourly"):
            if type(getattr(self, name)) not in (float, int):
                raise ValueError("Hourly limits must be numbers")
            finite_positive(getattr(self, name))
        for role, payload in (("CPU", self.cpu_payload), ("GPU", self.gpu_payload)):
            if (
                payload.get("computeType") != role
                or payload.get("cloudType") != "SECURE"
            ):
                raise ValueError(
                    "Only explicitly typed Secure Cloud pods are permitted"
                )
            if any(
                k in payload
                for k in (
                    "env",
                    "name",
                    "templateId",
                    "dockerEntrypoint",
                    "dockerStartCmd",
                )
            ):
                raise ValueError(
                    "Names, credentials and startup commands are managed by this module"
                )
            if (
                payload.get("interruptible", False)
                or payload.get("locked", False)
                or payload.get("volumeInGb") != 0
            ):
                raise ValueError(
                    "Use unlocked on-demand pods, existing network storage and disposable container disks"
                )
            if not re.search(r"@sha256:[a-f0-9]{64}$", payload.get("imageName", "")):
                raise ValueError("Pin container images by sha256 digest")
            if (
                payload.get("supportPublicIp") is not False
                or payload.get("containerDiskInGb", 0) <= 0
            ):
                raise ValueError(
                    "Disable public TCP IP exposure and supply a positive container disk"
                )
            if payload.get("dataCenterIds") != [self.region]:
                raise ValueError("Use the existing volume region")
            if role == "GPU":
                if (
                    payload.get("networkVolumeId") != self.volume_id
                    or payload.get("volumeMountPath") != "/workspace"
                ):
                    raise ValueError(
                        "GPU must use the existing /workspace network volume"
                    )
                if payload.get("ports") != ["8189/http"]:
                    raise ValueError("GPU uses its existing authenticated gateway")
                if (
                    len(payload.get("gpuTypeIds", [])) != 1
                    or type(payload.get("gpuCount")) is not int
                    or payload["gpuCount"] < 1
                ):
                    raise ValueError("Choose one GPU type and positive count")
            else:
                if payload.get("networkVolumeId") or payload.get("ports") != [
                    "8097/http"
                ]:
                    raise ValueError(
                        "CPU uses S3 for durable state and exposes only its authenticated control API"
                    )
                if (
                    len(payload.get("cpuFlavorIds", [])) != 1
                    or type(payload.get("vcpuCount")) is not int
                    or payload["vcpuCount"] < 1
                ):
                    raise ValueError("Choose one CPU flavor and positive vCPU count")

    def public(self) -> dict:
        return asdict(self)
