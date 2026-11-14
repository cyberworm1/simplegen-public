"""Desktop-owned Global Volume session. No CPU pod, S3 journal or cloud action on import."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import time
import uuid
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

import httpx

from simplegen_runpod.global_gpu import (
    CreateUncertain,
    GlobalGPUProfile,
    GlobalGPUProvider,
    load_global_profile,
)
from simplegen_runpod.provider import ProviderError, RunPod, available_gpu_centers, gpu_is_stopped
from simplegen_runpod.worker import ComfyMonitor

from .credentials import _read_private
from .settings import Settings

PROCESS_ID = uuid.uuid4().hex
POLL_GAP_SECONDS = 30
IDLE_SECONDS = 300
STARTUP_POD_CHECK_SECONDS = 30  # RunPod pod lookups while waiting for ComfyUI to start.


def is_global(settings: Settings, state: dict | None = None) -> bool:
    saved = (state or {}).get("managed_session", {})
    if saved.get("backend"):
        return saved["backend"] == "global_desktop"
    if saved.get("sid"):
        return False  # A saved schema-v2 session retains its original CPU recovery path.
    if not settings.managed_runpod or not settings.cloud:
        return False
    if settings.global_desktop:
        return True
    try:
        raw = json.loads(Path(settings.runpod_config_file).expanduser().read_text())
        return raw.get("mode") == "global_desktop"
    except (OSError, ValueError, TypeError):
        return False


def session_for(settings, folder, state, persist, emit, cancel, detach):
    if is_global(settings, state):
        return GlobalSession(settings, folder, state, persist, emit, cancel, detach)
    from .managed import ManagedSession
    return ManagedSession(settings, folder, state, persist, emit, cancel, detach)


@dataclass(frozen=True)
class GlobalConfig:
    global_profile: object
    gpu_tiers: list[str]
    gpu_max_hourly: float

    @property
    def gpu_payload(self):
        return {"gpuTypeIds": self.gpu_tiers[:1], "gpuCount": 1}


def _secret(settings, key):
    value = os.environ.get(key)
    if not value:
        data = _read_private(Path(settings.runpod_credentials_file).expanduser())
        value = data.get(key)
        if isinstance(value, dict) and set(value) == {"file", "key"}:
            value = _read_private(Path(value["file"]).expanduser()).get(value["key"])
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Global RunPod setup is missing {key}")
    return value.strip()


class GlobalSession:
    def __init__(self, settings, folder, state, persist, emit, cancel, detach):
        self.settings, self.folder, self.state = settings, Path(folder), state
        self.persist, self.emit, self.cancel, self.detach = persist, emit, cancel, detach
        self._provider = None
        self._monitor = None

    @property
    def saved(self):
        return self.state.setdefault("managed_session", {})

    @property
    def retired(self):
        return self.saved.get("retired", False)

    def config(self):
        if self.saved.get("frozen_profile"):
            frozen = self.saved["frozen_profile"]
            return GlobalConfig(GlobalGPUProfile(**frozen["profile"]),
                                frozen["gpu_tiers"], frozen["gpu_max_hourly"])
        path = Path(self.settings.runpod_config_file).expanduser()
        raw = json.loads(path.read_text())
        profile = load_global_profile(path)
        max_hourly = float(raw["gpu_max_hourly"])
        if self.state.get("members") and "max_hourly" in self.state:
            selected = self.state["max_hourly"]
            if type(selected) not in (int, float) or not math.isfinite(selected) or selected <= 0:
                raise ValueError("Batch hourly ceiling must be a positive finite amount")
            max_hourly = float(selected)
        return GlobalConfig(profile, list(dict.fromkeys([*raw["gpu_tiers"], *self.settings.runpod_gpu_tiers])),
                            max_hourly)

    def schema(self):
        from .settings import atomic_json
        target = self.folder / "managed-global-object-info.json"
        if self.saved.get("schema_sha256"):
            content = target.read_bytes()
            if hashlib.sha256(content).hexdigest() != self.saved["schema_sha256"]:
                raise ValueError("Saved Global Volume node schema changed")
            raw = json.loads(content)
        else:
            raw = json.loads(Path(self.settings.runpod_schema_file).expanduser().read_text())
        schema = raw.get("object_info", raw)
        if not isinstance(schema, dict) or not schema:
            raise ValueError("Cached global ComfyUI schema is missing")
        if not self.saved.get("schema_sha256"):
            atomic_json(target, schema)
            self.saved["schema_sha256"] = hashlib.sha256(target.read_bytes()).hexdigest()
            self.persist()
        return schema

    def _api(self):
        if self._provider is None:
            self._provider = GlobalGPUProvider(_secret(self.settings, "RUNPOD_API_KEY"))
        return self._provider

    def _endpoint(self, pod_id):
        from simplegen_runpod.provider import identifier

        from .credentials import _exclusive_json, _private_directory
        from .settings import user_dir
        login_file = self.settings.browser_login_file
        if not login_file:
            username, password = self._login()
            directory = user_dir() / "runpod" / "logins"
            _private_directory(directory)
            target = directory / (self.saved["sid"] + ".json")
            _exclusive_json(target, {"username": username, "password": password})
            login_file = str(target)
        return replace(self.settings, comfy_url=f"https://{identifier(pod_id)}-8189.proxy.runpod.net",
                       cloud=True, cloud_name="RunPod", token_env="", global_desktop=True,
                       browser_login_file=login_file)

    def _login(self):
        if self.settings.browser_login_file:
            login = _read_private(Path(self.settings.browser_login_file).expanduser())
            return login["username"], login["password"]
        return _secret(self.settings, "SG_COMFY_USERNAME"), _secret(self.settings, "SG_COMFY_PASSWORD")

    def _owned_pod(self):
        pod_id = self.saved.get("pod_id")
        if not pod_id:
            return None
        pod = self._api().get(pod_id)
        if pod is not None and pod.get("name") != self.saved.get("name"):
            raise RuntimeError("RunPod pod ownership mismatch")
        return pod

    def _startup_pod_gone(self) -> str | None:
        """Why the owned pod can no longer become ready, or None (also when RunPod can't be asked).

        Polled during the startup wait so a pod deleted or stopped outside
        simpleGen ends the wait in about a minute instead of the full startup
        timeout. Only a 404 confirms billing stopped; EXITED/TERMINATED end the
        wait but keep the pod owned (the failure shutdown deletes it).
        """
        try:
            pod = self._owned_pod()
        except ProviderError:
            return None  # Unknown is not absence; the startup timeout still applies.
        saved = self.saved
        if pod is None:
            saved.update(retired=True, billing_confirmed_stopped=True, phase="retired", idle_since=None)
            saved.setdefault("session_completed_at", datetime.now(UTC).isoformat())
            self.persist()
            return "GPU pod was deleted in RunPod during startup"
        if pod.get("desiredStatus") in ("EXITED", "TERMINATED"):
            saved["phase"] = "stopped"
            self.persist()
            return f"GPU pod was stopped in RunPod during startup ({pod['desiredStatus']})"
        return None

    def connect(self, choice=None):
        if self.cancel.is_set() or self.detach.is_set():
            raise RuntimeError("Global GPU creation was cancelled before provisioning")
        self.schema()
        cfg = self.config()
        saved = self.saved
        if not saved.get("name"):
            if not choice or choice.get("gpu_id") not in cfg.gpu_tiers:
                raise ValueError("Choose a GPU in simpleGen before provisioning RunPod")
            if choice.get("gpu_count") != 1 or choice.get("volume_id") != cfg.global_profile.volume_id:
                raise ValueError("Selected GPU does not match the global profile")
            cap = choice.get("hourly_usd")
            if type(cap) not in (int, float) or not 0 < cap <= cfg.gpu_max_hourly:
                raise ValueError("Selected GPU quote exceeds the configured hourly limit")
            # The fresh catalog quote is advisory; create's returned rate is checked too.
            quote = RunPod(_secret(self.settings, "RUNPOD_API_KEY"))
            try:
                fresh = quote.quote({"computeType": "GPU", "gpuTypeIds": [choice["gpu_id"]], "gpuCount": 1})
                catalog = quote._request("GET", "/catalog/gpus?include=AVAILABILITY&product=POD&cloud=SECURE&count=1&minCudaVersion="
                                         + cfg.global_profile.min_cuda_version,
                                         catalog=True)
                matching = [gpu for gpu in catalog.get("gpus", []) if gpu.get("id") == choice["gpu_id"]]
                if len(matching) != 1 or not matching[0].get("secure") or not available_gpu_centers(
                    matching[0].get("dataCenters")
                ):
                    raise ValueError("Selected GPU is no longer available in a CUDA-compatible Secure Cloud data center")
            finally:
                quote.client.close()
            if fresh > cap:
                raise ValueError("GPU price changed; choose compute again")
            saved.update(backend="global_desktop", sid=uuid.uuid4().hex,
                         name="sg-" + uuid.uuid4().hex + "-gpu", gpu_id=choice["gpu_id"],
                         placement="global", data_center_id=None,
                         quoted_hourly=fresh, accepted_hourly=cap, create_intent=True,
                         frozen_profile={"profile": asdict(cfg.global_profile),
                                         "gpu_tiers": cfg.gpu_tiers, "gpu_max_hourly": cfg.gpu_max_hourly,
                                         "keep_warm_minutes": getattr(self.settings, "keep_warm_minutes", 5)},
                         phase="creating", jobs={})
            self.persist()
        from .global_registry import register
        register(self.folder)  # Durable desktop monitor registration before paid POST.
        if not saved.get("pod_id"):
            # Never make a second POST for a persisted intent. Reconcile exact name.
            if saved.get("create_attempted"):
                matched = self._api().reconcile_exact_name(saved["name"])
                if matched is None:
                    raise CreateUncertain("Create outcome unresolved; no second pod was requested")
                saved["pod_id"] = matched["id"]
                if "costPerHr" in matched:  # Unknown cost keeps the accepted quote.
                    saved["actual_hourly"] = matched["costPerHr"]
                saved.setdefault("session_started_at", datetime.now(UTC).isoformat())
                self.persist()
            else:
                if self.cancel.is_set() or self.detach.is_set():
                    raise RuntimeError("Global GPU creation was cancelled before provisioning")
                saved["create_attempted"] = True
                self.persist()
                pod = self._api().deploy(cfg.global_profile, name=saved["name"],
                                         gpu_type_id=saved["gpu_id"],
                                         data_center_id=saved.get("data_center_id"))
                saved["pod_id"] = pod["id"]
                saved["actual_hourly"] = pod["costPerHr"]
                saved["session_started_at"] = datetime.now(UTC).isoformat()
                self.persist()
        saved["quoted_gpu_hourly_usd"] = saved.get("actual_hourly", saved.get("quoted_hourly"))
        saved["quoted_cpu_hourly_usd"] = 0.0
        self.persist()
        if saved.get("actual_hourly", 0) > min(cfg.gpu_max_hourly, saved.get("accepted_hourly", cfg.gpu_max_hourly)):
            raise RuntimeError("Allocated GPU exceeds configured hourly limit; inspect owned pod")
        pod = self._owned_pod()
        if pod is None or gpu_is_stopped(pod):
            saved["retired"] = pod is None
            saved["billing_confirmed_stopped"] = pod is None
            self.persist()
            raise RuntimeError("Owned Global Volume GPU is absent or stopped")
        endpoint = self._endpoint(saved["pod_id"])
        from .comfy import Comfy, ComfyRequestError
        deadline = time.monotonic() + 60 * getattr(self.settings, "startup_timeout_minutes", 45)
        comfy = Comfy(endpoint)
        last_note = 0.0
        last_pod_check = time.monotonic()
        try:
            while time.monotonic() < deadline:
                if time.monotonic() - last_pod_check >= STARTUP_POD_CHECK_SECONDS:
                    last_pod_check = time.monotonic()
                    gone = self._startup_pod_gone()
                    if gone:
                        from .cost_safety import record_owner_failure
                        if record_owner_failure(self.state, self.settings, gone, "startup"):
                            self.persist()
                        raise RuntimeError(f"{gone}; no render was submitted")
                try:
                    stats = comfy.request("GET", "/system_stats").json()
                    nodes = comfy.object_info()
                    if any(d.get("type") == "cuda" for d in stats.get("devices", [])) and isinstance(nodes, dict) and nodes:
                        self._check_drift(nodes)
                        return endpoint
                except ComfyRequestError as exc:
                    if "HTTP 401" in str(exc) or "HTTP 403" in str(exc):
                        raise RuntimeError("RunPod gateway credentials were rejected; owned compute remains active") from None
                except (ValueError, TypeError, KeyError):
                    pass
                if time.monotonic() - last_note >= 15:
                    self.emit({"stage": "cloud", "message": "RunPod startup: waiting for authenticated ComfyUI; GPU billing may be active.", "progress": None})
                    last_note = time.monotonic()
                if self.cancel.is_set() or self.detach.is_set():
                    raise RuntimeError("Global GPU readiness wait interrupted")
                time.sleep(2)
        finally:
            comfy.close()
        from .cost_safety import record_owner_failure
        minutes = getattr(self.settings, "startup_timeout_minutes", 45)
        if record_owner_failure(self.state, self.settings, f"GPU not ready within {minutes} minutes", "startup"):
            self.persist()
        raise RuntimeError("Global GPU readiness timed out; owned compute remains for inspection")

    def _check_drift(self, live):
        """Once per session: snapshot the live schema and record drift. Never edits the cache."""
        from .cost_safety import check_schema_drift, drift_notice
        try:
            record = check_schema_drift(self.folder, self.saved, self.settings, live)
        except (OSError, ValueError, TypeError) as error:
            self.saved["schema_drift"] = {"sid": self.saved.get("sid"), "error": type(error).__name__}
            self.persist()
            return
        if record is not None:
            self.persist()
            if record.get("drift"):
                self.emit({"stage": "cloud", "message": drift_notice(record), "progress": None})

    def _keep_warm_seconds(self):
        """New sessions record keep_warm_minutes in their frozen profile; older ones keep 5 minutes."""
        minutes = (self.saved.get("frozen_profile") or {}).get("keep_warm_minutes")
        return minutes * 60 if type(minutes) is int and 5 <= minutes <= 60 else IDLE_SECONDS

    def _spend_check(self):
        """Alert once when actual spend passes the batch cap. Never cancels work."""
        cap = self.state.get("max_total_usd")
        if not isinstance(cap, (int, float)) or isinstance(cap, bool):
            return
        from .cost_safety import alert, spent_usd
        spent = spent_usd(self.state)
        if spent is not None and spent > cap:
            alert(self.state, self.settings, "spend_cap_exceeded",
                  f"Spent about ${spent:.2f} of a ${cap:.2f} cap.", key=f"spend_cap:{cap}")

    def _failure_minutes(self):
        minutes = getattr(self.settings, "failure_shutdown_minutes", 15)
        return minutes if type(minutes) is int and 0 < minutes <= 240 else 0

    def _startup_failure_idle(self):
        """Startup never became ready and no job was ever registered: no queue evidence needed."""
        saved = self.saved
        failure = saved.get("owner_failure") or {}
        return (failure.get("kind") == "startup" and not saved.get("jobs")
                and not saved.get("downloads_verified") and self._failure_minutes() > 0)

    def _failure_countdown(self, eligible, now):
        """Advance or reset the failure countdown; continuity rules match the normal idle window."""
        saved = self.saved
        if (not eligible or saved.get("idle_process") != PROCESS_ID
                or now - saved.get("last_observed_monotonic", now) > POLL_GAP_SECONDS):
            saved["failure_idle_since"] = None
        if eligible and saved.get("failure_idle_since") is None:
            saved["failure_idle_since"] = now
            from .cost_safety import alert
            alert(self.state, self.settings, "failure_countdown",
                  f"Shutdown in {self._failure_minutes()} min unless work resumes.")
        saved["failure_eligible"] = bool(eligible)
        saved["failure_remaining_seconds"] = (
            max(0, 60 * self._failure_minutes() - int(now - saved["failure_idle_since"]))
            if saved.get("failure_idle_since") is not None else None)

    def before_submit(self, job_id=None):
        key = job_id or self.state["run_id"]
        jobs = self.saved.setdefault("jobs", {})
        if self.saved.get("downloads_verified"):
            raise RuntimeError("Global session is sealed after verified downloads")
        if key in jobs:
            return False
        jobs[key] = {"prompt_id": None, "status": "submission_pending", "outputs": []}
        self.persist()
        return True

    def bind(self, prompt_id, job_id=None):
        key = job_id or self.state["run_id"]
        job = self.saved.setdefault("jobs", {})[key]
        if job["prompt_id"] not in (None, prompt_id):
            raise ValueError("Another prompt is already bound")
        job["prompt_id"] = prompt_id
        self.persist()

    def status(self):
        saved = self.saved
        if saved.get("handed_off"):
            # Another batch adopted this pod; this record no longer owns it. No provider calls.
            return copy.deepcopy(saved)
        if saved.get("name") and not saved.get("pod_id") and saved.get("create_attempted"):
            matched = self._api().reconcile_exact_name(saved["name"])
            if matched is not None:
                saved["pod_id"] = matched["id"]
                if "costPerHr" in matched:  # Unknown cost keeps the accepted quote.
                    saved["actual_hourly"] = matched["costPerHr"]
                saved.setdefault("session_started_at", datetime.now(UTC).isoformat())
                saved["quoted_gpu_hourly_usd"] = saved.get("actual_hourly", saved.get("quoted_hourly"))
                saved["quoted_cpu_hourly_usd"] = 0.0
                self.persist()
        if not saved.get("pod_id"):
            return copy.deepcopy(saved) if saved else {"phase": "not_started"}
        try:
            pod = self._owned_pod()
        except (ProviderError, RuntimeError):
            saved["idle_since"] = None
            saved["failure_idle_since"] = None
            self.persist()
            raise
        if pod is None or gpu_is_stopped(pod):
            saved.update(retired=pod is None, billing_confirmed_stopped=pod is None,
                         phase="retired" if pod is None else "stopped", idle_since=None)
            if pod is None:
                saved.setdefault("session_completed_at", datetime.now(UTC).isoformat())
            self.persist()
            if pod is None:
                from .global_registry import remove
                remove(self.folder)
                self._record_cost()
            return copy.deepcopy(saved)
        if self._startup_failure_idle():
            now = time.monotonic()
            self._failure_countdown(True, now)
            saved["idle_process"] = PROCESS_ID
            saved["last_observed_monotonic"] = now
            saved["idle_since"] = None
            saved["phase"] = "startup_failed"
            self.persist()
            if now - saved["failure_idle_since"] >= 60 * self._failure_minutes():
                self._delete_owned("failure_idle")
            return copy.deepcopy(saved)
        try:
            if self._monitor is None:
                username, password = self._login()
                inherited = saved.get("inherited_prompt_ids")
                # An adopted pod's queue/history already holds the previous owner's prompts.
                self._monitor = (ComfyMonitor(saved["pod_id"], username, password, [], extra_known=inherited)
                                 if inherited else ComfyMonitor(saved["pod_id"], username, password, []))
            observed = self._monitor.poll(saved.get("jobs", {}))
        except (OSError, RuntimeError, ValueError, TypeError, KeyError, httpx.HTTPError):
            saved["idle_since"] = None
            saved["failure_idle_since"] = None
            self.persist()
            raise
        if observed.get("unregistered_jobs") is True:
            saved["unregistered_jobs_seen"] = True
        for job in saved.get("jobs", {}).values():
            pid = job.get("prompt_id")
            if pid:
                job["status"] = observed.get("jobs", {}).get(pid, job["status"])
                if pid in observed.get("outputs", {}):
                    job["outputs"] = observed["outputs"][pid]
        from simplegen_runpod.outputs import manifest
        saved["manifest"] = manifest(saved.get("jobs", {}))
        if saved.get("downloads_verified") and saved.get("acknowledged_manifest") != saved["manifest"]:
            saved["downloads_verified"] = False
        now = time.monotonic()
        eligible = (saved.get("downloads_verified") is True and observed.get("ready") is True
                    and observed.get("healthy") is True and observed.get("auth_verified") is True
                    and not saved.get("unregistered_jobs_seen")
                    and observed.get("queue_empty") is True and bool(saved.get("jobs"))
                    and all(j.get("status") == "success" and j.get("outputs") for j in saved["jobs"].values()))
        if (not eligible or saved.get("idle_process") != PROCESS_ID
                or now - saved.get("last_observed_monotonic", now) > POLL_GAP_SECONDS):
            saved["idle_since"] = None
        if eligible and saved["idle_since"] is None:
            saved["idle_since"] = now
        failure_eligible = self._failure_policy(observed)
        self._failure_countdown(failure_eligible, now)
        saved["idle_process"] = PROCESS_ID
        saved["last_observed_monotonic"] = now
        saved["last_observation_healthy"] = bool(observed.get("ready") is True and
                                                   observed.get("healthy") is True and
                                                   observed.get("auth_verified") is True)
        saved["queue_empty"] = observed.get("queue_empty") is True
        saved["phase"] = "downloaded" if saved.get("downloads_verified") else "ready" if observed.get("ready") else "starting"
        saved["idle_remaining_seconds"] = (
            max(0, self._keep_warm_seconds() - int(now - saved["idle_since"]))
            if saved["idle_since"] is not None else None
        )
        self._spend_check()
        if saved.get("owner_failure") and not saved.get("downloads_verified"):
            saved["phase"] = "failed"
        self.persist()
        if saved.get("handoff_intent"):
            pass  # A handoff is in progress: never delete; recovery finishes or rolls it back.
        elif eligible and now - saved["idle_since"] >= self._keep_warm_seconds():
            self._delete_owned("verified_downloads_idle")
        elif failure_eligible and now - saved["failure_idle_since"] >= 60 * self._failure_minutes():
            self._delete_owned("failure_idle")
        return copy.deepcopy(saved)

    def _failure_policy(self, observed):
        """Owner failed and nothing collectable remains: every successful job's outputs are
        downloaded and verified locally, nothing runs, queues are empty, observations healthy."""
        from .cost_safety import alert, owner_receipts, uncollected
        saved = self.saved
        failure = saved.get("owner_failure")
        if not failure or saved.get("downloads_verified"):
            saved.pop("attention", None)
            return False
        jobs = saved.get("jobs", {})
        healthy = (observed.get("ready") is True and observed.get("healthy") is True
                   and observed.get("auth_verified") is True)
        settled = all(j.get("status") in ("success", "error") for j in jobs.values())
        unresolved = sorted(k for k, j in jobs.items() if j.get("status") in ("missing", "submission_pending"))
        if (healthy and unresolved and observed.get("queue_empty") is True
                and not any(j.get("status") == "running" for j in jobs.values())):
            saved["unresolved_jobs"] = unresolved
            alert(self.state, self.settings, "unresolved_jobs", "Jobs: " + ", ".join(unresolved[:20]))
        elif healthy:
            saved.pop("unresolved_jobs", None)
        if not (healthy and settled and observed.get("queue_empty") is True):
            return False
        missing = uncollected(jobs, owner_receipts(self.folder, self.state)) if any(
            j.get("status") == "success" for j in jobs.values()) else []
        if missing:
            saved["attention"] = "outputs_not_collected"
            saved["uncollected_outputs"] = missing[:50]
            alert(self.state, self.settings, "outputs_not_collected",
                  f"{len(missing)} output(s) are not downloaded. Resume or retry downloads; the GPU is kept.")
            return False
        saved.pop("attention", None)
        saved.pop("uncollected_outputs", None)
        return (self._failure_minutes() > 0 and not saved.get("unregistered_jobs_seen")
                and (failure.get("kind") != "batch" or bool(saved.get("recovery_completed_at"))))

    def complete(self, downloads):
        from .engine import probe_video
        if self.saved.get("downloads_verified") and self.saved.get("retired"):
            return copy.deepcopy(self.saved)
        remote = self.status()
        manifest = remote.get("manifest", [])
        if (remote.get("last_observation_healthy") is not True
                or remote.get("unregistered_jobs_seen") or not remote.get("jobs")
                or any(j.get("status") != "success" or not j.get("outputs")
                       for j in remote["jobs"].values())):
            raise RuntimeError("Every registered Comfy job must finish before download acknowledgement")
        expected = {(x["prompt_id"], x["filename"], x.get("subfolder", ""), x.get("type", "output"))
                    for x in manifest}
        found = set()
        for item in downloads:
            key = (item.get("prompt_id"), item["filename"], item.get("subfolder", ""),
                   item.get("type", "output"))
            if key not in expected or key in found:
                continue
            path = Path(item["path"])
            if not path.is_file() or path.stat().st_size != item["size_bytes"] or item["size_bytes"] <= 0:
                raise ValueError("Global video download is missing or incomplete")
            probe_video(path)
            found.add(key)
        if not expected or found != expected:
            raise RuntimeError("Every generated video must be downloaded before GPU retirement")
        self.saved["downloads_verified"] = True
        self.saved["acknowledged_manifest"] = copy.deepcopy(manifest)
        self.saved["download_receipt"] = {"files": sorted([list(x) for x in found])}
        self.persist()
        return self.status()

    def _delete_owned(self, reason):
        saved = self.saved
        if saved.get("handed_off") or saved.get("handoff_intent"):
            raise RuntimeError("This pod is being or has been handed off to another batch; not deleting it")
        pod = self._owned_pod()
        if pod is None:
            saved.update(retired=True, billing_confirmed_stopped=True, phase="retired")
            saved.setdefault("session_completed_at", datetime.now(UTC).isoformat())
            self.persist()
            from .global_registry import remove
            remove(self.folder)
            self._record_cost()
            return
        if reason == "verified_downloads_idle" and not (
            saved.get("downloads_verified") is True
            and saved.get("acknowledged_manifest") == saved.get("manifest")
            and saved.get("idle_process") == PROCESS_ID
            and saved.get("idle_since") is not None
            and time.monotonic() - saved["idle_since"] >= self._keep_warm_seconds()
            and time.monotonic() - saved.get("last_observed_monotonic", 0) <= POLL_GAP_SECONDS
            and saved.get("last_observation_healthy") is True
            and saved.get("queue_empty") is True
            and not saved.get("unregistered_jobs_seen")
        ):
            raise RuntimeError("Verified downloads and a fresh keep-warm window of empty queues are required")
        if reason == "failure_idle":
            return self._delete_failed()
        saved.update(delete_intent=reason, phase="delete_requested")
        self.persist()
        self._api().delete(saved["pod_id"])
        if self._owned_pod() is None:
            saved.update(retired=True, billing_confirmed_stopped=True, phase="retired")
            saved.setdefault("session_completed_at", datetime.now(UTC).isoformat())
            self.persist()
            from .global_registry import remove
            remove(self.folder)
            self._record_cost()

    def _delete_failed(self):
        """Failure-policy retirement: persist intent, DELETE, confirm absence with GET.

        An uncertain DELETE is never reported as retirement; a later poll retries
        the same delete (never a create).
        """
        from .cost_safety import alert
        saved = self.saved
        now = time.monotonic()
        failure = saved.get("owner_failure")
        minutes = self._failure_minutes()
        if not (failure and minutes and saved.get("failure_eligible") is True
                and saved.get("idle_process") == PROCESS_ID
                and saved.get("failure_idle_since") is not None
                and now - saved["failure_idle_since"] >= 60 * minutes
                and now - saved.get("last_observed_monotonic", 0) <= POLL_GAP_SECONDS
                and not saved.get("unregistered_jobs_seen")
                and (self._startup_failure_idle() or (
                    saved.get("last_observation_healthy") is True and saved.get("queue_empty") is True
                    and not saved.get("attention")))):
            raise RuntimeError("Failure shutdown requires a recorded failure and a fresh idle grace period")
        jobs = saved.get("jobs", {})
        saved["failure_summary"] = {
            "kind": failure.get("kind"), "failure_at": failure.get("at"),
            "failed_jobs": sorted(k for k, j in jobs.items() if j.get("status") == "error"),
            "succeeded_jobs": sorted(k for k, j in jobs.items() if j.get("status") == "success"),
            "collected_outputs": sum(len(j.get("outputs", [])) for j in jobs.values()
                                     if j.get("status") == "success"),
            "lost_outputs": [], "idle_minutes": minutes,
        }
        saved.update(delete_intent="failure_idle", phase="delete_requested",
                     delete_requested_at=datetime.now(UTC).isoformat())
        self.persist()  # Durable intent before the paid-resource call.
        try:
            self._api().delete(saved["pod_id"])
            gone = self._owned_pod() is None
        except (ProviderError, RuntimeError, httpx.HTTPError):
            gone = False
        if not gone:
            saved["delete_uncertain_at"] = datetime.now(UTC).isoformat()
            alert(self.state, self.settings, "delete_uncertain",
                  "The pod was still present after the shutdown request; it will be retried.")
            self.persist()
            return
        summary = saved["failure_summary"]
        saved.update(retired=True, billing_confirmed_stopped=True, phase="retired")
        saved.setdefault("session_completed_at", datetime.now(UTC).isoformat())
        alert(self.state, self.settings, "failure_retired",
              f"{len(summary['failed_jobs'])} job(s) failed, {summary['collected_outputs']} output(s) "
              "collected, no outputs lost.")
        self.persist()
        from .global_registry import remove
        remove(self.folder)
        self._record_cost()

    def _record_cost(self):
        if self.state.get("run_id"):
            try:
                from .resources import record_run
                record_run(self.folder, self.state)
            except (OSError, ValueError):
                self.saved.setdefault("alerts", []).append("local_cost_history_write_failed")
                self.persist()

    def shutdown(self):
        self.status()  # Reconcile an uncertain create before deciding whether to delete.
        if self.saved.get("create_attempted") and not self.saved.get("pod_id"):
            raise CreateUncertain("Owned GPU creation is unresolved; no unrelated pod was terminated")
        if self.saved.get("pod_id") and not self.retired:
            self._delete_owned("explicit_user_shutdown")
        return self.status()

    def download(self, descriptor, target, cancel, on_progress):
        raise RuntimeError("Global GPU outputs were on disposable storage. Only previously downloaded local files can be recovered after termination.")

    def close(self):
        if self._provider is not None:
            self._provider.client.close()
        if self._monitor is not None:
            self._monitor.client.close()
