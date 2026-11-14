"""Managed session bridge. Desktop owns bootstrap only; the CPU owns execution."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import time
import uuid
from dataclasses import replace
from pathlib import Path, PurePosixPath

from simplegen_runpod.client import Client
from simplegen_runpod.config import Config
from simplegen_runpod.provider import RunPod, gpu_is_stopped
from simplegen_runpod.storage import S3Ledger

from .agents import Cancelled
from .credentials import _exclusive_json, _private_directory, load_credentials
from .settings import atomic_json, user_dir


def _hash(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _json_bytes(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _descriptor(value: dict) -> tuple[str, str, str]:
    name, folder, kind = value.get("filename"), value.get("subfolder", ""), value.get("type", "output")
    if (not isinstance(name, str) or not name or PurePosixPath(name).name != name
            or not isinstance(folder, str) or "\\" in name or "\\" in folder
            or PurePosixPath(folder).is_absolute() or ".." in PurePosixPath(folder).parts
            or name in (".", "..") or kind != "output"):
        raise ValueError("Output must be a safe persistent ComfyUI output path.")
    return name, folder, kind


class ManagedSession:
    def __init__(self, settings, folder, state, persist, emit, cancel, detach):
        self.settings, self.folder, self.state = settings, Path(folder), state
        self.persist, self.emit, self.cancel, self.detach = persist, emit, cancel, detach
        self._credentials = None
        self._provider = self._s3 = self._ledger = self._api = None

    @property
    def saved(self):
        return self.state.setdefault("managed_session", {})

    @property
    def retired(self):
        return self.saved.get("retired", False)

    def _check(self):
        if self.detach.is_set():
            from .lifecycle import Detached
            raise Detached("Detached; the RunPod session continues independently.")
        if self.cancel.is_set():
            raise Cancelled("RunPod operation cancelled.")

    def _wait(self):
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            self._check()
            self.cancel.wait(min(0.2, max(0, deadline - time.monotonic())))

    def _raw_config(self):
        try:
            raw = json.loads(Path(self.settings.runpod_config_file).expanduser().read_text())
            if not isinstance(raw, dict):
                raise TypeError
            return raw
        except (OSError, ValueError, TypeError):
            raise ValueError("Cannot read RunPod configuration JSON.") from None

    def config(self):
        raw = self.saved.get("config")
        if raw is None:
            raw = self._raw_config()
            raw = raw.get("policy", raw).copy()
            raw.pop("output_prefix", None)
            raw.pop("gpu_tiers", None)
        try:
            config = Config(**raw)
            config.validate()
            return config
        except (ValueError, TypeError, AttributeError):
            raise ValueError("RunPod configuration is incomplete or incompatible.") from None

    def schema(self):
        """Freeze node capabilities locally before agent usage or provisioning."""
        target = self.folder / "managed-object-info.json"
        if self.saved.get("schema_sha256"):
            try:
                content = target.read_bytes()
                if _hash(content) != self.saved["schema_sha256"]:
                    raise ValueError
                schema = json.loads(content)
            except (OSError, ValueError):
                raise ValueError("Saved RunPod node schema is missing or changed.") from None
        else:
            try:
                raw = json.loads(Path(self.settings.runpod_schema_file).expanduser().read_text())
                schema = raw.get("object_info", raw)
            except (OSError, ValueError, TypeError, AttributeError):
                raise ValueError("Link a cached cloud ComfyUI node schema before generation.") from None
            if not isinstance(schema, dict) or not schema or not all(
                isinstance(k, str) and isinstance(v, dict) and isinstance(v.get("input"), dict)
                for k, v in schema.items()
            ):
                raise ValueError("Cached RunPod node schema is empty or incompatible.")
            if any(node not in schema for node in self.config().required_nodes):
                raise ValueError("Cached cloud schema lacks required configured nodes.")
            atomic_json(target, schema)
            self.saved["schema_sha256"] = _hash(target.read_bytes())
            self.persist()
        return schema

    def _secrets(self):
        if self._credentials is None:
            self._credentials = load_credentials(self.settings)
        return self._credentials

    def _clients(self):
        secrets = self._secrets()
        config = self.config()
        if self._provider is None:
            self._provider = RunPod(secrets["RUNPOD_API_KEY"])
        if self._s3 is None:
            import boto3
            from botocore.config import Config as BotoConfig
            self._s3 = boto3.client(
                "s3", endpoint_url=f"https://s3api-{config.region.lower()}.runpod.io",
                region_name=config.region.lower(), aws_access_key_id=secrets["SG_S3_ACCESS_KEY"],
                aws_secret_access_key=secrets["SG_S3_SECRET_KEY"],
                config=BotoConfig(signature_version="s3v4", connect_timeout=5, read_timeout=10,
                                  retries={"total_max_attempts": 1, "mode": "standard"},
                                  s3={"addressing_style": "path"},
                                  request_checksum_calculation="when_required",
                                  response_checksum_validation="when_required"),
            )
        if self._ledger is None:
            self._ledger = S3Ledger(self._s3, config.volume_id, session_id=self.saved["sid"])

    def _remember(self, remote):
        if remote.get("id") != self.saved["sid"] or remote.get("schema") != 2:
            raise RuntimeError("Manager returned an incompatible session identity.")
        self.saved["remote"] = copy.deepcopy(remote)
        self.saved["control_url"] = remote.get("control_url", self.saved.get("control_url"))
        self.persist()
        return remote

    def _client(self):
        if self._api is None:
            url = self.saved.get("control_url")
            if not url:
                raise RuntimeError("RunPod controller is not yet connected.")
            self._api = Client(url, self._secrets()["SG_ADMIN_TOKEN"])
        return self._api

    def connect(self, choice=None):
        self._check()
        if not self.saved.get("sid"):
            raise ValueError("Regional CPU-manager sessions are deprecated. Select the Global Volume desktop profile for a new run.")
        self.schema()
        config = self.config()
        if not self.saved.get("sid"):
            if not getattr(self.settings, "managed_runpod", False) or not config.allow_provisioning or not choice:
                raise ValueError("Choose and approve RunPod compute before provisioning.")
            choice = {key: copy.deepcopy(choice[key]) for key in (
                "gpu_id", "gpu_count", "hourly_usd", "volume_id", "data_center", "cpu_hourly_usd"
            ) if key in choice}
            gpu_id = choice.get("gpu_id", config.gpu_payload["gpuTypeIds"][0])
            gpu_count = choice.get("gpu_count", config.gpu_payload["gpuCount"])
            if (not isinstance(gpu_id, str) or not gpu_id or type(gpu_count) is not int or gpu_count < 1
                    or choice.get("volume_id", config.volume_id) != config.volume_id
                    or choice.get("data_center", config.region) != config.region):
                raise ValueError("Selected RunPod compute does not match the configured volume and region.")
            gpu = {**config.gpu_payload, "gpuTypeIds": [gpu_id], "gpuCount": gpu_count}
            config = replace(config, gpu_payload=gpu)
            config.validate()
            cap = choice.get("hourly_usd", config.gpu_max_hourly)
            if type(cap) not in (int, float) or not 0 < cap <= config.gpu_max_hourly:
                raise ValueError("Accepted RunPod rate must be within the configured hourly limit.")
            config = replace(config, gpu_max_hourly=cap)
            self._provider = self._provider or RunPod(self._secrets()["RUNPOD_API_KEY"])
            current = float(self._provider.quote(gpu))
            if not 0 < current <= cap:
                raise RuntimeError("Selected GPU price changed; choose compute again before provisioning.")
            raw = self._raw_config()
            prefix = raw.get("output_prefix", "comfy-h3/ComfyUI/output")
            if (not isinstance(prefix, str) or not prefix or "\\" in prefix
                    or PurePosixPath(prefix).is_absolute() or ".." in PurePosixPath(prefix).parts):
                raise ValueError("RunPod output_prefix must be relative to the existing network volume.")
            self.saved.update(sid=uuid.uuid4().hex, config=config.public(), choice=choice,
                              output_prefix=prefix, bootstrap_intent=True)
            self.persist()
        self._clients()
        if not self.saved.get("control_url"):
            raise RuntimeError("Legacy CPU manager control URL is missing. CPU creation and recovery are deprecated; inspect the saved regional session manually.")
        deadline = time.monotonic() + self.settings.generation_timeout
        last_note = 0
        while time.monotonic() < deadline:
            self._check()
            try:
                remote = self.status()
                if self.retired:
                    pod_id = remote["resources"]["gpu"].get("pod_id")
                    if pod_id:
                        return self._endpoint(pod_id)
                    raise ValueError("RunPod controller retired before a GPU was recorded.")
                if not remote.get("activation_requested"):
                    raise RuntimeError("Regional CPU-manager provisioning is deprecated; existing sessions may only be inspected or shut down.")
                self._remember(remote)
                if remote.get("cleanup") or remote.get("cancel_request"):
                    raise ValueError("RunPod session is retiring; reconnect to recover its saved outputs.")
                if remote.get("phase") == "ready":
                    gpu = remote["resources"]["gpu"]
                    return self._endpoint(gpu["pod_id"])
            except RuntimeError:
                # Read-only/status or idempotent start reconciliation; never a prompt POST.
                pass
            if time.monotonic() - last_note > 15:
                self.emit({"stage": "cloud", "message": "RunPod compute is starting; paid billing may be active.", "progress": None})
                last_note = time.monotonic()
            self._wait()
        raise RuntimeError("RunPod startup needs attention; compute was retained. Reconnect to the saved session.")

    def _endpoint(self, pod_id):
        if not isinstance(pod_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", pod_id):
            raise ValueError("Invalid RunPod GPU identity.")
        secrets = self._secrets()
        login = {"username": secrets["SG_COMFY_USERNAME"], "password": secrets["SG_COMFY_PASSWORD"]}
        root = user_dir() / "runpod" / "logins"
        _private_directory(root)
        path = root / (_hash(_json_bytes(login)) + ".json")
        _exclusive_json(path, login)
        return replace(self.settings, comfy_url=f"https://{pod_id}-8189.proxy.runpod.net",
                       cloud=True, cloud_name="RunPod", token_env="", browser_login_file=str(path))

    def before_submit(self, job_id=None):
        self._check()
        job = self.saved.setdefault("job_intents", {}).setdefault(job_id, {}) if job_id else self.saved
        if job.get("submission_intent"):
            return False
        job["submission_intent"] = True
        self.persist()
        result = self._client().begin_job(self.saved["sid"], job_id or self.state["run_id"])
        self._remember(result)
        allowed = result.get("submission_allowed") is True
        job["submission_allowed"] = allowed
        self.persist()
        return allowed

    def bind(self, prompt_id, job_id=None):
        job = self.saved.setdefault("job_intents", {}).setdefault(job_id, {}) if job_id else self.saved
        previous = job.get("bind_prompt_id")
        if previous not in (None, prompt_id):
            raise ValueError("A different ComfyUI job is already bound to this session.")
        job["bind_prompt_id"] = prompt_id
        self.persist()
        if self.retired:
            remote = self.status()
            if remote.get("jobs", {}).get(job_id or self.state["run_id"], {}).get("prompt_id") != prompt_id:
                raise RuntimeError("Retired manager has not recorded this ComfyUI prompt identity.")
            return
        self._remember(self._client().bind_prompt(self.saved["sid"], job_id or self.state["run_id"], prompt_id))

    def status(self):
        if not self.saved.get("sid"):
            return {"phase": "not_started"}
        try:
            return self._remember(self._client().status(self.saved["sid"]))
        except RuntimeError:
            self._clients()
            self._ledger.reload()
            remote = self._ledger.get(self.saved["sid"])
            presence = {}
            for role in ("cpu", "gpu"):
                pod_id = remote["resources"][role].get("pod_id")
                if pod_id:
                    pod = self._provider.get(pod_id)
                    presence[role] = "absent" if pod is None else ("stopped" if gpu_is_stopped(pod) else "present")
                else:
                    presence[role] = "unconfirmed"
            self.saved["retired"] = presence["cpu"] == "absent" or presence["gpu"] in ("absent", "stopped")
            remote["provider_presence"] = presence
            remote["read_only_fallback"] = True
            return self._remember(remote)

    def complete(self, downloads):
        remote = self.status()
        manifest = remote.get("manifest", [])
        by_path = {_descriptor(item): item for item in manifest}
        if not manifest or len(by_path) != len(manifest) or len(downloads) != len(manifest):
            raise RuntimeError("Manager manifest does not yet match all completed videos; reconnect to acknowledge downloads.")
        receipts = []
        seen = set()
        for download in downloads:
            key = _descriptor(download)
            if key not in by_path or key in seen:
                raise ValueError("Download receipts must match every manager video exactly once.")
            seen.add(key)
            path = Path(download["path"])
            digest = hashlib.sha256()
            size = 0
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    self._check()
                    digest.update(chunk)
                    size += len(chunk)
            if size <= 0 or size != download["size_bytes"] or digest.hexdigest() != download["sha256"]:
                raise ValueError("A downloaded video changed before its completion receipt.")
            receipts.append({"id": by_path[key]["id"], "sha256": digest.hexdigest(), "size_bytes": size})
        intent = {"manifest_sha256": remote["manifest_sha256"], "receipts": receipts}
        if self.saved.get("completion_intent") not in (None, intent):
            raise ValueError("Completion receipt changed; retain the original session for recovery.")
        self.saved["completion_intent"] = intent
        self.persist()
        expected = {"manifest_sha256": intent["manifest_sha256"], "files": sorted(receipts, key=lambda r: r["id"])}
        if remote.get("download_receipt") == expected and remote.get("downloads_verified"):
            self.saved["completion_acknowledged"] = True
            self.persist()
            return remote
        if remote.get("read_only_fallback") and remote.get("provider_presence", {}).get("gpu") in ("absent", "stopped"):
            self.saved["completion_disposition"] = "saved_locally_after_compute_retired"
            self.saved["completion_acknowledged"] = False
            self.persist()
            return remote
        if remote.get("read_only_fallback"):
            raise RuntimeError("Videos are saved locally; controller is unavailable to acknowledge downloads. Reconnect or recover it manually.")
        result = self._client().complete(self.saved["sid"], downloads_verified=True, **intent)
        self._remember(result)
        self.saved["completion_acknowledged"] = True
        self.persist()
        return result

    def shutdown(self):
        remote = self.status()
        if remote.get("phase") == "not_started":
            return remote
        if remote.get("read_only_fallback"):
            return remote  # No blind provider DELETE or replacement to perform cleanup.
        if not self.saved.get("cancel_request_id"):
            self.saved["cancel_request_id"] = uuid.uuid4().hex
            self.persist()
        return self._remember(self._client().cancel(self.saved["sid"], request_id=self.saved["cancel_request_id"]))

    def download(self, descriptor, target, cancel, on_progress):
        """Read verified manifest outputs from persistent network-volume S3."""
        self._check()
        name, folder, _ = _descriptor(descriptor)
        remote = self.status()
        if (name, folder, "output") not in {_descriptor(item) for item in remote.get("manifest", [])}:
            raise ValueError("Requested video is not in the saved manager manifest.")
        self._clients()
        prefix = self.saved["output_prefix"]
        key = str(PurePosixPath(prefix) / folder / name)
        target = Path(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_name(target.name + ".part")
        body = None
        try:
            response = self._s3.get_object(Bucket=self.config().volume_id, Key=key)
            body = response["Body"]
            expected = response.get("ContentLength")
            if type(expected) is not int or expected <= 0:
                raise ValueError("Persistent video has an invalid size.")
            digest, size = hashlib.sha256(), 0
            with partial.open("wb") as handle:
                while chunk := body.read(1024 * 1024):
                    self._check()
                    if cancel.is_set():
                        raise Cancelled("Video download cancelled.")
                    handle.write(chunk)
                    digest.update(chunk)
                    size += len(chunk)
                    on_progress(size, expected)
                handle.flush()
                os.fsync(handle.fileno())
            if size != expected:
                raise ValueError("Persistent video download was incomplete.")
            partial.replace(target)
        except (Cancelled, ValueError):
            raise
        except Exception:  # noqa: BLE001 -- provider SDK errors must become one retryable failure
            from .lifecycle import Detached
            if self.detach.is_set():
                raise Detached("Detached during video download.") from None
            raise RuntimeError("Persistent video download failed; reconnect to retry. Cloud files were retained.") from None
        finally:
            if body is not None:
                body.close()
            partial.unlink(missing_ok=True)
        return {"path": str(target), "sha256": digest.hexdigest(), "size_bytes": size}

    def close(self):
        if self._api:
            self._api.close()
        if self._provider and hasattr(self._provider, "client"):
            self._provider.client.close()
        if self._s3 and hasattr(self._s3, "close"):
            self._s3.close()
