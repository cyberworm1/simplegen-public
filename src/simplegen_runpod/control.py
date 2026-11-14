"""One disposable session controller. Durable intent precedes every cloud mutation."""

from __future__ import annotations

import copy
import json
import os
import re
import sqlite3
import threading
import time
import uuid
from pathlib import Path

from .config import Config
from .outputs import digest, manifest
from .provider import ProviderError, finite_positive, gpu_is_stopped, identifier


class Conflict(RuntimeError):
    pass


def session_id(value: str) -> str:
    if uuid.UUID(value).hex != value:
        raise ValueError("Use a canonical lowercase UUID hex session ID")
    return value


class Ledger:
    """Local test/offline ledger. Production uses S3Ledger, not this disposable file."""

    def __init__(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, check_same_thread=False)
        os.chmod(path, 0o600)
        self.db.execute("PRAGMA journal_mode=DELETE")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS sessions (id TEXT PRIMARY KEY, body TEXT NOT NULL)"
        )
        self.db.commit()

    def get(self, sid):
        row = self.db.execute("SELECT body FROM sessions WHERE id=?", (sid,)).fetchone()
        if row is None:
            raise KeyError(sid)
        return json.loads(row[0])

    def all(self):
        return [
            json.loads(row[0]) for row in self.db.execute("SELECT body FROM sessions")
        ]

    def save(self, state):
        data = json.dumps(state, allow_nan=False, sort_keys=True)
        with self.db:
            self.db.execute(
                "INSERT INTO sessions VALUES (?, ?) ON CONFLICT(id) DO UPDATE SET body=excluded.body",
                (state["id"], data),
            )


def new_session(sid: str, config: Config, now: float) -> dict:
    config.validate()
    session_id(sid)
    return {
        "schema": 2,
        "id": sid,
        "config": config.public(),
        "created_at": now,
        "updated_at": now,
        "phase": "bootstrapping",
        "alerts": [],
        "activation_requested": False,
        "jobs": {},
        "resources": {
            role: {
                "name": f"sg-{sid}-{role}",
                "pod_id": None,
                "state": "reserved",
                "observed_present": False,
            }
            for role in ("cpu", "gpu")
        },
        "sealed": False,
        "downloads_verified": False,
        "download_receipt": None,
        "cancel_request": None,
        "cleanup": None,
        "observation": None,
        "observed_at": None,
        "idle_since": None,
        "manifest": [],
        "manifest_sha256": digest([]),
        "allocation_valid": False,
        "unregistered_jobs_seen": False,
        "unhealthy_since": None,
    }


class Controller:
    def __init__(
        self,
        ledger,
        provider,
        config: Config,
        secret: str,
        cpu_id: str,
        sid: str,
        monitor=None,
        clock=time.time,
        monotonic=time.monotonic,
        monitor_factory=None,
    ):
        config.validate()
        if len(secret) < 32:
            raise ValueError("Session secret must have at least 32 characters")
        self.ledger, self.provider, self.config = ledger, provider, config
        self.secret, self.cpu_id, self.sid = secret, identifier(cpu_id), session_id(sid)
        self.monitor, self.clock, self.monotonic = monitor, clock, monotonic
        self.monitor_factory = monitor_factory
        self.lock = threading.RLock()
        self.last_tick = 0.0
        self.idle_start = self.last_observation = None
        self._state()  # Reject legacy/wrong-owner state before doing any work.

    def _state(self):
        s = self.ledger.get(self.sid)
        if (
            s.get("schema") != 2
            or s.get("id") != self.sid
            or s.get("config") != self.config.public()
            or s["resources"]["cpu"]["pod_id"] != self.cpu_id
        ):
            raise Conflict(
                "Session schema/config/controller identity mismatch; manual recovery required"
            )
        return s

    def _save(self, s):
        s["updated_at"] = self.clock()
        self.ledger.save(s)

    def _alert(self, s, reason):
        if reason not in s["alerts"]:
            s["alerts"].append(reason)
        self.idle_start = None
        s["idle_since"] = None
        if not s["cleanup"] and not s["cancel_request"]:
            s["phase"] = "attention"

    def status(self, sid):
        if session_id(sid) != self.sid:
            raise KeyError(sid)
        with self.lock:
            s = self._state()
            # Persisted wall-clock timestamps alone never prove fresh monitoring.
            s["monitor_fresh"] = (
                self.last_observation is not None
                and self.monotonic() - self.last_observation
                <= 3 * self.config.poll_seconds
            )
            s["idle_remaining_seconds"] = (
                max(0, 300 - int(self.monotonic() - self.idle_start))
                if self.idle_start is not None
                else None
            )
            return s

    def start(self, sid):
        with self.lock:
            s = self.status(sid)
            s.pop("monitor_fresh", None)
            s.pop("idle_remaining_seconds", None)
            if s["activation_requested"] or s["cancel_request"] or s["cleanup"]:
                return s
            if not self.config.allow_provisioning:
                raise Conflict("Provisioning is disabled")
            cpu = self.provider.get(self.cpu_id)
            if not cpu or cpu.get("name") != s["resources"]["cpu"]["name"]:
                raise Conflict("CPU identity cannot be verified")
            if not self._check_rate(s, "cpu", cpu):
                self._save(s)
                raise Conflict("CPU allocation needs attention before GPU provisioning")
            volume = self.provider.volume(self.config.volume_id)
            if volume.get("dataCenterId") != self.config.region:
                raise Conflict("Existing volume region mismatch")
            s["activation_requested"] = True
            payload = copy.deepcopy(self.config.gpu_payload)
            payload.update(
                name=s["resources"]["gpu"]["name"],
                dockerStartCmd=["python", "/workspace/comfy-h3/gpu_boot.py"],
            )
            try:
                rate = finite_positive(self.provider.quote(payload))
                if rate > self.config.gpu_max_hourly:
                    raise ValueError()
            except (ProviderError, ValueError, KeyError, TypeError, StopIteration):
                self._alert(s, "gpu_quote_unavailable_or_over_cap")
                s["phase"] = "attention"
                self._save(s)
                return s
            gpu = s["resources"]["gpu"]
            gpu.update(
                state="creating", create_intent_at=self.clock(), quoted_hourly=rate
            )
            s["phase"] = "starting"
            self._save(s)
            try:
                pod = self.provider.create(payload)
            except ProviderError as exc:
                # Classify only the POST, never a subsequent GET, as rejected.
                gpu["state"] = (
                    "rejected"
                    if exc.status is not None
                    and 400 <= exc.status < 500
                    and exc.status not in (408, 429)
                    else "unknown"
                )
                self._alert(
                    s,
                    "gpu_create_rejected"
                    if gpu["state"] == "rejected"
                    else "gpu_create_uncertain",
                )
            else:
                try:
                    gpu["pod_id"] = identifier(pod["id"])
                    gpu["state"] = "active"
                except (TypeError, ValueError, KeyError):
                    gpu["state"] = "unknown"
                    self._alert(s, "gpu_create_uncertain")
            self._save(s)
            return copy.deepcopy(s)

    def begin_job(self, sid, job_id):
        identifier(job_id)
        with self.lock:
            s = self._state_for(sid)
            if job_id in s["jobs"]:
                return {**s, "submission_allowed": False}
            if (
                s["sealed"]
                or s["cleanup"]
                or s["cancel_request"]
                or s["phase"] != "ready"
                or not self._healthy(s)
                or not s["allocation_valid"]
                or s["unregistered_jobs_seen"]
            ):
                raise Conflict("Session is not accepting submissions")
            s["jobs"][job_id] = {
                "prompt_id": None,
                "status": "submission_pending",
                "outputs": [],
            }
            self.idle_start = None
            s["idle_since"] = None
            self._save(s)
            return {**s, "submission_allowed": True}

    def _state_for(self, sid):
        if session_id(sid) != self.sid:
            raise KeyError(sid)
        return self._state()

    def bind_prompt(self, sid, job_id, prompt_id):
        identifier(prompt_id)
        with self.lock:
            s = self._state_for(sid)
            job = s["jobs"][job_id]
            if job["prompt_id"] not in (None, prompt_id):
                raise Conflict("Different prompt already bound")
            if job["prompt_id"] == prompt_id:
                return s
            job.update(prompt_id=prompt_id, status="submitted")
            self._save(s)
            return s

    def seal(self, sid, downloads_verified, manifest_sha256, receipts):
        """Authenticated app attests verified local files; Boolean alone is insufficient."""
        with self.lock:
            s = self._state_for(sid)
            if (
                downloads_verified is not True
                or manifest_sha256 != s["manifest_sha256"]
                or not s["manifest"]
            ):
                raise Conflict(
                    "Verified receipts for the current complete video manifest are required"
                )
            if (
                s["unregistered_jobs_seen"]
                or not s["jobs"]
                or any(
                    j["status"] != "success" or not j["prompt_id"] or not j["outputs"]
                    for j in s["jobs"].values()
                )
            ):
                raise Conflict(
                    "Every registered job must have successful, recorded video outputs"
                )
            expected = {item["id"] for item in s["manifest"]}
            if not isinstance(receipts, list) or len(receipts) != len(expected):
                raise Conflict("Receipts must cover every required video exactly once")
            found = set()
            for r in receipts:
                if (
                    not isinstance(r, dict)
                    or set(r) != {"id", "sha256", "size_bytes"}
                    or r["id"] not in expected
                    or r["id"] in found
                    or not isinstance(r["sha256"], str)
                    or not re.fullmatch(r"[a-f0-9]{64}", r["sha256"])
                    or type(r["size_bytes"]) is not int
                    or r["size_bytes"] <= 0
                ):
                    raise Conflict("Invalid or duplicate verified file receipt")
                found.add(r["id"])
            receipt = {
                "manifest_sha256": manifest_sha256,
                "files": sorted(receipts, key=lambda r: r["id"]),
            }
            if s["sealed"]:
                if s["download_receipt"] != receipt:
                    raise Conflict(
                        "Completion already acknowledged with different receipts"
                    )
                return s
            s.update(
                sealed=True,
                downloads_verified=True,
                download_receipt=receipt,
                idle_since=None,
            )
            self.idle_start = None
            self._save(s)
            return s

    def cancel(self, sid, request_id):
        """Intentional manual override. Retains all network-volume files."""
        identifier(request_id)
        with self.lock:
            s = self._state_for(sid)
            if s["cancel_request"] is None:
                s["cancel_request"] = {
                    "id": request_id,
                    "at": self.clock(),
                    "kind": "explicit_user_shutdown",
                }
                s["sealed"] = True
                self._save(s)
            return s

    def _healthy(self, s):
        o = s.get("observation") or {}
        return (
            self.last_observation is not None
            and 0
            <= self.monotonic() - self.last_observation
            <= 3 * self.config.poll_seconds
            and all(
                o.get(k) is True
                for k in ("ready", "healthy", "auth_verified", "storage_ok")
            )
        )

    def observe(self, observation):
        """Internal monitor only; no public endpoint accepts fabricated idle evidence."""
        with self.lock:
            s = self._state()
            now = self.monotonic()
            if (
                self.last_observation is None
                or not 0 <= now - self.last_observation <= 3 * self.config.poll_seconds
            ):
                self.idle_start = None
            self.last_observation = now
            s["observed_at"] = self.clock()
            s["observation"] = {
                k: observation.get(k)
                for k in (
                    "ready",
                    "healthy",
                    "auth_verified",
                    "storage_ok",
                    "queue_empty",
                )
            }
            if self._healthy(s):
                s["unhealthy_since"] = None
            elif s.get("unhealthy_since") is None:
                s["unhealthy_since"] = self.clock()
            for job in s["jobs"].values():
                result = observation.get("jobs", {}).get(job["prompt_id"])
                # Persist completed histories so a Comfy restart cannot erase saved outputs.
                if (
                    result in ("error", "running", "missing", "success")
                    and job["status"] != "success"
                ):
                    job["status"] = result
                outputs = observation.get("outputs", {}).get(job["prompt_id"])
                if (
                    outputs is not None
                    and job["status"] == "success"
                    and not s["sealed"]
                ):
                    job["outputs"] = outputs
            s["manifest"] = manifest(s["jobs"])
            s["manifest_sha256"] = digest(s["manifest"])
            if observation.get("unregistered_jobs") is True:
                s["unregistered_jobs_seen"] = True
                self._alert(s, "unregistered_comfy_jobs_need_manual_attention")
            if observation.get("fatal") is True:
                self._alert(s, "comfy_readiness_failed")
            if any(j["status"] == "error" for j in s["jobs"].values()):
                self._alert(s, "generation_failed")
            if (
                self._healthy(s)
                and s["allocation_valid"]
                and not s["unregistered_jobs_seen"]
                and not s["cleanup"]
                and not s["cancel_request"]
            ):
                s["phase"] = "ready"
            eligible = (
                self._healthy(s)
                and not s["unregistered_jobs_seen"]
                and s["allocation_valid"]
                and s["downloads_verified"]
                and s["sealed"]
                and s["download_receipt"] is not None
                and s["download_receipt"]["manifest_sha256"] == s["manifest_sha256"]
                and observation.get("queue_empty") is True
                and bool(s["jobs"])
                and all(
                    j["status"] == "success" and j["outputs"]
                    for j in s["jobs"].values()
                )
            )
            if eligible:
                if self.idle_start is None:
                    self.idle_start = now
                s["idle_since"] = self.clock() - (now - self.idle_start)
            else:
                self.idle_start = None
                s["idle_since"] = None
            self._save(s)
            return s

    def _check_rate(self, s, role, pod):
        try:
            rate = finite_positive(pod["costPerHr"])
            s["resources"][role]["actual_hourly"] = rate
            if rate > getattr(self.config, role + "_max_hourly"):
                raise ValueError()
            return True
        except (ValueError, TypeError, KeyError):
            self._alert(s, role + "_actual_price_unknown_or_over_cap")
            return False

    def _gpu(self, s):
        gpu = s["resources"]["gpu"]
        s["allocation_valid"] = False
        if gpu["pod_id"] is None and gpu["state"] in ("creating", "unknown"):
            matches = [p for p in self.provider.list() if p.get("name") == gpu["name"]]
            if len(matches) != 1:
                self._alert(s, "gpu_creation_unresolved")
                return "unknown", None
            gpu["pod_id"] = identifier(matches[0]["id"])
            self._save(s)
        if gpu["pod_id"] is None:
            return "unallocated", None
        pod = self.provider.get(gpu["pod_id"])
        if pod is None:
            if gpu["observed_present"]:
                gpu["state"] = "deleted"
                return "deleted", None
            self._alert(s, "gpu_not_yet_visible")
            return "unknown", None
        if pod.get("name") != gpu["name"]:
            self._alert(s, "gpu_ownership_mismatch")
            return "unknown", None
        gpu["observed_present"] = True
        gpu["state"] = "active"
        valid_rate = self._check_rate(s, "gpu", pod)
        if gpu_is_stopped(pod):
            gpu["state"] = "stopped"
            return "stopped", pod
        volume = pod.get("networkVolume") or {}
        region = (pod.get("machine") or {}).get("dataCenterId") or volume.get(
            "dataCenterId"
        )
        if (
            pod.get("networkVolumeId", volume.get("id")) != self.config.volume_id
            or region != self.config.region
        ):
            self._alert(s, "gpu_volume_or_region_mismatch")
            return "unknown", pod
        s["allocation_valid"] = valid_rate
        return ("running" if pod.get("desiredStatus") == "RUNNING" else "unknown"), pod

    def _delete_gpu(self, s, authorization):
        if authorization not in ("explicit_user_shutdown", "verified_downloads_idle"):
            raise Conflict("GPU deletion is not authorized")
        if authorization == "explicit_user_shutdown":
            if not s["cancel_request"]:
                raise Conflict("Missing explicit cancellation record")
        elif not (
            s["downloads_verified"]
            and s["download_receipt"]
            and self._healthy(s)
            and s["observation"].get("queue_empty") is True
            and self.idle_start is not None
            and self.monotonic() - self.idle_start >= 300
        ):
            raise Conflict(
                "Verified downloads and fresh five-minute idle evidence required"
            )
        gpu = s["resources"]["gpu"]
        pod = self.provider.get(gpu["pod_id"])
        if not pod or pod.get("name") != gpu["name"]:
            raise Conflict("Reconcile GPU identity before deletion")
        if gpu_is_stopped(pod):
            self._retire_cpu(s, "gpu_stopped_externally")
            return False
        s["cleanup"] = {"kind": authorization, "at": self.clock()}
        s["phase"] = "gpu_delete_requested"
        gpu["state"] = "deleting"
        self._save(s)
        if authorization == "verified_downloads_idle" and not self._healthy(s):
            self._alert(s, "idle_evidence_expired_before_deletion")
            self._save(s)
            return False
        self.provider.delete(gpu["pod_id"])
        if self.provider.get(gpu["pod_id"]) is None:
            gpu["state"] = "deleted"
            self._save(s)
            return True
        return False

    def _retire_cpu(self, s, reason):
        # Call only after a fresh GPU state check or confirmed deletion in this tick.
        if reason not in (
            "gpu_stopped_externally",
            "gpu_deleted_externally",
            "explicit_user_shutdown",
            "verified_downloads_idle",
        ):
            raise Conflict("CPU retirement is not authorized")
        cpu = s["resources"]["cpu"]
        pod = self.provider.get(self.cpu_id)
        if pod is None or pod.get("name") != cpu["name"]:
            raise Conflict("CPU ownership cannot be verified")
        s["cleanup"] = {"kind": reason, "at": self.clock()}
        s["phase"] = "cpu_delete_requested"
        cpu["state"] = "deleting"
        self._save(s)
        # Do not claim we can durably confirm our own death.
        self.provider.delete(self.cpu_id)

    def tick(self):
        with self.lock:
            s = self._state()
            cpu = self.provider.get(self.cpu_id)
            if not cpu or cpu.get("name") != s["resources"]["cpu"]["name"]:
                raise Conflict("Controller pod identity is not confirmed")
            kind, pod = self._gpu(s)
            if (
                self.clock() - s["created_at"] >= self.config.session_alert_seconds
                and "long_session_billing_continues" not in s["alerts"]
            ):
                s["alerts"].append("long_session_billing_continues")
            created = s["resources"]["gpu"].get("create_intent_at")
            if (
                created is not None
                and not self._healthy(s)
                and self.clock() - created >= self.config.startup_alert_seconds
            ):
                self._alert(s, "startup_or_monitoring_needs_attention")
            if (
                s.get("unhealthy_since") is not None
                and self.clock() - s["unhealthy_since"]
                >= self.config.observation_alert_seconds
            ):
                self._alert(s, "comfy_or_storage_unavailable_compute_retained")
            self._save(s)
            # Only authoritative provider stop intent or confirmed absence counts.
            if kind in ("stopped", "deleted"):
                self._retire_cpu(s, "gpu_" + kind + "_externally")
            elif s["cancel_request"]:
                if (
                    kind == "unallocated"
                    and s["resources"]["gpu"]["state"]
                    in (
                        "reserved",
                        "rejected",
                    )
                    or pod is not None
                    and self._delete_gpu(s, "explicit_user_shutdown")
                ):
                    self._retire_cpu(s, "explicit_user_shutdown")
                # Unknown allocation or identity needs manual reconciliation, never another POST.
            elif kind == "running":
                if self.monitor is None and self.monitor_factory is not None:
                    self.monitor = self.monitor_factory(s["resources"]["gpu"]["pod_id"])
                if self.monitor is not None:
                    observation = self.monitor.poll(s["jobs"])
                    observation["storage_ok"] = (
                        True  # Subsequent durable save must succeed.
                    )
                    self.observe(observation)
                    s = self._state()
                if (
                    self.idle_start is not None
                    and self.monotonic() - self.idle_start >= 300
                    and self._healthy(s)
                    and self._delete_gpu(s, "verified_downloads_idle")
                ):
                    self._retire_cpu(s, "verified_downloads_idle")
            self.last_tick = self.clock()
