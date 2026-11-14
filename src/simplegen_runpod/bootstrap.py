"""Desktop CPU bootstrap and explicitly requested recovery, with durable handoff."""

from __future__ import annotations

import copy
import json
import time

from .control import Conflict, new_session, session_id
from .provider import ProviderError, finite_positive, identifier

REQUIRED_SECRETS = (
    "RUNPOD_API_KEY",
    "SG_ADMIN_TOKEN",
    "SG_S3_ACCESS_KEY",
    "SG_S3_SECRET_KEY",
    "SG_COMFY_USERNAME",
    "SG_COMFY_PASSWORD",
)


def _check_credentials(secrets):
    if (
        not all(
            isinstance(secrets.get(k), str) and secrets[k] for k in REQUIRED_SECRETS
        )
        or len(secrets["SG_ADMIN_TOKEN"]) < 32
    ):
        raise ValueError("Protected manager credentials are incomplete")


def bootstrap(ledger, provider, config, sid, secrets, clock=time.time):
    """Caller holds a project-local process lock. Desktop is sole writer until handoff."""
    config.validate()
    session_id(sid)
    try:
        s = ledger.get(sid)
    except KeyError:
        if not config.allow_provisioning:
            raise Conflict("Provisioning is disabled") from None
        s = new_session(sid, config, clock())
        ledger.save(s)
    if s.get("schema") != 2 or s.get("config") != config.public():
        raise Conflict(
            "Existing session uses a different policy; do not overwrite or adopt it"
        )
    cpu = s["resources"]["cpu"]
    if cpu["state"] == "active" or s["phase"] != "bootstrapping":
        # The manager owns all writes after the handoff, including after CPU loss.
        return s
    if cpu["state"] == "rejected":
        raise Conflict("CPU creation was rejected; manual resolution required")
    _check_credentials(secrets)
    if cpu["state"] == "reserved":
        if not config.allow_provisioning:
            raise Conflict("Provisioning is disabled")
        volume = provider.volume(config.volume_id)
        if volume.get("dataCenterId") != config.region:
            raise Conflict("Existing volume region mismatch")
        payload = copy.deepcopy(config.cpu_payload)
        rate = finite_positive(provider.quote(payload))
        if rate > config.cpu_max_hourly:
            raise Conflict("CPU quote exceeds selected hourly limit")
        payload.update(
            name=cpu["name"],
            dockerStartCmd=["simplegen-runpod", "manager"],
            env={
                **{k: secrets[k] for k in REQUIRED_SECRETS},
                "SG_SESSION_ID": sid,
                "SG_CONFIG_JSON": json.dumps(config.public(), separators=(",", ":")),
            },
        )
        cpu.update(state="creating", create_intent_at=clock(), quoted_hourly=rate)
        ledger.save(s)
        try:
            pod = provider.create(payload)
        except ProviderError as exc:
            if (
                exc.status is not None
                and 400 <= exc.status < 500
                and exc.status not in (408, 429)
            ):
                cpu["state"] = "rejected"
                ledger.save(s)
            raise Conflict(
                "CPU creation unresolved/rejected; reconcile this same session, never blindly retry POST"
            ) from None
        try:
            cpu["pod_id"] = identifier(pod["id"])
        except (ValueError, TypeError, KeyError):
            raise Conflict(
                "CPU creation response uncertain; reconcile this same session"
            ) from None
        ledger.save(s)
    if cpu["pod_id"] is None:
        matches = [p for p in provider.list() if p.get("name") == cpu["name"]]
        if len(matches) != 1:
            raise Conflict("CPU creation remains ambiguous; manual resolution required")
        cpu["pod_id"] = identifier(matches[0]["id"])
        ledger.save(s)
    pod = provider.get(cpu["pod_id"])
    if not pod or pod.get("name") != cpu["name"]:
        raise Conflict(
            "CPU identity is not yet confirmed; no replacement will be created"
        )
    try:
        rate = finite_positive(pod["costPerHr"])
        cpu["actual_hourly"] = rate
        if rate > config.cpu_max_hourly:
            raise ValueError()
    except (ValueError, TypeError, KeyError):
        s["alerts"].append("cpu_actual_price_unknown_or_over_cap")
    cpu.update(state="active", observed_present=True)
    s["phase"] = "awaiting_start"
    s["control_url"] = f"https://{cpu['pod_id']}-8097.proxy.runpod.net"
    s["updated_at"] = clock()
    ledger.save(s)  # Handoff: desktop writes again only through explicit recovery.
    return s


def recover(ledger, provider, config, sid, secrets, clock=time.time):
    """Explicit paid replacement after confirmed CPU absence; never automatic.

    Caller must hold the same project-local process lock as bootstrap. GET None
    must mean provider-confirmed absence, not an unavailable or stopped pod.
    The prior CPU's absence transfers single-writer ownership to this caller;
    bootstrap's final durable active revision hands it to the replacement CPU.
    """
    config.validate()
    session_id(sid)
    s = ledger.get(sid)
    if s.get("schema") != 2 or s.get("config") != config.public():
        raise Conflict("Recovery requires schema 2 and the existing session policy")
    if not config.allow_provisioning:
        raise Conflict("Provisioning is disabled")
    if (
        s.get("cancel_request") is not None
        or s.get("cleanup") is not None
        or s["phase"] in ("gpu_delete_requested", "cpu_delete_requested", "closed")
        or s["resources"]["cpu"]["state"] in ("deleting", "deleted")
    ):
        raise Conflict("A cancelling or retiring session cannot create a recovery CPU")
    _check_credentials(secrets)
    cpu = s["resources"]["cpu"]
    pending = s["phase"] == "bootstrapping"
    previous = s.get("previous_cpu") if pending else cpu
    if not isinstance(previous, dict) or previous.get("observed_present") is not True:
        raise Conflict("Recovery requires a previously observed controller CPU")
    try:
        previous_id = identifier(previous["pod_id"])
    except (KeyError, TypeError, ValueError):
        raise Conflict(
            "Recovery requires the previous controller CPU identity"
        ) from None
    generation = previous.get("generation", 1)
    if type(generation) is not int or generation < 1:
        raise Conflict("Invalid controller CPU generation")
    next_generation = generation + 1
    next_name = f"sg-{sid}-cpu-r{next_generation}"
    if pending and (
        cpu.get("generation") != next_generation
        or cpu.get("name") != next_name
        or cpu.get("state") not in ("reserved", "creating", "rejected")
        or "absence_confirmed_at" not in previous
    ):
        raise Conflict("Pending CPU recovery identity is inconsistent")
    # Do not catch provider failures or infer absence from runtime/heartbeat.
    if provider.get(previous_id) is not None:
        raise Conflict("Previous controller CPU still exists; recovery is not allowed")
    if not pending:
        now = clock()
        s["previous_cpu"] = {
            **copy.deepcopy(cpu),
            "generation": generation,
            "absence_confirmed_at": now,
        }
        s["resources"]["cpu"] = {
            "name": next_name,
            "pod_id": None,
            "state": "reserved",
            "observed_present": False,
            "generation": next_generation,
        }
        s["phase"] = "bootstrapping"
        s["updated_at"] = now
        s["idle_since"] = None
        s.pop("control_url", None)
        ledger.save(s)
    return bootstrap(ledger, provider, config, sid, secrets, clock=clock)
