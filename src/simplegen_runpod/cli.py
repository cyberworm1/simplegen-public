"""Manual bootstrap and a disposable, supervised session manager entry point."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import time
from pathlib import Path


def _config(path=None):
    from .config import Config

    raw = Path(path).read_text() if path else os.environ["SG_CONFIG_JSON"]
    cfg = Config(**json.loads(raw))
    cfg.validate()
    return cfg


def main():
    parser = argparse.ArgumentParser(
        description="One disposable CPU manager per simpleGen session"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser(
        "validate-config", help="Offline validation, no cloud requests"
    )
    validate.add_argument("path")
    for command, help_text in (
        ("bootstrap", "PAID: explicitly create one session CPU; never a GPU"),
        ("recover", "PAID: manual replacement only after old CPU confirmed absent"),
        (
            "inspect",
            "Read-only durable state and current pod presence, no compute changes",
        ),
    ):
        entry = commands.add_parser(command, help=help_text)
        entry.add_argument("--config", required=True)
        entry.add_argument("--session-id", required=True)
        if command != "inspect":
            entry.add_argument(
                "--project",
                required=True,
                help="Original project folder holds bootstrap/recovery lock",
            )
    commands.add_parser(
        "manager", help="Run inside the disposable CPU; no independent host"
    )
    args = parser.parse_args()
    cfg = _config(
        args.path
        if args.command == "validate-config"
        else getattr(args, "config", None)
    )
    if args.command == "validate-config":
        print(
            "Configuration valid. Provisioning enabled:",
            cfg.allow_provisioning,
            "No time-based shutdown; files remain until manual cleanup.",
        )
        return
    from .control import Controller, session_id
    from .provider import ProviderError, RunPod, gpu_is_stopped, identifier
    from .storage import S3Ledger

    os.umask(0o077)
    sid = session_id(
        args.session_id if args.command != "manager" else os.environ["SG_SESSION_ID"]
    )
    if args.command == "inspect":
        ledger = S3Ledger.from_env(cfg.volume_id, cfg.region, session_id=sid)
        state = ledger.get(sid)
        provider = RunPod(os.environ["RUNPOD_API_KEY"])
        checks = {}
        for role, resource in state["resources"].items():
            if resource["pod_id"] is None:
                checks[role] = (
                    "unresolved"
                    if resource["state"] in ("creating", "unknown")
                    else "unallocated"
                )
                continue
            try:
                pod = provider.get(resource["pod_id"])
                if pod is None:
                    checks[role] = (
                        "absent_confirmed"
                        if resource.get("observed_present")
                        else "absence_unconfirmed"
                    )
                elif pod.get("name") != resource["name"]:
                    checks[role] = "ownership_mismatch"
                else:
                    checks[role] = "stopped" if gpu_is_stopped(pod) else "present"
            except ProviderError:
                checks[role] = "unknown"
        print(
            json.dumps({"durable_session": state, "provider_checks": checks}, indent=2)
        )
        return
    if args.command in ("bootstrap", "recover"):
        from .bootstrap import bootstrap, recover

        create_cpu = recover if args.command == "recover" else bootstrap

        project = Path(args.project).expanduser().resolve()
        project.mkdir(parents=True, exist_ok=True)
        with (project / ".manager-bootstrap.lock").open("a") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            descriptor = project / "managed-session.json"
            if descriptor.exists():
                previous = json.loads(descriptor.read_text())
                if previous["session_id"] != sid:
                    raise SystemExit("Project already belongs to another session")

            def save_descriptor(data):
                temporary = descriptor.with_suffix(".tmp")
                with temporary.open("w") as out:
                    json.dump(data, out, indent=2)
                    out.flush()
                    os.fsync(out.fileno())
                temporary.replace(descriptor)

            base = {
                "schema": 2,
                "session_id": sid,
                "volume_id": cfg.volume_id,
                "region": cfg.region,
            }
            save_descriptor(base)  # Before any paid POST; no credentials stored.
            ledger = S3Ledger.from_env(cfg.volume_id, cfg.region, session_id=sid)
            state = create_cpu(
                ledger, RunPod(os.environ["RUNPOD_API_KEY"]), cfg, sid, os.environ
            )
            save_descriptor(
                {
                    **base,
                    "cpu_pod_id": state["resources"]["cpu"]["pod_id"],
                    "control_url": state.get("control_url"),
                }
            )
            print(
                json.dumps(
                    {
                        "session_id": sid,
                        "phase": state["phase"],
                        "control_url": state.get("control_url"),
                    }
                )
            )
        return
    import uvicorn

    from .service import create_app
    from .worker import ComfyMonitor

    cpu_id = identifier(os.environ["RUNPOD_POD_ID"])
    # This local lock serializes process restarts on THIS pod. It is not a distributed lease.
    with open("/tmp/simplegen-manager.lock", "a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            try:
                ledger = S3Ledger.from_env(cfg.volume_id, cfg.region, session_id=sid)
                state = ledger.get(sid)
                if (
                    state["resources"]["cpu"]["state"] in ("active", "deleting")
                    and state["resources"]["cpu"]["pod_id"] == cpu_id
                ):
                    break
                if state["phase"] != "bootstrapping":
                    raise SystemExit(
                        "Manager identity/handoff mismatch; manual recovery required"
                    )
            except (KeyError, RuntimeError):
                pass
            print("waiting_for_durable_bootstrap_handoff; no_gpu_mutation", flush=True)
            time.sleep(10)
        controller = Controller(
            ledger,
            RunPod(os.environ["RUNPOD_API_KEY"]),
            cfg,
            os.environ["SG_ADMIN_TOKEN"],
            cpu_id,
            sid,
            monitor_factory=lambda pid: ComfyMonitor(
                pid,
                os.environ["SG_COMFY_USERNAME"],
                os.environ["SG_COMFY_PASSWORD"],
                cfg.required_nodes,
            ),
        )
        uvicorn.run(
            create_app(controller),
            host="0.0.0.0",
            port=8097,
            workers=1,
            access_log=False,
        )


if __name__ == "__main__":
    main()
