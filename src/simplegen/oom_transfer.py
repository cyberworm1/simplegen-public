"""Copy a local OOM request into a new cloud project, without reusing execution state."""
from __future__ import annotations

import copy
import hashlib
import json
import re
from dataclasses import replace
from pathlib import Path

from .settings import Settings, atomic_json, with_generation_location


def is_oom_message(message: str) -> bool:
    return bool(re.search(r"out[ _-]*of[ _-]*memory|OutOfMemoryError|\bOOM\b", message, re.IGNORECASE))


def local_oom(folder: Path, state: dict) -> bool:
    if state.get("settings", {}).get("cloud") or state.get("settings", {}).get("managed_runpod"):
        return False
    if state.get("status") != "failed":
        return False
    if state.get("failure_kind") == "out_of_memory" or is_oom_message(state.get("error", "")):
        return True
    report = folder / "report.txt"
    if report.is_file() and is_oom_message(report.read_text(encoding="utf-8")):
        return True
    for child in state.get("sequence", {}).get("children", []):
        sub = folder / child["folder"]
        if local_oom(sub, json.loads((sub / "state.json").read_text())):
            return True
    return False


def cloud_settings(state: dict, current: Settings) -> Settings:
    """Keep the failed request's creative settings and current cloud configuration."""
    old = Settings(**state["settings"])
    connection = with_generation_location(current, True)
    workflow = Path(current.runpod_default_workflow).expanduser()
    if not workflow.is_absolute():
        workflow = Path(current.template_folder).expanduser() / workflow
    return replace(old, cloud=True, managed_runpod=True, global_desktop=True,
                   comfy_url="", token_env="", browser_login_file="", cloud_name="RunPod",
                   connection_profiles=connection.connection_profiles,
                   runpod_config_file=current.runpod_config_file,
                   runpod_credentials_file=current.runpod_credentials_file,
                   runpod_schema_file=current.runpod_schema_file,
                   runpod_gpu_tiers=current.runpod_gpu_tiers.copy(),
                   runpod_default_workflow=current.runpod_default_workflow,
                   runpod_default_model=current.runpod_default_model,
                   runpod_default_clip=current.runpod_default_clip,
                   template_folder=str(workflow.parent), template_file=str(workflow))


def create_cloud_copy(source: Path, settings: Settings, concept: str):
    """Snapshot the cloud template with explicit cloud loaders; never mutate templates."""
    from .engine import create_run, now
    source = source.expanduser().resolve()
    old = json.loads((source / "state.json").read_text())
    if not local_oom(source, old):
        raise ValueError("Choose a failed local run that ran out of memory.")
    if not settings.managed_runpod or not settings.global_desktop:
        raise ValueError("OOM transfer requires managed RunPod batch settings.")
    template = Path(settings.template_file).expanduser()
    if not template.is_absolute():
        template = Path(settings.template_folder).expanduser() / template
    workflow = copy.deepcopy(json.loads(template.read_text()))
    if not isinstance(workflow.get("nodes"), list):
        raise TypeError("Select a native RunPod H3 workflow in Settings.")
    updated = {"UNETLoader": 0, "CLIPLoader": 0}
    def patch(value):
        if isinstance(value, dict):
            kind = value.get("type")
            if isinstance(kind, str) and kind in updated:
                widgets = value.get("widgets_values")
                if not isinstance(widgets, list) or not widgets:
                    raise ValueError(f"Cannot locate {kind} model selector in cloud workflow.")
                model = settings.runpod_default_model if kind == "UNETLoader" else settings.runpod_default_clip
                if not model.strip():
                    raise ValueError("Set the RunPod default model and CLIP in Settings.")
                widgets[0] = model.strip()
                updated[kind] += 1
            for item in value.values():
                patch(item)
        elif isinstance(value, list):
            for item in value:
                patch(item)
    patch(workflow)
    if any(count != 1 for count in updated.values()):
        raise ValueError("Cloud transfer needs one UNETLoader and one CLIPLoader; choose the RunPod H3 template.")
    payload = json.dumps(workflow, indent=2, ensure_ascii=False).encode()
    folder, state = create_run(settings, concept)
    from .scheduling import atomic_bytes
    atomic_bytes(folder / "template.original.json", payload)
    state.update(template_path=str(template), template_sha256=hashlib.sha256(payload).hexdigest(),
                 forked_from={"run_id": old["run_id"], "folder": str(source), "at": now(),
                              "reason": "local_out_of_memory"})
    atomic_json(folder / "state.json", state)
    return folder, state
