"""Deterministic readiness checks that run before any agent or paid work.

Pure code: no agent calls, no provider lifecycle. A check that cannot run
(ComfyUI unreachable, schema cache missing, network mount timing out) reports
``unverified``, never ``pass``.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

from .settings import Settings

PASS, FAIL, UNVERIFIED = "pass", "fail", "unverified"
FS_TIMEOUT = 5.0
_MODEL_INPUTS = {"model", "model_name", "ckpt_name", "unet_name", "vae_name",
                 "clip_name", "clip_name1", "clip_name2", "clip_name3", "lora_name"}


_OUTSTANDING: dict[str, object] = {}
_OUTSTANDING_LOCK = threading.Lock()


class StillHung(TimeoutError):
    """An earlier probe of the same path has not returned; no new thread was started."""


def _bounded(operation: Callable, timeout: float, key: str | None = None):
    """Run a filesystem call that may hang on a vanished SFTP mount.

    A hung call cannot be killed, so at most one probe per ``key`` (a path) is
    outstanding: while it is still hung, later probes fail at once with
    ``StillHung`` instead of starting another thread.
    """
    with _OUTSTANDING_LOCK:
        previous = _OUTSTANDING.get(key) if key is not None else None
        if previous is not None and not previous.done():
            raise StillHung(key)
        pool = ThreadPoolExecutor(max_workers=1)
        future = pool.submit(operation)
        pool.shutdown(wait=False)
        if key is not None:
            _OUTSTANDING[key] = future
    try:
        return future.result(timeout=timeout)
    finally:
        if key is not None and future.done():
            with _OUTSTANDING_LOCK:
                if _OUTSTANDING.get(key) is future:
                    del _OUTSTANDING[key]


def _check(identity: str, status: str, message: str, **details) -> dict:
    return {"id": identity, "status": status, "message": message, "details": details}


def _refmod_check(settings: Settings, schema: dict | None) -> dict:
    """Card RefMods against ComfyUI's loader list; an unreadable folder or schema is unverified."""
    from . import h3scene_refmod as refmod
    try:
        problems, names, verified = refmod.availability(Path(settings.project_folder).expanduser().resolve(), schema)
    except (OSError, UnicodeDecodeError) as error:
        return _check("h3scene_refmods", UNVERIFIED, f"Asset cards could not be read for RefMods ({error}).")
    if problems:
        return _check("h3scene_refmods", FAIL, "RefMods need attention: " + "; ".join(problems))
    if not names:
        return _check("h3scene_refmods", PASS, "No asset card names a RefMod (add a '## RefMod' section); "
                      "this run will use pictures only.")
    listed = ", ".join(sorted(names))
    if not verified:
        return _check("h3scene_refmods", UNVERIFIED, f"RefMods not verified against ComfyUI: {listed}.")
    return _check("h3scene_refmods", PASS, f"RefMods installed in ComfyUI: {listed}.")


def schema_hash(object_info: dict) -> str:
    return hashlib.sha256(json.dumps(object_info, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def target_for(settings: Settings) -> str:
    return "cloud" if settings.cloud or settings.managed_runpod else "local"


def _folder(label: str, value: str, timeout: float) -> tuple[str, str]:
    if not value or not value.strip():
        return FAIL, f"{label} folder is not set."
    path = Path(value).expanduser()

    def probe():
        if not path.exists():
            return UNVERIFIED, f"{label} folder is unavailable (missing or not mounted): {path}"
        if not path.is_dir():
            return FAIL, f"{label} path is not a folder: {path}"
        if not os.access(path, os.R_OK | os.X_OK):
            return FAIL, f"{label} folder is not readable: {path}"
        with os.scandir(path) as entries:
            next(entries, None)
        return PASS, f"{label} folder is available."
    try:
        return _bounded(probe, timeout, str(path))
    except StillHung:
        return UNVERIFIED, f"{label} folder is still not responding (an earlier check has not returned): {path}"
    except TimeoutError:
        return UNVERIFIED, f"{label} folder did not respond within {timeout:g}s: {path}"
    except OSError as error:
        return UNVERIFIED, f"{label} folder could not be read ({type(error).__name__}): {path}"


def _choices(spec: list):
    expected, options = spec[0], spec[1] if len(spec) > 1 else {}
    if isinstance(expected, list):
        return expected
    if expected == "COMBO":
        return options.get("options")
    return None


def missing_models(graph: dict, object_info: dict) -> list[dict]:
    """Name every loader whose selected model the target does not offer."""
    from .workflows import _expanded_specs
    missing = []
    for node_id, node in graph.items():
        kind = node.get("class_type")
        if kind not in object_info:
            continue
        inputs = node.get("inputs", {})
        try:
            specs, _ = _expanded_specs(object_info[kind], inputs)
        except ValueError:
            continue
        for name, value in inputs.items():
            if name not in _MODEL_INPUTS or name not in specs or not isinstance(value, str):
                continue
            choices = _choices(specs[name])
            if choices is not None and value not in choices:
                missing.append({"node_id": str(node_id), "class_type": kind, "input": name,
                                "model": value, "available": [c for c in choices if isinstance(c, str)]})
    return missing


def _load_cloud_schema(settings: Settings, now: float, timeout: float) -> tuple[dict | None, dict]:
    path = Path(settings.runpod_schema_file).expanduser() if settings.runpod_schema_file else None
    if path is None:
        return None, {"error": "No cached cloud node schema is configured."}

    def read():
        raw = json.loads(path.read_text(encoding="utf-8"))
        return raw.get("object_info", raw) if isinstance(raw, dict) else raw, path.stat().st_mtime
    try:
        schema, modified = _bounded(read, timeout, str(path))
    except TimeoutError:
        return None, {"error": f"Cached cloud schema did not respond within {timeout:g}s: {path}"}
    except (OSError, ValueError) as error:
        return None, {"error": f"Cached cloud schema is missing or unreadable ({type(error).__name__}): {path}"}
    if not isinstance(schema, dict) or not schema or not all(
            isinstance(v, dict) and isinstance(v.get("input"), dict) for v in schema.values()):
        return None, {"error": f"Cached cloud schema is empty or incompatible: {path}"}
    age = max(0.0, now - modified)
    return schema, {"source": str(path), "schema_age_seconds": round(age),
                    "schema_modified_at": datetime.fromtimestamp(modified, UTC).isoformat()}


def _load_local_schema(settings: Settings, client, label: str = "Local") -> tuple[dict | None, dict]:
    """Read the configured endpoint's live schema (Local, or a manual cloud endpoint)."""
    from .comfy import Comfy
    try:
        comfy = Comfy(settings, client)
    except ValueError as error:
        return None, {"error": str(error)}
    try:
        schema = comfy.object_info()
    except Exception as error:  # noqa: BLE001 -- any transport failure means unverified
        return None, {"error": f"{label} ComfyUI node schema could not be read: {error}"}
    finally:
        if client is None:
            comfy.close()
    if not isinstance(schema, dict) or not schema:
        return None, {"error": f"{label} ComfyUI returned an empty node schema."}
    return schema, {"source": f"{settings.comfy_url.rstrip('/')}/object_info"}


def target_schema(settings: Settings, target: str | None = None, *, comfy_client=None,
                  fs_timeout: float = FS_TIMEOUT, now: float | None = None) -> tuple[dict | None, dict]:
    """The schema a request is checked against: the cached RunPod file for managed
    RunPod, else the configured endpoint's live ``/object_info``."""
    target = target or target_for(settings)
    if target == "cloud" and (settings.managed_runpod or not settings.cloud):
        return _load_cloud_schema(settings, time.time() if now is None else now, fs_timeout)
    return _load_local_schema(settings, comfy_client, "Cloud" if target == "cloud" else "Local")


def _age(seconds: int) -> str:
    if seconds < 3600:
        return f"{seconds // 60} min"
    if seconds < 172800:
        return f"{seconds / 3600:.1f} h"
    return f"{seconds / 86400:.1f} days"


def run_preflight(
    settings: Settings,
    *,
    template_path: Path | None = None,
    target: str | None = None,
    object_info: dict | None = None,
    comfy_client=None,
    resolve: Callable[[str], str] | None = None,
    fs_timeout: float = FS_TIMEOUT,
    clock: Callable[[], float] = time.time,
    compiled: bool = True,
    cached_references: bool = False,
    agent_needed: bool = True,
    upscale_client=None,
) -> dict:
    """Return a JSON-serializable pass/fail/unverified result with per-check detail.

    ``template_path`` defaults to the linked workflow; saved runs pass their
    snapshot. ``object_info`` overrides the schema source (tests, callers that
    already hold it). ``compiled`` (new runs, schema 5 saved runs) also requires
    the Phase 3 compiler to build the template; older saved runs skip that check.
    ``cached_references`` (accepted work with a complete, verified
    ``references.json``) skips the linked Project folder: it is only needed
    for new agent preparation.
    """
    from .agents import executable
    resolve = resolve or executable
    target = target or target_for(settings)
    checks: list[dict] = []
    now = clock()
    result = {"status": UNVERIFIED, "target": target, "template": "", "template_sha256": None,
              "schema_sha256": None, "schema_source": None, "schema_age_seconds": None,
              "checked_at": datetime.fromtimestamp(now, UTC).isoformat(), "checks": checks}

    # linked_folders
    from .builtin_workflows import is_builtin_template
    folders = [] if cached_references else [("Project", settings.project_folder)]
    if template_path is None and not is_builtin_template(settings.template_file, settings.template_folder):
        folders.append(("Workflow", settings.template_folder))
    folder_results = [(label, *_folder(label, value, fs_timeout)) for label, value in folders]
    worst = FAIL if any(s == FAIL for _, s, _ in folder_results) else (
        UNVERIFIED if any(s == UNVERIFIED for _, s, _ in folder_results) else PASS)
    problems = [m for _, s, m in folder_results if s != PASS]
    checks.append(_check("linked_folders", worst,
                         " ".join(problems) if problems else
                         "References are cached locally; the linked folders are not needed." if not folders
                         else "Linked folders are available.",
                         folders={label: {"path": value, "status": s, "message": m}
                                  for (label, value), (_, s, m) in zip(folders, folder_results, strict=True)}))

    # Template
    template = None
    template_problem: tuple[str, str] | None = None
    try:
        if template_path is None:
            from .engine import _template
            template_path = _bounded(lambda: _template(settings), fs_timeout, settings.template_folder)
        path = Path(template_path)
        result["template"] = str(path)
        payload = _bounded(path.read_bytes, fs_timeout, str(path))
        result["template_sha256"] = hashlib.sha256(payload).hexdigest()
        template = json.loads(payload)
        if not isinstance(template, dict):
            template, template_problem = None, (FAIL, "Workflow JSON is not a workflow object.")
    except TimeoutError:
        template_problem = (UNVERIFIED, f"Workflow did not respond within {fs_timeout:g}s.")
    except ValueError as error:
        workflow_folder = next((s for label, s, _ in folder_results if label == "Workflow"), PASS)
        template_problem = ((UNVERIFIED, "Workflow folder is unavailable; the workflow cannot be read.")
                            if workflow_folder == UNVERIFIED else (FAIL, f"Workflow is unreadable: {error}"))
    except OSError as error:
        template_problem = (UNVERIFIED, f"Workflow could not be read ({type(error).__name__}).")

    from . import h3scene_family as scene
    if template is not None and scene.is_scene_chain(template):
        problem = scene.settings_problem(settings, target) or (None if cached_references else scene.asset_problem(settings))
        checks.append(_check("h3scene_scope", FAIL if problem else PASS, problem or "Local H3 Scene Chain Qty 1 run."))
        try:
            tested = scene.envelope(template, settings)
            # Informational: the tested envelope must never require an override or block dispatch.
            checks.append(_check("h3scene_envelope", PASS, tested["message"],
                                 verification="verified" if tested["verified"] else "unverified"))
        except ValueError as error:
            checks.append(_check("h3scene_profile", FAIL, str(error)))

    # Target schema
    if object_info is not None:
        schema, meta = object_info, {"source": "supplied"}
    else:
        schema, meta = target_schema(settings, target, comfy_client=comfy_client, fs_timeout=fs_timeout, now=now)
    if schema is not None:
        result["schema_sha256"] = schema_hash(schema)
        result["schema_source"] = meta.get("source")
        result["schema_age_seconds"] = meta.get("schema_age_seconds")
    label = "Cloud" if target == "cloud" else "Local"

    # template_structure
    if template_problem:
        checks.append(_check("template_structure", *template_problem))
    elif schema is None:
        checks.append(_check("template_structure", UNVERIFIED,
                             f"Workflow structure not verified: {meta['error']}"))
    else:
        from .singleshot import is_single_shot
        from .singleshot import support_problem as single_shot_problem
        from .workflow_edits import edit_contract
        try:
            if scene.is_scene_chain(template):
                problem = scene.support_problem(template, schema, settings, target, cached_references=cached_references)
                checks.append(_check("template_structure", FAIL if problem else PASS,
                                     problem or "H3 Scene Chain workflow structure is supported."))
            elif is_single_shot(template):
                problem = single_shot_problem(copy.deepcopy(template), schema, settings, target)
                checks.append(_check("template_structure", FAIL if problem else PASS,
                                     f"Single-shot voice workflow is not supported: {problem}" if problem
                                     else "Single-shot voice workflow structure is supported."))
            else:
                edit_contract(copy.deepcopy(template), schema, megapixels=settings.megapixels,
                              duration=settings.duration, run_id="preflight", check_model_availability=False,
                              aspect_ratio=settings.aspect_ratio, steps=settings.steps)
                from .compiler import support_problem
                problem = support_problem(copy.deepcopy(template), schema, settings) if compiled else None
                if problem:
                    checks.append(_check("template_structure", FAIL,
                                         f"Workflow is not supported by the workflow compiler: {problem}"))
                else:
                    checks.append(_check("template_structure", PASS, "Workflow structure is supported."))
        except ValueError as error:
            checks.append(_check("template_structure", FAIL, f"Workflow is not supported: {error}"))
        except Exception as error:  # noqa: BLE001 -- a crashed check is not a pass
            checks.append(_check("template_structure", UNVERIFIED,
                                 f"Workflow structure check could not run: {type(error).__name__}"))

    # h3scene_refmods: every card RefMod must be installed before the agent may choose it.
    if template is not None and scene.is_refmod(template):
        checks.append(_refmod_check(settings, schema))

    # target_compatibility
    if template_problem:
        checks.append(_check("target_compatibility", template_problem[0],
                             "Target compatibility not checked because the workflow could not be read."))
    elif schema is None:
        checks.append(_check("target_compatibility", UNVERIFIED,
                             f"{label} compatibility not verified: {meta['error']}"))
    else:
        from .workflows import _flatten, validate_graph
        age = meta.get("schema_age_seconds")
        age_note = f" (cached schema is {_age(age)} old)" if age is not None else ""
        models: list[dict] = []
        try:
            graph, _ = _flatten(copy.deepcopy(template), schema)
            models = missing_models(graph, schema)
            validate_graph(graph, schema)
            checks.append(_check("target_compatibility", PASS,
                                 f"Workflow is compatible with {label}{age_note}.", **meta))
        except ValueError as error:
            if models:
                message = " ".join(
                    f"{m['class_type']} (node {m['node_id']}) {m['input']}: model '{m['model']}' "
                    f"is not available on {label}. Available: {', '.join(m['available']) or 'none'}."
                    for m in models)
            else:
                message = f"Workflow is not compatible with {label}: {error}"
            checks.append(_check("target_compatibility", FAIL, message + age_note,
                                 error=str(error), missing_models=models, **meta))
        except Exception as error:  # noqa: BLE001 -- a crashed check is not a pass
            checks.append(_check("target_compatibility", UNVERIFIED,
                                 f"{label} compatibility check could not run: {type(error).__name__}", **meta))

    # agent: the same rule build_command enforces at launch (agents.resolve_agent)
    if agent_needed:
        from .agents import agent_problem
        problem = agent_problem(settings)
        checks.append(_check("agent", FAIL if problem else PASS,
                             f"Agent settings can't run: {problem}" if problem else "Agent settings are valid.",
                             provider=settings.provider, tier=settings.model_tier,
                             model=settings.model, reasoning=settings.reasoning))
    else:
        checks.append(_check("agent", PASS, "No agent pass needed: accepted work is reused."))

    # tools
    missing_tools = {}
    for name in (settings.provider, "ffprobe"):
        try:
            resolve(name)
        except (ValueError, OSError) as error:
            missing_tools[name] = str(error)
    checks.append(_check("tools", FAIL if missing_tools else PASS,
                         " ".join(missing_tools.values()) if missing_tools
                         else "Agent CLI and ffprobe are installed.", missing=sorted(missing_tools)))

    # disk_space
    minimum = int(settings.preflight_min_free_gb * 1024**3)
    output = Path(settings.output_folder).expanduser() if settings.output_folder.strip() else None
    if output is None:
        checks.append(_check("disk_space", FAIL, "Output folder is not set."))
    else:
        def free():
            probe = output
            while not probe.exists() and probe.parent != probe:
                probe = probe.parent
            return shutil.disk_usage(probe).free
        try:
            available = _bounded(free, fs_timeout, str(output))
            ok = available >= minimum
            checks.append(_check(
                "disk_space", PASS if ok else FAIL,
                f"{available / 1024**3:.1f} GB free in the output folder"
                + ("." if ok else f"; at least {settings.preflight_min_free_gb:g} GB is required."),
                free_bytes=available, minimum_bytes=minimum))
        except (TimeoutError, OSError) as error:
            checks.append(_check("disk_space", UNVERIFIED,
                                 f"Free space in the output folder could not be read ({type(error).__name__}).",
                                 minimum_bytes=minimum))

    # 4K60 upscale: advisory only. The request is durable and waits for the host,
    # so an unavailable upscale service never blocks or delays the render.
    if getattr(settings, "upscale", False):
        from .upscale import preflight_check
        checks.append(preflight_check(settings, client=upscale_client))

    statuses = {c["status"] for c in checks if not c.get("advisory")}
    result["status"] = FAIL if FAIL in statuses else UNVERIFIED if UNVERIFIED in statuses else PASS
    return result


def run_saved_preflight(folder: Path, **kwargs) -> dict:
    """Check a saved request's own snapshot and settings (dispatch re-check)."""
    folder = Path(folder)
    state = json.loads((folder / "state.json").read_text(encoding="utf-8"))
    kwargs.setdefault("compiled", state.get("schema_version", 1) >= 5)
    kwargs.setdefault("agent_needed", not state.get("workflow_acceptance"))
    if "cached_references" not in kwargs:
        from .refcache import complete
        kwargs["cached_references"] = complete(folder, state)
    return run_preflight(Settings(**state["settings"]), template_path=folder / "template.original.json", **kwargs)


def summary(result: dict) -> str:
    """One user-facing line: the first failing check, else the first unverified one."""
    for status in (FAIL, UNVERIFIED):
        for check in result.get("checks", []):
            if check["status"] == status and not check.get("advisory"):
                return check["message"]
    return "Preflight passed."


def receipt(result: dict, *, override_unverified: bool = False) -> dict:
    return {"status": result["status"], "checked_at": result["checked_at"],
            "schema_sha256": result.get("schema_sha256"), "template_sha256": result.get("template_sha256"),
            "override_unverified": override_unverified, "result": result}
