"""A single persisted concept-to-video run, independent of the desktop UI."""

from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import math
import re
import subprocess
from collections.abc import Callable
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from threading import Event

from . import agents
from .agents import Cancelled, redact
from .comfy import VIDEO_SUFFIXES, Comfy, new_id, output_files
from .h3scene_compat import read_state
from .lifecycle import Detached, Interrupt
from .settings import Settings, atomic_json
from .workflows import prepare_workflow, validate_graph

STATE_VERSION = 5  # 5: agent writes a creative plan; code compiles the workflow.


def now() -> str:
    return datetime.now(UTC).isoformat()


@contextmanager
def run_lock(folder: Path):
    with (folder / ".run.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(
                "This project is already running in another simpleGen window."
            ) from None
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def probe_video(path: Path) -> dict:
    probe = agents.executable("ffprobe")
    completed = subprocess.run(
        [
            probe,
            "-v",
            "error",
            "-show_entries",
            "format=duration:stream=codec_type,width,height,duration",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if completed.returncode:
        raise RuntimeError("Saved final video could not be inspected by ffprobe.")
    data = json.loads(completed.stdout)
    video = next((s for s in data.get("streams", []) if s.get("codec_type") == "video"), None)
    if not video or not video.get("width") or not video.get("height"):
        raise RuntimeError("The final output does not contain a playable video stream.")
    duration = float(data.get("format", {}).get("duration", video.get("duration", 0)))
    if duration <= 0:
        raise RuntimeError("The final video has no positive duration.")
    return {
        "size_bytes": path.stat().st_size,
        "width": video["width"],
        "height": video["height"],
        "duration": duration,
    }


def _verified_receipts(folder: Path, state: dict) -> list[dict]:
    """Return manager receipts tied to the actual local downloaded files."""
    receipts = []
    descriptors = state.get("download_files", {})
    for key, receipt in state.get("downloads", {}).items():
        descriptor = descriptors.get(key)
        if not descriptor:
            raise RuntimeError("A downloaded video is missing its durable descriptor.")
        path = folder / receipt["path"]
        if not path.is_file() or not _downloaded(receipt, folder):
            raise RuntimeError("A required video is missing or failed download verification.")
        receipts.append({**descriptor, "prompt_id": state.get("prompt_id"), "path": str(path),
                         "size_bytes": receipt["size_bytes"],
                         **({"sha256": receipt["sha256"]} if "sha256" in receipt else {})})
    return receipts


def _history_from_manifest(status: dict, graph: dict, prompt_id: str) -> dict:
    """Build the minimal output history needed for resumed S3 fallback downloads."""
    outputs = {}
    for item in status.get("manifest", []):
        if item.get("prompt_id", prompt_id) != prompt_id:
            continue
        node = str(item.get("node_id", ""))
        if not node:
            remote_path = str(Path(item.get("subfolder", "")) / item["filename"])
            matches = [key for key, value in graph.items()
                       if value["class_type"] == "SaveVideo"
                       and remote_path.startswith(value["inputs"]["filename_prefix"] + "_")]
            if len(matches) != 1:
                raise RuntimeError("Saved output cannot be matched to its workflow video node.")
            node = matches[0]
        outputs.setdefault(node, {}).setdefault("videos", []).append({
            "filename": item["filename"], "subfolder": item.get("subfolder", ""),
            "type": item.get("type", "output"),
        })
    return {"status": {"completed": True, "status_str": "success"}, "outputs": outputs}


def _inside(path: str, roots: list[Path]) -> Path:
    source = Path(path).expanduser()
    if not source.is_absolute():
        source = roots[0] / source
    source = source.resolve()
    if not any(source.is_relative_to(root) for root in roots) or not source.is_file():
        raise ValueError(f"Reference is missing or outside the linked folders: {path}")
    if any(
        part.startswith(".")
        for part in source.relative_to(next(r for r in roots if source.is_relative_to(r))).parts
    ):
        raise ValueError("Hidden files are not production references.")
    return source


def _inventory(root: Path) -> list[str]:
    # Supply a cheap starting index, not the contents of the entire production.
    extensions = {".txt", ".md", ".pdf", ".docx", ".png", ".jpg", ".jpeg", ".webp",
                  ".wav", ".flac", ".mp3", ".m4a", ".ogg"}  # audio: voice samples for single-shot templates
    result = []
    for path in sorted(root.iterdir()):
        if path.name.startswith("."):
            continue
        if path.is_dir():
            result.append(path.name + "/")
        elif path.suffix.lower() in extensions:
            result.append(path.name)
        if len(result) >= 150:
            break
    return result


def _prompt_brief(slots: list[dict]) -> list[dict]:
    """Send the format, not repeated sample scenes, to the agent."""
    result = []
    for slot in slots:
        brief = {key: value for key, value in slot.items() if key != "text"}
        brief["sections"] = slot.get("sections") or re.findall(
            r"(?m)^([a-z_]+):", slot.get("text", "")
        )
        result.append(brief)
    return result


def apply_plan(
    prepared: dict, plan: dict, roots: list[Path]
) -> tuple[dict, list[tuple], list[str]]:
    if (
        not isinstance(plan, dict)
        or not isinstance(plan.get("script"), str)
        or not plan["script"].strip()
    ):
        raise ValueError("Agent returned no script.")
    graph = copy.deepcopy(prepared["graph"])
    expected = {(str(s["node_id"]), s["input"]) for s in prepared["prompt_slots"]}
    found: set = set()
    for replacement in plan.get("prompts", []):
        key = (str(replacement["node_id"]), replacement["input"])
        if key not in expected or key in found:
            raise ValueError("Agent returned an unknown or duplicate prompt slot.")
        text = replacement["text"]
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Agent returned an empty prompt.")
        if text.count("<d>") != text.count("</d>"):
            raise ValueError("Agent returned unbalanced H3 dialogue tags.")
        graph[key[0]]["inputs"][key[1]] = text
        found.add(key)
    if found != expected:
        raise ValueError("Agent did not write every required prompt; no generation was submitted.")
    expected_refs = {(str(s["node_id"]), s["input"]) for s in prepared["reference_slots"]}
    selected: set = set()
    uploads = []
    used: set[str] = set()
    for replacement in plan.get("references", []):
        key = (str(replacement["node_id"]), replacement["input"])
        if key not in expected_refs or key in selected:
            raise ValueError("Agent returned an unknown or duplicate reference slot.")
        source = _inside(replacement["path"], roots)
        used.add(str(source))
        uploads.append((key[0], key[1], source))
        selected.add(key)
    if selected != expected_refs:
        raise ValueError("Agent did not select a local file for every reference slot.")
    for item in plan.get("used_references", []):
        used.add(str(_inside(item, roots)))
    return graph, uploads, sorted(used)


def _downloaded(receipt: dict, folder: Path | None = None) -> bool:
    path = Path(receipt["path"])
    if folder is not None:
        path = (folder / path).resolve()
        if not path.is_relative_to(folder.resolve()):
            return False
    if not path.is_file() or path.stat().st_size != receipt["size_bytes"]:
        return False
    if receipt.get("verification") == "size":
        return receipt["size_bytes"] > 0
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest() == receipt["sha256"]


def _execution_seconds(history: dict) -> float | None:
    """Use ComfyUI execution timestamps, excluding time queued behind other work."""
    messages = history.get("status", {}).get("messages", [])
    stamps = {}
    for message in messages if isinstance(messages, list) else []:
        if not isinstance(message, (list, tuple)) or len(message) != 2 or not isinstance(message[1], dict):
            continue
        name, detail = message
        stamp = detail.get("timestamp")
        if name in ("execution_start", "execution_success") and type(stamp) in (int, float) and math.isfinite(stamp):
            stamps[name] = stamp
    start, finish = stamps.get("execution_start"), stamps.get("execution_success")
    if start is None or finish is None or finish < start:
        return None
    seconds = (finish - start) / (1000 if start > 100_000_000_000 else 1)
    return seconds if math.isfinite(seconds) and seconds >= 0 else None


def _template(settings: Settings) -> Path:
    from .builtin_workflows import builtin_name, is_builtin_template, resolve_builtin_template
    if is_builtin_template(settings.template_file, settings.template_folder):
        path = Path(resolve_builtin_template(settings.template_file, settings.template_folder)).expanduser().resolve()
        if not path.is_file():
            raise ValueError(f"The built-in {builtin_name(path)} workflow is unavailable; refresh the workflow list.")
        return path
    root = Path(settings.template_folder).expanduser().resolve()
    if settings.template_file:
        path = Path(settings.template_file).expanduser()
        if not path.is_absolute():
            path = root / path
    else:
        preferred = root / "H3_Persistent_References_Clonable_Chain.json"
        paths = sorted(root.glob("*.json"))
        path = preferred if preferred.is_file() else (paths[0] if paths else root / "missing.json")
    path = path.resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("Choose a workflow JSON in the linked template folder.")
    return path


def fork_uses_snapshot(settings: Settings, fork_seed: dict | None) -> bool:
    if not fork_seed or not fork_seed.get("template_sha256"):
        return False
    def full(folder, filename):
        path = Path(filename).expanduser()
        return (path if path.is_absolute() else Path(folder).expanduser() / path).resolve()
    return full(settings.template_folder, settings.template_file) == full(
        fork_seed["template_folder"], fork_seed["template_file"])


def create_run(settings: Settings, concept: str, fork_seed: dict | None = None) -> tuple[Path, dict]:
    """Reserve a durable project without invoking an agent or provider."""
    seeded_template = fork_uses_snapshot(settings, fork_seed)
    settings.validate(require_folders=not seeded_template)
    if seeded_template and (not Path(settings.project_folder).expanduser().is_dir()
                            or not settings.output_folder.strip()):
        raise ValueError("Choose an existing project folder and an output folder for the new run.")
    if not concept.strip():
        raise ValueError("Enter a concept first.")
    source_folder = None
    source_state = None
    payload = None
    sha = None
    if fork_seed:
        source_folder = Path(fork_seed["folder"]).expanduser().resolve()
        source_state = read_state(source_folder / "state.json")
        if source_state.get("run_id") != fork_seed["run_id"]:
            raise ValueError("The source run changed. Reopen it before making a new run.")
        if seeded_template:
            from .editing import read_json
            snapshot = source_folder / "template.original.json"
            if snapshot.is_symlink() or not snapshot.is_file():
                raise ValueError("The source workflow snapshot is missing.")
            payload = snapshot.read_bytes()
            sha = hashlib.sha256(payload).hexdigest()
            if sha != fork_seed["template_sha256"]:
                raise ValueError("The source workflow snapshot changed. Reopen the saved run.")
            read_json(snapshot)
    run_id = new_id()
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", concept.strip())[:45].strip("-") or "concept"
    folder = Path(settings.output_folder).expanduser().resolve() / (
        datetime.now(UTC).strftime("%Y%m%d-%H%M%S-") + slug + "-" + run_id[:8]
    )
    folder.mkdir(parents=True, exist_ok=False)
    state = {
        "schema_version": STATE_VERSION, "run_id": run_id, "client_id": new_id(),
        "created_at": now(), "settings": settings.public_dict(), "concept": concept,
        "status": "created", "downloads": {},
    }
    if fork_seed:
        state["forked_from"] = {"run_id": fork_seed["run_id"],
                                "folder": str(source_folder), "at": now()}
        if seeded_template:
            from .scheduling import atomic_bytes
            atomic_bytes(folder / "template.original.json", payload)
            state.update(template_path=source_state.get("template_path", settings.template_file),
                         template_sha256=sha)
    atomic_json(folder / "state.json", state)
    return folder, state


def _scene_scope(folder, state, settings):
    """Refuse unsupported scene-chain work before provider setup or variation expansion."""
    if state.get("schema_version", 1) < 5:
        return
    from . import h3scene_family as scene
    from .preflight import target_for
    snapshot = folder / "template.original.json"
    try:
        source = snapshot if snapshot.is_file() else _template(settings)
        template = json.loads(source.read_text())
    except (OSError, ValueError):
        return  # Ordinary preparation reports unreadable templates as before.
    if scene.is_scene_chain(template):
        problem = scene.settings_problem(settings, target_for(settings))
        if state.get("batch_parent"):
            problem = "H3 scene chains do not support batches; use a local Qty 1 run."
        if state.get("h3scene_decision", {}).get("action") == "deny":
            problem = "This H3 Scene Chain plan was denied; use Edit as new run."
        if problem:
            raise ValueError(problem)


def run(
    settings: Settings,
    concept: str,
    on_event: Callable[[dict], None],
    cancel: Event,
    resume_folder: Path | None = None,
    *,
    detach: Event | None = None,
    choose_gpu: Callable[[list[dict]], dict | None] | None = None,
    fork_seed: dict | None = None,
) -> dict:
    if resume_folder is not None:
        if fork_seed is not None:
            raise ValueError("A saved run cannot be resumed as a new run in one request.")
        folder = Path(resume_folder).expanduser().resolve()
        state = read_state(folder / "state.json")
        if state.get("sequence_parent") == "..":
            folder = folder.parent
            state = read_state(folder / "state.json")
        if state.get("schema_version") not in (1, 2, 3, 4, 5):
            raise ValueError("Unsupported project version.")
        settings = Settings(**state["settings"])
        concept = state["concept"]
    else:
        folder, state = create_run(settings, concept, fork_seed)
    with run_lock(folder):
        if resume_folder is not None:
            state = read_state(folder / "state.json")
            settings = Settings(**state["settings"])
            concept = state["concept"]
            from .cost_safety import clear_owner_failure
            if clear_owner_failure(state):
                # Resume stops any failure-shutdown countdown before new work.
                atomic_json(folder / "state.json", state)
        if state.get("schedule"):
            if settings.cloud or settings.managed_runpod:
                raise ValueError("Scheduled generation is available for Local only.")
            if (not state["schedule"].get("released")
                    or datetime.fromisoformat(state["schedule"]["due_at"]) > datetime.now(UTC)):
                raise ValueError("This request is scheduled. Use Scheduled to run now or reschedule it.")
        if state.get("batch_parent"):
            raise ValueError("This run belongs to a batch. Resume it from Batch manager.")
        _scene_scope(folder, state, settings)
        if settings.quantity > 1:
            from .variations import run_sequence
            return run_sequence(settings, concept, folder, state, on_event, cancel, detach, choose_gpu)
        return _run_locked(settings, concept, folder, state, on_event, cancel, detach, choose_gpu)


def cloud_action(folder: Path, action: str) -> dict:
    """Reconnect to a saved managed session for status or explicit shutdown.

    This path is intentionally small and read-only by default.  It uses the same
    durable project state as generation, so reopening the desktop can recover a
    session without submitting another ComfyUI job or creating another pod.
    """
    from .global_session import session_for

    project = Path(folder).expanduser().resolve()
    initial = read_state(project / "state.json")
    if initial.get("batch_parent"):
        raise ValueError("This run belongs to a batch. Manage its shared session from Batch manager.")
    if initial.get("sequence_parent") == "..":
        project = project.parent
        if read_state(project / "state.json").get("batch_parent"):
            raise ValueError("This variation belongs to a batch. Manage its shared session from Batch manager.")
    if action not in {"status", "cancel"}:
        raise ValueError("Unknown cloud action.")
    with run_lock(project):
        state = read_state(project / "state.json")
        settings = Settings(**state["settings"])
        if not settings.managed_runpod:
            raise ValueError("This project does not use a managed RunPod session.")

        def persist() -> None:
            state["updated_at"] = now()
            atomic_json(project / "state.json", state)

        session = session_for(
            settings,
            project,
            state,
            persist,
            lambda _event: None,
            Event(),
            Event(),
        )
        try:
            remote = session.status() if action == "status" else session.shutdown()
            state["cloud_status"] = copy.deepcopy(remote)
            if action == "cancel":
                state["status"] = "shutdown_requested"
            persist()
            return remote
        finally:
            session.close()


def _run_locked(settings, concept, folder, state, on_event, cancel, detach=None, choose_gpu=None,
                *, prepare_only=False, sequence=None, batch_phase="run"):
    if (batch_phase not in ("run", "stage", "submit")
            or (batch_phase != "run" and (not sequence or not sequence.get("batch") or not settings.managed_runpod))):
        raise ValueError("Batch staging and submission require an owning session")
    interrupt = Interrupt(cancel, detach)
    managed = None
    def persist(status: str | None = None):
        if status:
            state["status"] = status
        state["updated_at"] = now()
        atomic_json(folder / "state.json", state)

    def emit(event: dict):
        event = {**event, "folder": str(folder)}
        event["message"] = redact(str(event.get("message", "")))
        with (folder / "events.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"time": now(), **event}, ensure_ascii=False) + "\n")
        on_event(event)

    def check_cancel():
        interrupt.raise_if_set()

    comfy = None
    try:
        emit(
            {
                "stage": "Starting",
                "message": "Project saved · preparing generation",
                "progress": None,
            }
        )
        settings.validate(require_folders=False)
        agents.executable("ffprobe")  # Fail early, not after an expensive render.
        check_cancel()
        _scene_scope(folder, state, settings)
        if getattr(settings, "managed_runpod", False):
            from .global_session import session_for
            managed = session_for(
                settings, sequence["folder"] if sequence else folder,
                sequence["state"] if sequence else state,
                sequence["persist"] if sequence else persist, emit, cancel, interrupt.detach)
        session_state = sequence["state"] if sequence else state
        if (
            state.get("status") == "complete"
            and state.get("result")
            and state.get("downloads")
            and all(_downloaded(r, folder) for r in state["downloads"].values())
            and state["result"]["final_path"] in {r["path"] for r in state["downloads"].values()}
        ):
            if managed and not sequence:
                managed.complete(_verified_receipts(folder, state))
                persist("complete")
            from .upscale import enqueue_if_requested
            enqueue_if_requested(folder, state)
            emit({"stage": "Complete", "message": state["result"]["final_path"], "progress": 100})
            return {
                **state["result"],
                "folder": str(folder),
                "final_path": str(folder / state["result"]["final_path"]),
            }
        if batch_phase == "stage" and (state.get("prompt_id") or state.get("submission_started")):
            return {"folder": str(folder), "prompt_id": state.get("prompt_id"), "staged": True}
        if managed and (state.get("prompt_id") or state.get("submission_started")):
            settings = managed.connect()
            state["settings"] = settings.public_dict()
            persist()
        comfy = Comfy(settings) if not managed or state.get("prompt_id") or state.get("submission_started") else None
        if not state.get("prompt_id") and state.get("submission_started"):
            recovered = comfy.find_submission(state["run_id"])
            if not recovered:
                raise RuntimeError(
                    "Submission outcome is uncertain and no matching job was found. No duplicate was submitted. Check ComfyUI queue/history before starting a new run."
                )
            state["prompt_id"] = recovered
            persist("generating")
        workflow_path = folder / "workflow.api.json"
        if (sequence and state.get("batch_staged")
                and (not workflow_path.is_file()
                     or hashlib.sha256(workflow_path.read_bytes()).hexdigest() != state.get("batch_staged_sha256")
                     or state.get("batch_staged_sid") != session_state.get("managed_session", {}).get("sid"))):
            raise ValueError("Staged batch workflow or GPU session changed; review before submission")
        if not state.get("prompt_id"):
            if batch_phase == "submit" and not state.get("batch_staged"):
                raise RuntimeError("Batch workflow was not staged before queue submission")
            if sequence and state.get("batch_staged") and workflow_path.is_file():
                if batch_phase == "stage":
                    return {"folder": str(folder), "staged": True}
                if managed:
                    settings = managed.connect(session_state.get("resource_choice"))
                    state["settings"] = settings.public_dict()
                    persist()
                if comfy:
                    comfy.close()
                comfy = Comfy(settings)
                graph = json.loads(workflow_path.read_text())
                validate_graph(graph, comfy.object_info())
            else:
                from . import refcache
                accepted = bool(state.get("workflow_acceptance"))
                # Accepted work with a complete, verified reference cache needs no linked mount.
                cached = accepted and state["schema_version"] >= 2 and refcache.complete(folder, state)
                settings.validate(require_folders=not cached)
                roots = settings.reference_roots()
                info = managed.schema() if managed else comfy.object_info()
                if state["schema_version"] >= 2:
                    from .editing import prepare_compiled, prepare_edit
                    manifest = refcache.manifest_for(folder, state) if accepted else None

                    def resolve(value):
                        entry = refcache.find(manifest, value, roots)
                        if entry is not None:
                            return Path(entry["original"])
                        if not accepted and not roots[0].is_dir():
                            raise refcache.ReferenceCacheError(
                                "The linked project folder is unavailable, so references cannot be captured. "
                                "Nothing was accepted; reconnect it and Resume (saved agent work is reused).")
                        return _inside(value, roots)

                    def references(prepared, response):
                        uploads = []
                        used = set()
                        for slot in prepared["reference_slots"]:
                            source = resolve(slot["value"])
                            uploads.append((slot["node_id"], slot["input"], source))
                            used.add(str(source))
                        for path in response["used_references"]:
                            used.add(str(resolve(path)))
                        for slot in prepared["prompt_slots"]:
                            text = slot["text"]
                            if text.count("<d>") != text.count("</d>"):
                                raise ValueError("Agent wrote unbalanced H3 dialogue tags.")
                        return uploads, sorted(used)

                    brief = {
                        "concept": concept,
                        "project_folder": str(roots[0]),
                        # Accepted work never reaches the agent; don't touch the mount for it.
                        "project_index": [] if accepted else _inventory(roots[0]),
                        "template_folder": str(roots[1]) if len(roots) > 1 else "",
                        "duration_seconds": settings.duration,
                        "megapixels": settings.megapixels,
                        "aspect_ratio": settings.aspect_ratio or "template",
                        "quantity": settings.quantity,
                        "steps": settings.steps,
                    }
                    source = (
                        Path(state["template_path"])
                        if state.get("template_path")
                        else _template(settings)
                    )
                    prepare = prepare_compiled if state["schema_version"] >= 5 else prepare_edit
                    prepared, plan, uploads, used = prepare(
                        settings,
                        folder,
                        state,
                        info,
                        interrupt,
                        emit,
                        persist,
                        source,
                        brief,
                        references,
                    )
                    graph = copy.deepcopy(prepared["graph"])
                    emit({"stage": "Validated", "message": prepared["summary"], "progress": None})
                else:
                    prepared_path = folder / "prepared.json"
                    if prepared_path.is_file():
                        prepared = json.loads(prepared_path.read_text())
                    else:
                        source = _template(settings)
                        prepared = prepare_workflow(
                            json.loads(source.read_text()),
                            info,
                            megapixels=settings.megapixels,
                            duration=settings.duration,
                            aspect_ratio=settings.aspect_ratio,
                            run_id=state["run_id"],
                        )
                        for slot in prepared["prompt_slots"]:
                            slot["sections"] = re.findall(r"(?m)^([a-z_]+):", slot.get("text", ""))
                            slot["text"] = ""
                            prepared["graph"][slot["node_id"]]["inputs"][slot["input"]] = ""
                        atomic_json(prepared_path, prepared)
                        state["template_path"] = str(source)
                        state["template_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
                        persist("preparing")
                    validate_graph(prepared["graph"], info)
                    emit({"stage": "Preparing", "message": prepared["summary"], "progress": None})
                    check_cancel()
                    plan_path = folder / "plan.json"
                    if plan_path.is_file():
                        plan = json.loads(plan_path.read_text())
                    else:
                        brief = {
                            "concept": concept,
                            "project_folder": str(roots[0]),
                            "project_index": _inventory(roots[0]),
                            "template_folder": str(roots[1]) if len(roots) > 1 else "",
                            "duration_seconds": settings.duration,
                            "megapixels": settings.megapixels,
                            "aspect_ratio": settings.aspect_ratio or "template",
                            "workflow_summary": prepared["summary"],
                            "prompt_slots": _prompt_brief(prepared["prompt_slots"]),
                            "reference_slots": prepared["reference_slots"],
                        }
                        persist("preparing")
                        plan = agents.generate_plan(settings, folder, brief, interrupt, emit)
                    graph, uploads, used = apply_plan(prepared, plan, roots)
                (folder / "script.txt").write_text(plan["script"], encoding="utf-8")
                (folder / "used_references.txt").write_text("\n".join(used) + "\n", encoding="utf-8")
                state["final_node"] = prepared["final_node"]
                state["summary"] = prepared["summary"]
                persist("prepared")
                emit({"stage": "Story and prompts saved", "message": "Agent preparation is complete. Checking ComfyUI readiness.", "progress": None})
                if prepare_only:
                    return {"prepared": prepared, "plan": plan, "info": info}
                try:
                    validate_graph(graph, info)
                except ValueError as exc:
                    raise ValueError(
                        f"Story, prompts and workflow copy are saved in {folder}. "
                        f"Rendering is blocked: {exc}. Resolve the ComfyUI setup, then Resume saved run; "
                        "the accepted agent work will be reused."
                    ) from exc
                if managed:
                    if not session_state.get("managed_session", {}).get("sid"):
                        from .credentials import load_credentials
                        from .resources import gpu_options
                        if choose_gpu is None:
                            raise RuntimeError("Select a GPU in the application before provisioning RunPod.")
                        persist("choosing_gpu")
                        estimate_settings = Settings(**session_state["settings"])
                        choices = gpu_options(managed.config(), load_credentials(settings),
                                              settings.runpod_gpu_tiers, graph,
                                              sequence["folder"] if sequence else folder,
                                              estimate_settings, interrupt, emit)
                        check_cancel()
                        selected = choose_gpu(choices)
                        check_cancel()
                        if selected is None:
                            raise Detached("GPU selection closed. No new compute was requested; Resume retains the prepared concept.")
                        if selected not in choices:
                            raise ValueError("GPU selection does not match current available options.")
                        session_state["resource_choice"] = selected
                        (sequence["persist"] if sequence else persist)()
                    settings = managed.connect(session_state.get("resource_choice"))
                    state["settings"] = settings.public_dict()
                    if comfy:
                        comfy.close()
                    comfy = Comfy(settings)
                    validate_graph(graph, comfy.object_info())
                persist("uploading")
                uploaded: dict[Path, str] = {}
                manifest = refcache.manifest_for(folder, state) if state["schema_version"] >= 2 else None

                def report(message):
                    state.setdefault("reference_cache_problems", []).append({"at": now(), "message": message})
                    persist()
                    emit({"stage": "Preparing", "message": message, "progress": None})
                for node, name, source in uploads:
                    check_cancel()
                    emit(
                        {
                            "stage": "Preparing",
                            "message": f"Sending reference: {source.name}",
                            "progress": None,
                        }
                    )
                    if source not in uploaded:
                        # Phase 6: read the verified cached copy, never an unverified file.
                        entry = refcache.find(manifest, str(source), roots)
                        data = refcache.localize(entry, report) if entry is not None else source
                        cache = session_state.setdefault("reference_uploads", {}) if sequence else {}
                        cache_key = hashlib.sha256((settings.comfy_url + "\0" +
                            (entry["sha256"] if entry is not None
                             else hashlib.sha256(data.read_bytes()).hexdigest())).encode()).hexdigest()
                        uploaded[source] = cache.get(cache_key) or comfy.upload(data, state["run_id"], interrupt)
                        cache[cache_key] = uploaded[source]
                        if sequence:
                            sequence["persist"]()
                    graph[node]["inputs"][name] = uploaded[source]
                # Refresh combo choices after new reference files were uploaded.
                validate_graph(graph, comfy.object_info())
                atomic_json(workflow_path, graph)
                if batch_phase == "stage":
                    state["batch_staged"] = True
                    state["batch_staged_sha256"] = hashlib.sha256(workflow_path.read_bytes()).hexdigest()
                    state["batch_staged_sid"] = session_state.get("managed_session", {}).get("sid")
                    persist("staged")
                    return {"folder": str(folder), "staged": True}
            check_cancel()
            if session_state.get("schedule"):
                while True:
                    check_cancel()
                    queue = comfy.queue()
                    if not all(isinstance(queue.get(k), list) for k in ("queue_running", "queue_pending")):
                        raise RuntimeError("Cannot verify that the local ComfyUI queue is empty.")
                    if not queue["queue_running"] and not queue["queue_pending"]:
                        break
                    emit({"stage": "Waiting", "message": "Scheduled request is waiting for the local ComfyUI queue to clear.", "progress": None})
                    interrupt.wait(5)
            persist("submitting")
            state["submission_started"] = True
            persist()
            allowed = (managed.before_submit(state["run_id"]) if sequence else managed.before_submit()) if managed else True
            if not allowed:
                recovered = comfy.find_submission(state["run_id"])
                if not recovered:
                    raise RuntimeError("Submission permission was already consumed. Resume will reconcile; no duplicate render was submitted.")
                state["prompt_id"] = recovered
            else:
                state["prompt_id"] = comfy.submit(graph, state["run_id"], state["client_id"])
            state.setdefault("generation_started_at", now())
            persist("generating")
        else:
            graph = json.loads(workflow_path.read_text())
        if managed:
            if sequence:
                managed.bind(state["prompt_id"], state["run_id"])
            else:
                managed.bind(state["prompt_id"])
        if batch_phase == "submit":
            emit({"stage": "Queued", "message": f"ComfyUI job {state['prompt_id']} queued", "progress": None})
            return {"folder": str(folder), "prompt_id": state["prompt_id"], "queued": True}
        history_path = folder / "comfy-history.json"
        if history_path.is_file():
            history = json.loads(history_path.read_text())
        else:
            emit({"stage": "Generating", "message": "Waiting for ComfyUI", "progress": None})
            if managed and getattr(managed, "retired", False):
                remote = managed.status()
                if sequence:
                    remote = {**remote, "manifest": remote.get("jobs", {}).get(state["run_id"], {}).get("outputs", [])}
                history = _history_from_manifest(remote, graph, state["prompt_id"])
            elif managed or detach is not None:
                history = comfy.monitor(state["prompt_id"], state["client_id"], graph, cancel, emit,
                                        detach=interrupt.detach, cancel_remote=not bool(managed))
            else:
                history = comfy.monitor(state["prompt_id"], state["client_id"], graph, cancel, emit)
            atomic_json(history_path, history)
        state.setdefault("generation_completed_at", now())
        state["execution_seconds"] = _execution_seconds(history)
        persist("downloading")
        files = output_files(history)
        final_files = [
            f
            for f in files
            if f["node_id"] == str(state["final_node"])
            and Path(f["filename"]).suffix.lower() in VIDEO_SUFFIXES
        ]
        if len(final_files) != 1:
            raise RuntimeError(
                "The workflow did not return one identifiable final video. Inspect comfy-history.json."
            )
        # Every explicit SaveVideo branch must return media, including shot saves.
        returned_nodes = {f["node_id"] for f in files}
        missing = {
            node for node, value in graph.items() if value["class_type"] == "SaveVideo"
        } - returned_nodes
        if missing:
            raise RuntimeError(
                "ComfyUI did not return all saved shots: " + ", ".join(sorted(missing))
            )
        final_path = None
        for index, descriptor in enumerate(files):
            check_cancel()
            key = hashlib.sha256(json.dumps(descriptor, sort_keys=True).encode()).hexdigest()
            final = descriptor == final_files[0]
            node_dir = re.sub(r"[^A-Za-z0-9_-]", "_", descriptor["node_id"])
            target = (
                folder
                / ("final" if final else "shots")
                / ("" if final else node_dir)
                / (key[:8] + "_" + descriptor["filename"])
            )
            receipt = state["downloads"].get(key)
            emit(
                {
                    "stage": "Downloading",
                    "message": f"Saving {'final cut' if final else 'generated output'} · {index + 1}/{len(files)}",
                    "progress": 100 * index / len(files),
                }
            )
            if not receipt or not _downloaded(receipt, folder):

                def progress(size, total, filename=descriptor["filename"], position=index):
                    emit(
                        {
                            "stage": "Downloading",
                            "message": f"{filename} · {size / 1_000_000:.1f} MB",
                            "progress": 100 * (position + size / total) / len(files)
                            if total
                            else None,
                        }
                    )

                if managed and getattr(managed, "retired", False):
                    receipt = managed.download(descriptor, target, interrupt, progress)
                else:
                    receipt = comfy.download(descriptor, target, interrupt, progress)
                state["downloads"][key] = receipt
                state["downloads"][key]["path"] = str(target.relative_to(folder))
                persist()
            state.setdefault("download_files", {})[key] = descriptor
            if managed and Path(descriptor["filename"]).suffix.lower() in VIDEO_SUFFIXES:
                probe_video(folder / state["downloads"][key]["path"])
            persist()
            if final:
                final_path = folder / state["downloads"][key]["path"]
        if final_path is None:
            raise RuntimeError("Final video was not saved.")
        metadata = probe_video(final_path)
        result = {"folder": str(folder), "final_path": str(final_path), "metadata": metadata}
        state["result"] = {
            **result,
            "folder": ".",
            "final_path": str(final_path.relative_to(folder)),
        }
        persist("downloaded" if managed and not sequence else "complete")
        from .upscale import enqueue_if_requested
        enqueue_if_requested(folder, state)  # Durable request only; the upscale host is never contacted here.
        if managed and not sequence:
            check_cancel()
            managed.complete(_verified_receipts(folder, state))
            persist("complete")
            from .resources import record_run
            record_run(folder, state)
        report = (
            f"SUCCESS\n{state.get('summary', '')}\n\nFinal video: {final_path}\nProject folder: {folder}\n"
            f"Saved outputs: {len(files)}\nDimensions: {metadata['width']} x {metadata['height']}\n"
            f"Duration: {metadata['duration']:.3f} seconds\nSize: {metadata['size_bytes']} bytes\n"
            f"Agent: {settings.provider}; model: {settings.model or agents.suggest_model(settings.provider, settings.model_tier)}; "
            f"reasoning: {settings.reasoning}\nComfyUI job: {state['prompt_id']}\n"
            "References are listed in used_references.txt; source assets were not duplicated.\n"
        )
        if (state.get("workflow_acceptance") or {}).get("scene_chain"):
            report += "".join(f"Warning: {warning}\n" for warning in state["workflow_acceptance"].get("lint_warnings", []))
        if managed and sequence:
            report += "Variation verified. Cloud retirement waits for the entire requested set.\n"
        elif managed:
            from .global_session import is_global
            report += ("Verified downloads acknowledged. Keep simpleGen running: normal GPU retirement follows five minutes with both queues empty. Undownloaded GPU files are disposable.\n"
                       if is_global(settings, session_state) else
                       "Verified downloads acknowledged. Normal cloud retirement follows five minutes with both queues empty. Remote files are retained.\n")
        elif settings.cloud:
            report += f"Cloud GPU: {settings.cloud_name}. Instance billing can continue after completion.\n"
        (folder / "report.txt").write_text(redact(report), encoding="utf-8")
        if sequence is None:
            # A one-off or scheduled run is complete: its cached references can go.
            from .refcache import release_quietly
            release_quietly([state["run_id"]])
        emit({"stage": "Complete", "message": str(final_path), "progress": 100})
        return result
    except Exception as error:  # noqa: BLE001 -- durable failure boundary for provider/OS errors
        if interrupt.detach.is_set() or isinstance(error, Detached):
            error = Detached("Detached. The saved run and any cloud work are preserved; Resume reconnects.")
        elif cancel.is_set() and managed and not (sequence and sequence.get("batch")):
            try:
                managed.shutdown()
            except Exception:  # noqa: BLE001 -- failure must not claim cloud retirement
                error = Cancelled("Shutdown could not be confirmed. Session is saved; inspect cloud status or intervene manually.")
            else:
                from .global_session import is_global
                error = Cancelled("Manual cloud shutdown requested. Undownloaded GPU outputs are lost."
                                  if is_global(settings, session_state) else
                                  "Manual cloud shutdown requested. Saved remote outputs are retained.")
        message = redact(str(error))
        state["error"] = message
        if (managed and not sequence and not isinstance(error, (Detached, Cancelled))
                and not cancel.is_set() and not interrupt.detach.is_set()):
            from .cost_safety import record_owner_failure
            record_owner_failure(session_state, settings, message, "run")  # Persisted just below.
        from .oom_transfer import is_oom_message
        if not settings.cloud and is_oom_message(message) and not cancel.is_set() and not interrupt.detach.is_set():
            state["failure_kind"] = "out_of_memory"
        from .h3scene_review import OverflowReview
        persist("needs_review" if isinstance(error, OverflowReview) else "detached" if isinstance(error, Detached) else "cancelled" if isinstance(error, Cancelled) else "failed")
        report = f"{state['status'].upper()}\n{message}\n\nProject folder: {folder}\n"
        if state.get("prompt_id"):
            report += f"ComfyUI job: {state['prompt_id']}\nResume reconnects to this job without submitting it again.\n"
        if settings.cloud:
            report += f"Cloud GPU: {settings.cloud_name}. Instance billing may continue.\n"
        (folder / "report.txt").write_text(report, encoding="utf-8")
        emit({"stage": state["status"].capitalize(), "message": message, "progress": None})
        raise RuntimeError(message) from None
    finally:
        if comfy:
            comfy.close()
        if managed:
            managed.close()
