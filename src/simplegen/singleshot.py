"""Single-shot H3 Ref2VA templates with a voice-timbre reference (h3voice).

A single-shot template is a ComfyUI workflow exported by h3voice: its
``extra.h3voice`` names the target profile (loaders, models, render defaults).
The agent writes a shot spec — pictures, voice clips, subjects, dialogue — and
code renders the six-block prompt, lints it, builds the API graph and fills
widgets from the target schema. Timing, numbering and wiring never come from
the agent. Everything after preparation (uploads, submission, downloads,
RunPod lifecycle) is the ordinary schema-5 path.
"""

from __future__ import annotations

import copy
import hashlib
import math
from pathlib import Path

from .compiler import _normal, quote_needle, quoted_lines, reference_problem
from .dimensions import resolve_dimensions
from .h3voice import graph as hgraph
from .h3voice import lint as hlint
from .h3voice import prompt as hprompt
from .h3voice import refs as hrefs
from .h3voice import schema as hschema
from .h3voice import spec as hspec

KIND = "ref2va_voice_single_shot"
FORMAT = 1
MAX_SECONDS = 15.0
MIN_SECONDS = 4.0
AUDIO_SUFFIXES = {".wav", ".flac", ".mp3", ".m4a", ".ogg", ".opus", ".aac"}
SKILL = "h3-voice-shot"


def meta(template: dict) -> dict | None:
    value = template.get("extra", {}).get("h3voice") if isinstance(template, dict) else None
    return value if isinstance(value, dict) and value.get("kind") == KIND else None


def is_single_shot(template: dict) -> bool:
    return meta(template) is not None


def profile(template: dict) -> dict:
    data = meta(template)
    if data is None:
        raise ValueError("Not an h3voice single-shot workflow")
    if data.get("format") != FORMAT:
        raise ValueError(f"Unsupported h3voice workflow format {data.get('format')!r}; rebuild it with h3voice")
    prof = data.get("profile") or {}
    if not isinstance(prof.get("models"), dict) or not {"unet", "clip", "video_vae", "audio_vae"} <= set(prof["models"]):
        raise ValueError("h3voice workflow profile is missing its model loaders")
    return copy.deepcopy(prof)


def settings_problem(settings) -> str | None:
    if settings.quantity != 1:
        return "Single-shot voice workflows render one video; set Qty to 1."
    if not MIN_SECONDS <= settings.duration <= MAX_SECONDS:
        return (f"Single-shot voice workflows render {MIN_SECONDS:g}–{MAX_SECONDS:g} seconds; "
                f"set Duration to {MAX_SECONDS:g} or less (it is {settings.duration:g}).")
    return None


def _render(prof: dict, settings) -> dict:
    """Profile render defaults with simpleGen's steps and Megapixel/aspect choice."""
    render = dict(prof.get("render") or {})
    base_w, base_h = hgraph.resolution(render)
    width, height = resolve_dimensions(settings.megapixels, base_w, base_h, settings.aspect_ratio)
    render.update(width=width, height=height, steps=settings.steps, output_prefix="")
    render.pop("megapixels", None)
    render.pop("aspect", None)
    return render


def seed_for(run_id: str) -> int:
    return int(hashlib.sha256(run_id.encode()).hexdigest()[:12], 16)


def brief(template: dict, settings, run_id: str, read_only_path: Path) -> dict:
    prof = profile(template)
    render = _render(prof, settings)
    return {
        "workflow_brief": True,
        "single_shot": True,
        "skill": SKILL,
        "duration_seconds": settings.duration,
        "fps": hrefs.FPS,
        "frames": hrefs.align_frames(round(settings.duration * hrefs.FPS)),
        "speech_words_per_second": hrefs.WORDS_PER_SECOND,
        "speech_words_budget": math.floor(settings.duration * hrefs.WORDS_PER_SECOND),
        "limits": {"pictures": hrefs.MAX_IMAGES, "voices": hrefs.MAX_AUDIOS,
                   "total_voice_seconds": hrefs.MAX_AUDIO_SECONDS},
        "resolution": {"width": render["width"], "height": render["height"],
                       "megapixels": settings.megapixels, "aspect_ratio": settings.aspect_ratio or "template"},
        "steps": settings.steps,
        "quantity": settings.quantity,
        "target_profile": prof.get("name"),
        "workflow_file_read_only": str(read_only_path),
        "prompt_blocks": [name for name, _ in hprompt.BLOCKS],
        "instructions": [
            "Return a shot spec; the app numbers references, writes the six prompt blocks and builds the workflow.",
            ("pictures: identity images (absolute paths under a linked folder). voices: clean speech clips of the "
             "speaking characters (absolute paths); set start/duration to a clean stretch about as long as the lines."),
            "A voice clip sets how a character sounds; the spoken words come only from lines[].text.",
            "Refer to pictures and subjects in free text as [[id]]; never write <Picture N>, <Audio N> or <Subject N>.",
        ],
    }


def plan_schema(brief: dict, quantity: int = 1) -> dict:
    def obj(properties: dict) -> dict:
        return {"type": "object", "properties": properties, "required": list(properties),
                "additionalProperties": False}
    string, number = {"type": "string"}, {"type": "number"}
    strings = {"type": "array", "items": string}
    shot = obj({
        "pictures": {"type": "array", "minItems": 1, "maxItems": hrefs.MAX_IMAGES,
                     "items": obj({"id": string, "file": string, "note": string})},
        "voices": {"type": "array", "maxItems": hrefs.MAX_AUDIOS,
                   "items": obj({"id": string, "file": string, "subject": string, "start": number,
                                 "duration": number, "language": string, "delivery": string})},
        "subjects": {"type": "array", "minItems": 1,
                     "items": obj({"id": string, "who": string, "pictures": strings, "keep": string})},
        "shots": {"type": "array", "minItems": 1,
                  "items": obj({"description": string, "after": string,
                                "lines": {"type": "array", "items": obj({"speaker": string, "text": string,
                                                                         "verb": string})}})},
        "style": string, "soundscape": string, "music": string,
    })
    return obj({"script": string, "used_references": strings, "shot": shot})


def to_spec(plan: dict, settings, run_id: str) -> hspec.Spec:
    if not isinstance(plan, dict) or set(plan) != {"script", "used_references", "shot"}:
        raise ValueError("Creative plan must have exactly script, used_references and shot")
    if not isinstance(plan["script"], str) or not plan["script"].strip():
        raise ValueError("Creative plan needs a nonempty script")
    if not isinstance(plan["used_references"], list) or not all(isinstance(p, str) for p in plan["used_references"]):
        raise ValueError("used_references must be a list of paths")
    shot = plan["shot"]
    if not isinstance(shot, dict):
        raise ValueError("shot must be an object")  # noqa: TRY004 -- agent data error
    data = copy.deepcopy(shot)
    for voice in data.get("voices", []):
        if isinstance(voice, dict):
            voice["mode"] = "reference"
    data.update(name="final", duration=float(settings.duration), seed=seed_for(run_id))
    try:
        return hspec.from_dict(data, Path("/"))
    except hspec.SpecError as error:
        raise ValueError(f"Shot spec: {error}") from None


def check(plan: dict, settings, run_id: str, *, concept: str, roots: list[Path],
          warnings: list | None = None) -> tuple[hspec.Spec, str]:
    """Shape + content checks phrased for the agent's single correction pass."""
    warnings = warnings if warnings is not None else []
    spec = to_spec(plan, settings, run_id)
    refs = hrefs.assign(spec)
    try:
        text = hprompt.render(spec, refs)
    except (ValueError, KeyError) as error:
        raise ValueError(f"Shot spec: {error}") from None
    errors = []
    for picture in spec.pictures:
        problem = reference_problem(picture.file, roots)
        if problem:
            errors.append(f"Picture {picture.id} {problem}: {picture.file}")
    for voice in spec.voices:
        problem = audio_problem(voice.file, roots)
        if problem:
            errors.append(f"Voice {voice.id} {problem}: {voice.file}")
    ctx = hlint.run(spec, refs, text, check_files=False)
    for issue in ctx.issues:
        if issue.level == "warning":
            warnings.append(f"{issue.code}: {issue.message}")
        else:  # errors, and audio that could not be measured (ffprobe is required here)
            errors.append(issue.message)
    expected = quoted_lines(concept)
    if expected:
        spoken = "   ".join(_normal(line.text) for s in spec.shots for line in s.lines)
        position = 0
        for line in expected:
            needle = quote_needle(line)
            count = spoken.count(needle)
            found = spoken.find(needle, position)
            if count != 1 or found < 0:
                errors.append(f"The concept's quoted line \"{line}\" must be spoken verbatim, in order, "
                              f"exactly once (found {count})")
            else:
                position = found + len(needle)
    if errors:
        errors = list(dict.fromkeys(errors))
        more = f" (and {len(errors) - 12} more)" if len(errors) > 12 else ""
        raise ValueError("Creative plan needs changes: " + "; ".join(errors[:12]) + more)
    return spec, text


def audio_problem(path: str, roots: list[Path]) -> str | None:
    source = Path(path).expanduser()
    if not source.is_absolute():
        return "must be an absolute local path"
    try:
        resolved = source.resolve()
        inside = next((root for root in roots if resolved.is_relative_to(root)), None)
        if inside is None:
            return "is outside the linked folders"
        if any(part.startswith(".") for part in resolved.relative_to(inside).parts):
            return "is a hidden file"
        if not resolved.is_file():
            return "does not exist"
    except OSError as error:
        return f"cannot be read ({type(error).__name__})"
    if resolved.suffix.lower() not in AUDIO_SUFFIXES:
        return f"is not an audio file ({', '.join(sorted(AUDIO_SUFFIXES))})"
    if hlint.probe_seconds(resolved) is None:
        return "is not readable audio (ffprobe)"
    return None


def compile_graph(template: dict, spec: hspec.Spec, text: str, settings, run_id: str, object_info: dict) -> dict:
    """API graph; LoadImage/LoadAudio hold absolute sources until the engine uploads them."""
    prof = profile(template)
    prof["render"] = _render(prof, settings)
    files = {item.id: str(Path(item.file).expanduser().resolve()) for item in [*spec.pictures, *spec.voices]}
    graph = hschema.complete(hgraph.build(spec, prof, text, files).api(), object_info)
    graph["save"]["inputs"]["filename_prefix"] = f"simpleGen/{run_id}/final"
    if "save_audio" in graph:
        graph["save_audio"]["inputs"]["filename_prefix"] = f"simpleGen/{run_id}/shots/voice_check"
    return graph


ALLOWED_CLASSES = {
    "UNETLoader", "CLIPLoader", "VAELoader", "LoraLoaderModelOnly", "LoadImage", "LoadAudio", "TrimAudioDuration",
    "MiniMaxH3ReferenceToVideo", "MiniMaxH3AddGuide", "RandomNoise", "KSamplerSelect", "BasicScheduler",
    "BasicGuider", "SamplerCustomAdvanced", "VAEDecode", "VAEDecodeAudio", "CreateVideo", "SaveVideo", "SaveAudio",
    "VRAM_Debug",  # comfyui-kjnodes: free VRAM before decode (local template)
}


def verify(graph: dict, template: dict) -> None:
    """Structural acceptance check (the single-shot counterpart of validate_edited_workflow).

    Only allow-listed node classes plus the template's own loader classes; the
    loaders select exactly the template's models; links point at existing
    nodes; one final SaveVideo; the conditioning node feeds the sampler.
    """
    prof = profile(template)
    allowed = ALLOWED_CLASSES | {m["class"] for m in prof["models"].values()}
    unknown = sorted({n["class_type"] for n in graph.values()} - allowed)
    if unknown:
        raise ValueError(f"Single-shot workflow contains unexpected node types: {', '.join(unknown)}")
    for key, model in prof["models"].items():
        node = graph.get(key)
        if not node or node["class_type"] != model["class"]:
            raise ValueError(f"Single-shot workflow must load {key} with {model['class']}")
        for name, value in model["inputs"].items():
            if node["inputs"].get(name) != value:
                raise ValueError(f"Single-shot workflow changed {key}.{name}; the template selects {value!r}")
    for node_id, node in graph.items():
        for name, value in node["inputs"].items():
            if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str) and value[0] not in graph:
                raise ValueError(f"Node {node_id}.{name} links to missing node {value[0]}")
    saves = [k for k, n in graph.items() if n["class_type"] == "SaveVideo"]
    if saves != ["save"]:
        raise ValueError("Single-shot workflow must have exactly one SaveVideo (the final cut)")
    if graph.get("sample", {}).get("inputs", {}).get("latent_image") != ["r2v", 1]:
        raise ValueError("Single-shot sampler must start from the Reference to Video latent")


def prepared(graph: dict) -> dict:
    """The engine's prepared-workflow record for a compiled single-shot graph."""
    r2v = graph["r2v"]["inputs"]
    frames, width, height = r2v["length"], r2v["width"], r2v["height"]
    slots = [{"node_id": node_id, "input": key, "value": node["inputs"][key],
              "title": node.get("_meta", {}).get("title", node_id)}
             for node_id, node in graph.items()
             for key in [hschema.INPUT_FILES.get(node["class_type"])] if key]
    voices = sum(1 for n in graph.values() if n["class_type"] == "LoadAudio")
    return {
        "graph": graph,
        "final_node": "save",
        "prompt_slots": [{"node_id": "r2v", "input": "prompt", "text": r2v["prompt"]}],
        "reference_slots": slots,
        "summary": (f"Single-shot H3 Ref2VA voice: {frames} frames ({frames / hrefs.FPS:.3f}s at 24 fps); "
                    f"{width} × {height} ({width * height / 1_000_000:.3f} MP); "
                    f"{len(slots) - voices} picture(s), {voices} voice clip(s)."),
    }


def multigpu_nodes(template: dict) -> list[str]:
    """MultiGPU loader classes the template's profile uses (local-only: the pod has one GPU)."""
    return sorted({m["class"] for m in profile(template)["models"].values() if "MultiGPU" in m["class"]})


def support_problem(template: dict, object_info: dict, settings, target: str | None = None) -> str | None:
    """Dry-run build with placeholder content, like compiler.support_problem.

    ``target`` "cloud" also refuses MultiGPU loaders: cloud templates use core loaders only.
    """
    problem = settings_problem(settings)
    if problem:
        return problem
    if target == "cloud":
        try:
            found = multigpu_nodes(template)
        except ValueError as error:
            return str(error)
        if found:
            return (f"this workflow uses MultiGPU loaders ({', '.join(found)}), which are local-only; "
                    "choose the cloud single-shot workflow (core loaders) for RunPod")
    from .workflows import validate_graph
    try:
        spec = hspec.from_dict({
            "name": "final", "duration": float(settings.duration), "seed": 1,
            "pictures": [{"id": "p", "file": "/preflight/placeholder.png"}],
            "voices": [{"id": "v", "file": "/preflight/placeholder.wav", "subject": "s", "duration": 4.0}],
            "subjects": [{"id": "s", "who": "the person", "pictures": ["p"]}],
            "shots": [{"description": "Preflight [[s]].", "lines": [{"speaker": "s", "text": "Preflight line."}]}],
        }, Path("/"))
        text = hprompt.render(spec, hrefs.assign(spec))
        graph = compile_graph(template, spec, text, settings, "preflight", object_info)
        validate_graph(graph, object_info, check_model_availability=False)
    except (ValueError, KeyError, TypeError, IndexError, AttributeError) as error:
        return str(error) or type(error).__name__
    return None
