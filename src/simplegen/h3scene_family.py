"""Local H3 Scene Chain adapter; the approved builder owns every graph input."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from . import h3scene_refmod as refmod
from .dimensions import resolve_dimensions
from .h3scene import build, produce, run
from .h3scene import plan as contract
from .h3scene_compat import normalize_profile, template_metadata
from .singleshot import seed_for

KIND = "scene_chain"
FORMAT = 1
SKILL = "h3-scene-production"
FIXED = {
    "models": copy.deepcopy(build.MODELS), "fps": 24, "context_frames": 22,
    "extra_trim": 3, "scheduler": "beta", "ref_image_size": "match",
    "merge_seam_smooth": 0, "merge_color_match": "disabled", "merge_blend_frames": 0,
    "encode_mode": "video", "extender_crop": "disabled", "use_prev_latent": True,
    "sampler": "res_multistep", "frames": "17k+5", "trained_frames": [124, 362],
    "prompt": SKILL, "sections": list(run.SECTIONS),
    "overflow_allowance_seconds": 15, "width": 960, "height": 544,
    "joins": ["first", "continue", "cut"],
    "production_prompt_sha256": hashlib.sha256(
        (Path(__file__).parent / "skills" / SKILL / "SKILL.md").read_bytes()).hexdigest(),
    "agent_input": {"owner_text": "verbatim", "index": "INDEX.md",
                    "card_categories": ["characters", "places", "props"], "other_files": False},
    "checks": ["picture_paths", "verbatim_dialogue_in_order_with_duplicates",
               "card_verbatim_constraints", "duration", "overflow"],
    "plan_contract": "h3scene plan schema plus optional warnings and overflow",
}
# The RefMod variant: the same approved chain, plus RefMods named on the asset cards.
FIXED_REFMOD = {
    **FIXED, "prompt": refmod.SKILL,
    "production_prompt_sha256": hashlib.sha256(
        (Path(__file__).parent / "skills" / refmod.SKILL / "SKILL.md").read_bytes()).hexdigest(),
    "checks": FIXED["checks"] + ["refmod_card_names", "refmod_replaces_card_pictures",
                                 "refmod_card_verbatim_constraints"],
    "plan_contract": FIXED["plan_contract"] + " plus refmods",
    "refmod": refmod.PROFILE,
}


def is_scene_chain(template):
    return template_metadata(template) is not None


def is_refmod(template):
    data = template_metadata(template)
    return isinstance(data, dict) and isinstance(data.get("profile"), dict) and "refmod" in data["profile"]


def skill_for(prof):
    return refmod.SKILL if "refmod" in prof else SKILL


def profile(template):
    if not is_scene_chain(template):
        raise ValueError("Not an H3 scene chain template")
    data = template_metadata(template)
    # JSON round trip normalizes the builder's model tuples.
    fixed = json.loads(json.dumps(FIXED_REFMOD if is_refmod(template) else FIXED))
    stored = normalize_profile(data.get("profile"))
    if data.get("format") != FORMAT or stored != fixed:
        raise ValueError("H3 Scene Chain profile differs from the approved fixed values; rebuild the template")
    return copy.deepcopy(stored)


def settings_problem(settings, target=None):
    if target == "cloud":
        return "H3 scene chains use MultiGPU loaders and are local-only; RunPod/Cloud has not been tested."
    if settings.quantity != 1:
        return "H3 scene chains support Qty 1 only; variations and batches have not been tested."
    return None


def asset_problem(settings):
    root = Path(settings.project_folder).expanduser().resolve()
    index = root / "INDEX.md"
    if not index.is_file() or not index.resolve().is_relative_to(root):
        return "H3 Scene Chain asset folder must contain INDEX.md and characters/places/props cards."
    for category in ("characters", "places", "props"):
        if any(not p.resolve().is_relative_to(root) for p in (root / category).rglob("card.md")):
            return "H3 Scene Chain cards must stay inside the linked asset folder, including symlinks."
    return None


def render_settings(template, settings, run_id):
    prof = profile(template)
    width, height = resolve_dimensions(settings.megapixels, prof["width"], prof["height"], settings.aspect_ratio)
    return {**prof, "width": width, "height": height, "steps": settings.steps, "seed_base": seed_for(run_id)}


def envelope(template, settings, *, duration=None):
    render = render_settings(template, settings, "preflight")
    duration = settings.duration if duration is None else duration
    mp = render["width"] * render["height"] / 1_000_000
    verified = (duration <= 60 and mp <= 960 * 544 / 1_000_000) or (
        duration <= 30 and mp <= 1120 * 640 / 1_000_000)
    nearest = "60 s at 960×544 (0.522 MP)" if duration > 30 else "30 s at 1120×640 (0.717 MP)"
    message = ("Verified local tested envelope." if verified else
               f"Unverified local tested envelope; nearest tested point: {nearest}.")
    if "refmod" in render:
        verified = False
        message = ("Unverified: RefMods have been tested only in single 5-second shots at 864×480, "
                   "not in a scene chain.")
    ram_warning = duration >= 45 and mp >= 0.7
    if ram_warning:
        message += " Host-RAM warning: 60 s at 1120×640 was OOM-killed at final merge on a 64 GB host."
    return {"verified": verified, "message": message, "megapixels": mp, "ram_warning": ram_warning}


def brief(template, settings, run_id, read_only_path, *, concept):
    skill = skill_for(profile(template))
    problem = asset_problem(settings)
    if problem:
        raise ValueError(problem)
    master = (Path(__file__).parent / "skills" / skill / "SKILL.md").read_text()
    messages = produce.build_messages(master, concept, settings.duration, [], run_id,
                                      Path(settings.project_folder).expanduser().resolve())
    result = {"workflow_brief": True, "scene_chain": True, "skill": skill, "quantity": 1,
              "scene_messages": messages}
    if skill == refmod.SKILL:
        result["refmod"] = True
    return result


def _strict(node):
    """Close every object and require every property (Codex and Claude structured output)."""
    if isinstance(node, dict):
        if node.get("type") == "object" and "properties" in node:
            node["additionalProperties"] = False
            node["required"] = list(node["properties"])
        for value in node.values():
            _strict(value)
    elif isinstance(node, list):
        for value in node:
            _strict(value)
    return node


def plan_schema(brief=None, quantity=1):
    """The scene plan schema made strict. The skill's optional fields become required:
    ``warnings`` may be empty and ``overflow`` is null when the plan fits."""
    schema = copy.deepcopy(contract.SCHEMA)
    schema["properties"]["plans"]["maxItems"] = 1
    item = schema["properties"]["plans"]["items"]
    overflow = {"type": "object", "properties": {
        "needs_review": {"type": "boolean"}, "extra_seconds": {"type": "number"},
        "reason": {"type": "string"}, "options": {"type": "string"}}}
    item["properties"].update(
        input_mode={"type": "string", "enum": ["prompt", "script", "concept", "mixed"]},
        warnings={"type": "array", "items": {"type": "string"}},
        overflow={"anyOf": [overflow, {"type": "null"}]})
    if isinstance(brief, dict) and brief.get("refmod"):
        # Pictures may be replaced entirely; code requires at least one picture or RefMod.
        item["properties"]["pictures"]["minItems"] = 0
        item["properties"]["refmods"] = {"type": "array", "maxItems": refmod.SLOTS, "items": {
            "type": "object", "properties": {"name": {"type": "string"}, "subject": {"type": "string"}}}}
    return _strict(schema)


def _optional_fields(plan):
    """Strict output sends null or empty values where the skill says to leave a field out."""
    overflow = plan.get("overflow")
    if isinstance(overflow, dict):
        overflow = {key: value for key, value in overflow.items() if value is not None and value != ""}
        plan["overflow"] = overflow
    if overflow is None or overflow == {}:
        plan.pop("overflow", None)


PICTURE_COUNT = ("pictures not 1-9", "need 1-9 pictures")


def check(plan, settings, run_id, *, concept, roots, warnings=None, template=None):
    root = roots[0].resolve()
    _optional_fields(plan)
    plan.update(concept_id=run_id, script=concept, requested_seconds=settings.duration)
    errors, duration = produce.check_candidate(plan, concept, settings.duration, root)
    errors.extend(run.validate_plan(plan, root))
    if template is not None and is_refmod(template):
        if plan.get("pictures") == []:
            # RefMods may replace every picture; refmod.check requires one picture or RefMod.
            errors = [e for e in errors if e not in PICTURE_COUNT]
        refmod.check(plan, root, errors)
    pictures = plan.get("pictures")
    for picture in pictures if isinstance(pictures, list) else []:
        if isinstance(picture, dict) and isinstance(picture.get("path"), str) and ".." in Path(picture["path"]).parts:
            errors.append("Picture paths cannot contain '..'")
    if errors:
        raise ValueError("Creative plan needs changes: " + "; ".join(dict.fromkeys(errors)))
    plan["used_references"] = [str(run.picture_path(p["path"], root)) for p in plan["pictures"]]
    if warnings is not None:
        warnings.extend(plan.get("warnings", []))
    return duration


def compile_graph(template, plan, settings, run_id, object_info=None, *, warnings=None):
    root = Path(settings.project_folder).expanduser().resolve()
    spec = run.make_spec(plan, render_settings(template, settings, run_id), run_id, root)
    if is_refmod(template):
        spec["refmods"] = [r["name"].strip() for r in plan.get("refmods", [])]
    return compile_spec(spec, warnings=warnings)


def compile_spec(spec, *, warnings=None):
    """Build an approved mechanical spec (also used by scene-chain golden fixtures)."""
    warn = warnings.append if warnings is not None else print
    if "refmods" in spec:
        return refmod.build(spec, warn=warn).api()
    graph, _ = build.build(spec, spec["pictures"], warn=warn)
    return graph.api()


ALLOWED_CLASSES = {
    "UNETLoaderMultiGPU", "CLIPLoaderMultiGPU", "VAELoaderMultiGPU", "KSamplerSelect", "BasicScheduler",
    "LoadImage", "MiniMaxH3ReferenceToVideo", "MiniMaxH3VideoExtender", "RandomNoise", "BasicGuider",
    "SamplerCustomAdvanced", "VRAM_Debug", "VAEDecode", "VAEDecodeAudio", "MiniMaxH3LoopTrim",
    "CreateVideo", "SaveVideo", "MiniMaxH3VideoMerge",
}


def verify(graph, template):
    profile(template)
    allowed = ALLOWED_CLASSES | ({refmod.LOADER, refmod.APPLY_NODE} if is_refmod(template) else set())
    if not graph or {n["class_type"] for n in graph.values()} - allowed:
        raise ValueError("H3 Scene Chain workflow contains unexpected node types")
    if is_refmod(template):
        refmod.verify(graph)
    expected, _ = build.build({"pictures": ["placeholder.png"], "width": 960, "height": 544,
                               "segments": [{"join": "first", "seconds": 5, "seed": 1, "prompt": "x"}]}, warn=lambda _: None)
    for key in ("1", "2", "3", "4"):
        if graph.get(key) != expected.api()[key]:
            raise ValueError(f"H3 Scene Chain workflow changed approved model loader {key}")
    for node_id, node in graph.items():
        for name, value in node["inputs"].items():
            if isinstance(value, list) and (len(value) != 2 or value[0] not in graph or
                                          not isinstance(value[1], int) or value[1] < 0):
                raise ValueError(f"H3 Scene Chain {node_id}.{name} has an invalid link")
    final = [k for k, n in graph.items() if n["class_type"] == "SaveVideo" and
             n.get("_meta", {}).get("title") == "Final - Save final"]
    if len(final) != 1:
        raise ValueError("H3 Scene Chain workflow must have exactly one final SaveVideo")


def prepared(graph, plan=None):
    r2vs = [(k, n["inputs"]) for k, n in graph.items() if n["class_type"] == "MiniMaxH3ReferenceToVideo"]
    trims = [n["inputs"]["trim_frames"] for n in graph.values() if n["class_type"] == "MiniMaxH3LoopTrim"]
    frames = sum(n["length"] - t for (_, n), t in zip(r2vs, trims))
    total = sum(s["seconds"] for s in plan["segments"]) if plan else frames / 24
    first = r2vs[0][1]
    mods = [v for n in graph.values() if n["class_type"] == refmod.LOADER
            for k, v in n["inputs"].items() if k.startswith("mod_") and v != "(none)"]
    return {"graph": graph, "final_node": next(k for k, n in graph.items() if
            n["class_type"] == "SaveVideo" and n.get("_meta", {}).get("title") == "Final - Save final"),
            "reference_slots": [{"node_id": k, "input": "image", "value": n["inputs"]["image"]}
                                for k, n in graph.items() if n["class_type"] == "LoadImage"],
            "prompt_slots": [{"node_id": k, "input": "prompt", "text": n["prompt"]} for k, n in r2vs],
            "summary": f"H3 scene chain: plan total {total:g}s; delivered {frames / 24:.3f}s; "
                       f"{first['width']} × {first['height']}; {len(r2vs)} segments"
                       + (f"; RefMods: {', '.join(mods)}." if mods else ".")}


def support_problem(template, object_info, settings, target=None, *, cached_references=False):
    problem = settings_problem(settings, target) or (None if cached_references else asset_problem(settings))
    if problem:
        return problem
    from .workflows import validate_graph
    try:
        plan = {"title": "Preflight", "pictures": [{"path": "placeholder.png"}], "refmods": [],
                "segments": [{"join": join, "seconds": 5, "sections": {k: "Preflight" for k in run.SECTIONS}}
                             for join in ("first", "continue", "cut")]}
        graph = compile_graph(template, plan, settings, "preflight")
        verify(graph, template)
        validate_graph(graph, object_info, check_model_availability=False)
    except (ValueError, KeyError, TypeError) as error:
        return str(error)
    return None
