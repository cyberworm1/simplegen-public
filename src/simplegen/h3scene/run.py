"""H3 Scene Chain plan validation and render spec."""
import json
import math
import re
import subprocess
from pathlib import Path


SECTIONS = (
    "subject_definitions", "summary", "retention_analysis",
    "detailed_description", "overall_soundscape", "non_diegetic_music",
)


COMPONENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*\Z")


def command(args):
    result = subprocess.run(args, capture_output=True, text=True, check=False)
    if result.returncode:
        raise RuntimeError(f"{args[0]} failed: {result.stderr.strip()[-2000:]}")
    return result


def image_file(path):
    """Probe actual contents, rather than trusting the filename extension."""
    data = json.loads(command([
        "ffprobe", "-v", "error", "-show_format", "-show_streams",
        "-of", "json", str(path),
    ]).stdout)
    formats = data.get("format", {}).get("format_name", "").split(",")
    image_container = any(f.endswith("_pipe") or f in
                          {"image2", "gif", "ico", "webp"} for f in formats)
    return image_container and any(s.get("codec_type") == "video"
                                   for s in data.get("streams", []))


def picture_path(path, library):
    """Absolute paths stay as given; others are relative to the asset folder."""
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate.resolve()
    return (Path(library) / candidate).resolve()


def validate_plan(plan, library):
    """Return every mechanical problem without changing the agent's content."""
    errors = []
    if not isinstance(plan, dict):
        return ["plan must be an object"]
    concept_id = plan.get("concept_id")
    if not isinstance(concept_id, str) or not COMPONENT.fullmatch(concept_id):
        errors.append("concept_id must be a non-empty filename component (letters, digits, _ or -)")
    for key in ("title", "script"):
        if not isinstance(plan.get(key), str) or not plan[key].strip():
            errors.append(f"{key} must be non-empty text")
    pictures = plan.get("pictures")
    if not isinstance(pictures, list):
        errors.append("pictures must be a list")
        pictures = []
    if not 1 <= len(pictures) <= 9:
        errors.append("need 1-9 pictures")
    folder = Path(library).resolve()
    for i, picture in enumerate(pictures, 1):
        label = f"picture {i}"
        if not isinstance(picture, dict):
            errors.append(f"{label} must be an object")
            continue
        if not isinstance(picture.get("subject"), str) or not picture["subject"].strip():
            errors.append(f"{label}: subject must be non-empty text")
        path = picture.get("path")
        if not isinstance(path, str) or not path.strip():
            errors.append(f"{label}: path must be non-empty text")
            continue
        try:
            resolved = picture_path(path, folder)
            if not resolved.is_relative_to(folder):
                errors.append(f"{label}: path must be inside the asset folder (including symlinks)")
            if not resolved.is_file():
                errors.append(f"{label}: file does not exist: {path}")
            elif not image_file(resolved):
                errors.append(f"{label}: file is not an image: {path}")
        except (OSError, ValueError, RuntimeError) as exc:
            errors.append(f"{label}: cannot read image: {exc}")
    segments = plan.get("segments")
    if not isinstance(segments, list) or not segments:
        errors.append("segments must be a non-empty list")
        segments = []
    for i, segment in enumerate(segments, 1):
        label = f"segment {i}"
        if not isinstance(segment, dict):
            errors.append(f"{label} must be an object")
            continue
        join = segment.get("join")
        if join not in ("first", "continue", "cut"):
            errors.append(f"{label}: join must be first, continue or cut")
        if (i == 1) != (join == "first"):
            errors.append(f"{label}: only segment 1 must use first")
        seconds = segment.get("seconds")
        if (isinstance(seconds, bool) or not isinstance(seconds, (int, float))
                or not math.isfinite(seconds) or seconds <= 0):
            errors.append(f"{label}: seconds must be a positive finite number")
        sections = segment.get("sections")
        if not isinstance(sections, dict):
            errors.append(f"{label}: sections must be an object")
            sections = {}
        for name in SECTIONS:
            text = sections.get(name)
            if not isinstance(text, str) or not text.strip():
                errors.append(f"{label}: {name} must be present and non-empty")
        # Inspect all text supplied, including unexpected section keys.
        for name, text in sections.items():
            if not isinstance(text, str):
                continue
            for token in re.findall(r"<Picture\s*(-?\d+)\s*>", text):
                if not 1 <= int(token) <= len(pictures):
                    errors.append(f"{label}/{name}: <Picture {token}> is outside 1-{len(pictures)}")
        depth = 0
        balanced = True
        for name in SECTIONS:
            text = sections.get(name, "")
            if not isinstance(text, str):
                continue
            for token in re.findall(r"</?d>", text):
                depth += 1 if token == "<d>" else -1
                if depth not in (0, 1):
                    balanced = False
        if depth or not balanced:
            errors.append(f"{label}: <d>/</d> must be balanced, ordered and not nested")
    return errors


def render_prompt(sections):
    return "\n\n".join(f"{name}:\n{sections[name]}" for name in SECTIONS)


def make_spec(plan, settings, run_id, library):
    return {
        "name": plan["title"],
        "pictures": [str(picture_path(p["path"], library)) for p in plan["pictures"]],
        "width": settings["width"], "height": settings["height"],
        "steps": settings["steps"], "context_frames": settings["context_frames"],
        "ref_image_size": "match", "scheduler": "beta",
        "output_prefix": f"simpleGen/{run_id}",
        "extra_trim": int(settings.get("extra_trim", 0)),
        "segments": [{"join": s["join"], "seconds": s["seconds"],
                      "seed": settings["seed_base"] + i,
                      "prompt": render_prompt(s["sections"])}
                     for i, s in enumerate(plan["segments"])],
    }
