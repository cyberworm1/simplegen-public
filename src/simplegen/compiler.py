"""Agent writes content; code builds the native workflow (schema 5).

Pure and deterministic: timing options, the agent's workflow brief, creative
plan checks (lint) and the compiler that turns a creative plan into a native
H3 workflow. Every compiled workflow must pass
``workflow_edits.validate_edited_workflow`` unchanged.
"""

from __future__ import annotations

import copy
import json
import math
import re
from pathlib import Path

from .workflow_edits import _mapping, _setup
from .workflows import _duration_frames

FPS = 24
SPEECH_WORDS_PER_SECOND = 2.5
# Over budget × WARN ratio + words: a recorded warning. Over × ERROR ratio + words:
# an error that feeds the correction pass (salvage keeps it as a warning).
SPEECH_WARN_RATIO = 1.2
SPEECH_ERROR_RATIO = 1.5
SPEECH_TOLERANCE_WORDS = 3
# A template sample section shorter than this is boilerplate (e.g. "N/A"), not sample prose.
SAMPLE_MIN_CHARS = 40
MIN_QUOTED_WORDS = 4
COMPILER_VERSION = 1
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff"}
_IMAGE_MAGIC = (b"\x89PNG\r\n\x1a\n", b"\xff\xd8\xff", b"GIF87a", b"GIF89a", b"BM", b"II*\x00", b"MM\x00*")
_SECTION = re.compile(r"^([a-z][a-z_]{1,60}):[ \t]*$")


# 1. Timing options


def _layout(duration: float, context: int, count: int) -> list[int] | None:
    """Frames for ``count`` segments on the H3 lattice, or None when not legal."""
    if count < 1:
        return None
    target = math.ceil(duration * FPS)
    steps = max(count * 6, math.ceil((target + (count - 1) * context - count * 5) / 17))
    if steps > count * 21:
        return None
    quotient, remainder = divmod(steps, count)
    frames = [5 + 17 * (quotient + (i < remainder)) for i in range(count)]
    actual = (sum(frames) - (count - 1) * context) / FPS
    if not duration <= actual < duration + 17 / FPS + 1e-9:
        return None
    return frames


def _option(frames: list[int], context: int, recommended: bool) -> dict:
    segments = []
    for index, count in enumerate(frames):
        overlap = context if index else 0
        usable = (count - overlap) / FPS
        segments.append({"index": index + 1, "frames": count, "duration_override": min(15.0, count / FPS),
                         "usable_seconds": round(usable, 3), "overlap_seconds": round(overlap / FPS, 3),
                         "speech_words": int(usable * SPEECH_WORDS_PER_SECOND)})
    return {"id": f"segments-{len(frames)}", "segment_count": len(frames), "recommended": recommended,
            "actual_total_seconds": round((sum(frames) - (len(frames) - 1) * context) / FPS, 3),
            "segments": segments}


def segment_options(duration: float, default_seconds: float, context_frames: int) -> list[dict]:
    """Up to three legal layouts: the current recommendation and its ±1-segment neighbours."""
    recommended = _duration_frames(duration, default_seconds, context_frames)
    options = {len(recommended): _option(recommended, context_frames, True)}
    for count in (len(recommended) - 1, len(recommended) + 1):
        frames = _layout(duration, context_frames, count)
        if frames is not None:
            options[count] = _option(frames, context_frames, False)
    return [options[count] for count in sorted(options)]


# Template analysis


def _normal_link(link) -> tuple:
    if isinstance(link, list):
        return (*link[:5], link[5] if len(link) > 5 else None)
    return (link["id"], link["origin_id"], link["origin_slot"], link["target_id"], link["target_slot"],
            link.get("type"))


def _links(workflow: dict) -> list[tuple]:
    """(id, origin, origin_slot, target, target_slot, type) for every link form the validator accepts."""
    from .workflow_edits import _edges
    _edges(workflow)  # Same accepted forms and errors as validate_edited_workflow.
    return [_normal_link(link) for link in workflow.get("links", [])]


def _make_link(sample, values: tuple):
    """A new link in the container's own representation (6-list, 5-list or object)."""
    identity, origin, origin_slot, target, target_slot, kind = values
    if isinstance(sample, dict):
        link = {"id": identity, "origin_id": origin, "origin_slot": origin_slot, "target_id": target,
                "target_slot": target_slot}
        if "type" in sample:
            link["type"] = kind
        return link
    if isinstance(sample, list) and len(sample) == 5:
        return [identity, origin, origin_slot, target, target_slot]
    return [identity, origin, origin_slot, target, target_slot, kind]


def _widget_names(node: dict) -> list[str]:
    return [inp["name"] for inp in node.get("inputs", []) if inp.get("widget")]


def _slot(node: dict, name: str, kind: str = "inputs") -> int:
    return next(i for i, item in enumerate(node.get(kind, [])) if item["name"] == name)


def section_names(prompt: str) -> list[str]:
    names = []
    for line in prompt.splitlines():
        match = _SECTION.match(line)
        if match and match.group(1) not in names:
            names.append(match.group(1))
    return names


RAW_SECTION = "prompt"


def parse_sections(prompt: str, names: list[str], raw: bool = False) -> dict:
    """Split a rendered prompt into its named sections (only the given names are headings).

    ``raw``: the template's prompt has no named sections; the whole text is one section.
    """
    if raw:
        return {RAW_SECTION: prompt.strip()}
    result, current = {}, None
    for line in prompt.splitlines():
        match = _SECTION.match(line)
        if match and match.group(1) in names and match.group(1) not in result:
            current = match.group(1)
            result[current] = []
        elif current is not None:
            result[current].append(line)
    return {name: "\n".join(lines).strip() for name, lines in result.items()}


def render_prompt(sections: dict, names: list[str], raw: bool = False) -> str:
    if raw:
        return sections[RAW_SECTION].strip()
    return "\n\n".join(f"{name}:\n{sections[name].strip()}" for name in names)


def analyze(template: dict, object_info: dict, settings, run_id: str = "compile") -> dict:
    """Everything the brief, lint and compiler need from a native H3 template."""
    setup = _setup(template, object_info, settings.megapixels, settings.duration, run_id, settings.aspect_ratio)
    nodes = setup["nodes"]
    links = _links(template)
    by_target = {(link[3], link[4]): link for link in links}
    kind = setup["segment_type"]
    card_ids = {n["id"] for n in setup["segments"]}
    previous = {}
    for card in setup["segments"]:
        edge = by_target.get((card["id"], _slot(card, "previous")))
        previous[card["id"]] = edge[1] if edge else None
    chain, cursor = [], None
    while len(chain) < len(card_ids):
        matches = [i for i in card_ids - set(chain) if previous[i] == cursor]
        if len(matches) != 1:
            raise ValueError("Template segment cards must form one linear chain")
        cursor = matches[0]
        chain.append(cursor)
    first = nodes[chain[0]]
    report = next((i for i, output in enumerate(first.get("outputs", [])) if output.get("name") == "segment_report"),
                  None)
    previews = {}
    for link in links:
        if (report is not None and link[1] in card_ids and link[2] == report
                and nodes.get(link[3], {}).get("type") == "PreviewAny"):
            previews[link[1]] = link[3]
    names = dict(zip(_widget_names(first), first.get("widgets_values", [])))
    sample = names.get("prompt") if isinstance(names.get("prompt"), str) else ""
    sections = section_names(sample)
    raw = not sections
    if raw:
        sections = [RAW_SECTION]  # One free-text prompt per segment.
    samples = []
    for card_id in chain:
        values = dict(zip(_widget_names(nodes[card_id]), nodes[card_id].get("widgets_values", [])))
        if isinstance(values.get("prompt"), str) and values["prompt"].strip():
            samples.append(values["prompt"])
    project = setup["project"]
    reference_order = {}
    for link in links:
        if link[3] == project["id"] and nodes.get(link[1], {}).get("type") == "LoadImage":
            reference_order[link[1]] = link[4]
    loaders = [n for n in template["nodes"] if n["type"] == "LoadImage"]
    loaders.sort(key=lambda n: (reference_order.get(n["id"], 10_000), template["nodes"].index(n)))
    slots = [{"node_id": str(n["id"]), "title": n.get("title", "Reference image"), "order": i + 1,
              "picture": f"<Picture {i + 1}>"} for i, n in enumerate(loaders)]
    values = setup["values"]
    return {"setup": setup, "segment_type": kind, "chain": chain, "raw_prompt": raw, "previews": previews, "report_slot": report,
            "sections": sections, "sample_prompts": samples, "reference_slots": slots,
            "context_frames": int(values["context_frames"]), "default_seconds": float(values["default_seconds"]),
            "options": segment_options(settings.duration, float(values["default_seconds"]),
                                       int(values["context_frames"]))}


# 2. Workflow brief


def workflow_brief(template: dict, object_info: dict, settings, run_id: str, read_only_path: Path) -> dict:
    """JSON-serializable brief; replaces the editing contract for schema 5 runs."""
    info = analyze(template, object_info, settings, run_id)
    setup = info["setup"]
    example = "…" if info["raw_prompt"] else "\n\n".join(f"{name}:\n…" for name in info["sections"])
    return {
        "workflow_brief": True,
        "compiler_version": COMPILER_VERSION,
        "duration_seconds": settings.duration,
        "fps": FPS,
        "context_frames": info["context_frames"],
        "speech_words_per_second": SPEECH_WORDS_PER_SECOND,
        "timing_options": info["options"],
        "recommended_option": next(o["id"] for o in info["options"] if o["recommended"]),
        "prompt_sections": info["sections"],
        "prompt_structure_example": example,
        "reference_slots": info["reference_slots"],
        "resolution": {"width": setup["width"], "height": setup["height"], "megapixels": settings.megapixels,
                       "aspect_ratio": settings.aspect_ratio or "template"},
        "steps": settings.steps,
        "quantity": settings.quantity,
        "workflow_file_read_only": str(read_only_path),
        "instructions": [
            "Read the workflow and references to understand the chain; do not edit any file.",
            "Choose one timing_option id. Write exactly that many segments, in order.",
            "Each segment supplies every prompt section by name; code renders the prompts and builds the workflow.",
            ("Fit spoken words to each segment's speech_words budget; overlap_seconds repeats the previous "
             "segment's tail as motion context."),
            "references maps every reference slot node_id to an existing absolute local image under a linked folder.",
        ],
    }


def plan_schema(brief: dict, quantity: int) -> dict:
    """Strict JSON schema for the creative plan (agent output)."""
    def obj(properties: dict) -> dict:
        return {"type": "object", "properties": properties, "required": list(properties),
                "additionalProperties": False}
    string = {"type": "string"}
    segments = {"type": "array", "minItems": 1,
                "items": obj({"sections": obj({name: string for name in brief["prompt_sections"]})})}
    properties = {
        "script": string,
        "used_references": {"type": "array", "items": string},
        "timing_option": {"type": "string", "enum": [o["id"] for o in brief["timing_options"]]},
        "references": obj({slot["node_id"]: string for slot in brief["reference_slots"]}),
        "segments": segments,
    }
    if quantity > 1:
        properties["variations"] = {"type": "array", "minItems": quantity - 1, "maxItems": quantity - 1,
                                    "items": obj({"script": string, "segments": segments})}
    return obj(properties)


# 3 and 5. Plan shape and content checks


def _option_for(info: dict, plan: dict) -> dict:
    option = next((o for o in info["options"] if o["id"] == plan.get("timing_option")), None)
    if option is None:
        raise ValueError(f"timing_option must be one of: {', '.join(o['id'] for o in info['options'])}")
    return option


def check_shape(plan: dict, info: dict, quantity: int) -> dict:
    """Structural plan checks; returns the chosen timing option."""
    if not isinstance(plan, dict):
        raise ValueError("Creative plan must be a JSON object")  # noqa: TRY004 -- agent data error
    required = {"script", "used_references", "timing_option", "references", "segments"}
    required |= {"variations"} if quantity > 1 else set()
    missing = sorted(required - plan.keys())
    if missing:
        raise ValueError("Creative plan is missing field(s): " + ", ".join(missing))
    extra = sorted(plan.keys() - required)
    if extra:
        raise ValueError("Creative plan has unexpected field(s): " + ", ".join(extra))
    if not isinstance(plan["script"], str) or not plan["script"].strip():
        raise ValueError("Creative plan needs a nonempty script")
    if not isinstance(plan["used_references"], list) or not all(isinstance(p, str) for p in plan["used_references"]):
        raise ValueError("used_references must be a list of paths")
    if not isinstance(plan["references"], dict) or not all(
            isinstance(k, str) and isinstance(v, str) for k, v in plan["references"].items()):
        raise ValueError("references must map reference slot node_id to a path")
    option = _option_for(info, plan)
    groups = [("Plan", plan["segments"])]
    if quantity > 1:
        variations = plan["variations"]
        if not isinstance(variations, list) or len(variations) != quantity - 1:
            raise ValueError(f"Return exactly {quantity - 1} variations after the base plan")
        for index, variant in enumerate(variations, 2):
            if (not isinstance(variant, dict) or set(variant) != {"script", "segments"}
                    or not isinstance(variant["script"], str) or not variant["script"].strip()):
                raise ValueError(f"Variation {index} needs only a nonempty script and segments")
            groups.append((f"Variation {index}", variant["segments"]))
    for label, segments in groups:
        if not isinstance(segments, list) or len(segments) != option["segment_count"]:
            raise ValueError(f"{label}: {option['id']} needs exactly {option['segment_count']} segments")
        for number, segment in enumerate(segments, 1):
            if not isinstance(segment, dict) or set(segment) != {"sections"} or not isinstance(segment["sections"], dict):
                raise ValueError(f"{label} segment {number} must be {{\"sections\": {{...}}}}")
            unknown = sorted(set(segment["sections"]) - set(info["sections"]))
            if unknown:
                raise ValueError(f"{label} segment {number} has unknown section(s): {', '.join(unknown)}")
            if not all(isinstance(v, str) for v in segment["sections"].values()):
                raise ValueError(f"{label} segment {number}: section text must be strings")
    return option


def reference_problem(path: str, roots: list[Path]) -> str | None:
    """Why a reference cannot be used, or None: must exist, be an image, and sit inside a linked folder."""
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
        if resolved.suffix.lower() not in IMAGE_SUFFIXES:
            return "is not an image file"
        with resolved.open("rb") as handle:
            head = handle.read(12)
    except OSError as error:
        return f"cannot be read ({type(error).__name__})"
    if not (head.startswith(_IMAGE_MAGIC) or (head[:4] == b"RIFF" and head[8:12] == b"WEBP")):
        return "is not a readable image"
    return None


def _dialogue(text: str) -> list[str]:
    return [re.sub(r"^\s*\[[^\]]{1,40}\]\s*", "", m).strip() for m in re.findall(r"<d>(.*?)</d>", text, re.DOTALL)]


def _normal(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("’", "'").replace("‘", "'")).strip().lower()


_SPEECH_BEFORE = re.compile(
    r"\b(?:say|says|said|saying|shout|shouts|shouted|yell|yells|yelled|scream|screams|screamed|ask|asks|asked|"
    r"tell|tells|told|reply|replies|replied|answer|answers|answered|whisper|whispers|whispered|mutter|mutters|"
    r"muttered|sing|sings|sang|chant|chants|announce|announces|announced|exclaim|exclaims|exclaimed|"
    r"speak|speaks|spoke|line|lines|dialogue|quote|quotes)\b[^\n\"“”]{0,40}$", re.IGNORECASE)
_LABEL_BEFORE = re.compile(r"[A-Za-z0-9)\]]\s*:\s*$")


def _spoken_context(concept: str, start: int, end: int) -> bool:
    """A quote is required dialogue only when its context says it is spoken.

    The whole line is the quote (screenplay dialogue), a speech verb shortly precedes
    it on the same line, or a speaker/label colon directly precedes it. Quotes inside
    stage directions, descriptions and editorial notes are not dialogue.
    """
    line_start = concept.rfind("\n", 0, start) + 1
    line_end = concept.find("\n", end)
    before = concept[line_start:start]
    after = concept[end:len(concept) if line_end < 0 else line_end]
    if not before.strip(" \t>*_-") and not after.strip(" \t.,;:!?*_"):
        return True
    return bool(_SPEECH_BEFORE.search(before) or _LABEL_BEFORE.search(before))


def quoted_lines(concept: str) -> list[str]:
    """Conservatively extract quoted dialogue lines (straight or curly double quotes, several words).

    Only quotes in a spoken context count (see ``_spoken_context``); a quoted phrase
    that describes action, appearance or an editorial note is not required speech.
    """
    lines = []
    concept = concept or ""
    for match in re.finditer(r"“([^”\n]{3,400})”|\"([^\"\n]{3,400})\"", concept):
        text = (match.group(1) or match.group(2)).strip()
        if len(text.split()) >= MIN_QUOTED_WORDS and _spoken_context(concept, match.start(), match.end()):
            lines.append(text)
    return lines


def quote_needle(line: str) -> str:
    """Normalized form used to find a required line in the spoken text (trailing punctuation ignored)."""
    return _normal(line).rstrip(" .,;:!?…")


def lint(plan: dict, info: dict, option: dict, *, concept: str, roots: list[Path], quantity: int,
         warnings: list | None = None, budget_errors: bool = True) -> list[str]:
    """Content problems, phrased for the agent's single correction pass.

    Speech over budget × SPEECH_WARN_RATIO (+ words) is appended to ``warnings``;
    over × SPEECH_ERROR_RATIO it is an error unless ``budget_errors`` is False
    (salvage of work written before budgets existed), then also a warning.
    """
    warnings = warnings if warnings is not None else []
    errors: list[str] = []
    slots = info["reference_slots"]
    samples = [_normal(s) for s in info["sample_prompts"]]
    sample_sections = [parse_sections(s, info["sections"], info["raw_prompt"]) for s in info["sample_prompts"]]
    sample_texts = {_normal(v) for sections in sample_sections for v in sections.values()
                    if len(v.strip()) >= SAMPLE_MIN_CHARS}
    groups = [("Plan", plan["segments"])] + (
        [(f"Variation {i}", v["segments"]) for i, v in enumerate(plan.get("variations", []), 2)]
        if quantity > 1 else [])
    for label, segments in groups:
        pairs: dict[str, str] = {}
        spoken: list[str] = []
        for number, segment in enumerate(segments, 1):
            where = f"{label} segment {number}"
            sections = segment["sections"]
            for name in info["sections"]:
                if not sections.get(name, "").strip():
                    errors.append(f"{where}: section '{name}' is missing or empty")
            text = render_prompt({n: sections.get(n, "") for n in info["sections"]}, info["sections"],
                                 info["raw_prompt"])
            if _normal(text) in samples:
                errors.append(f"{where}: the template's sample prompt was left in place; write new content")
            for name, value in sections.items():
                if _normal(value) in sample_texts:
                    errors.append(f"{where}: section '{name}' repeats the template's sample text")
            if text.count("<d>") != text.count("</d>") or re.search(r"<d>(?:(?!</d>).)*<d>", text, re.DOTALL):
                errors.append(f"{where}: unbalanced <d>/</d> dialogue tags")
            for subject, speaker in re.findall(r"<Subject (\d+)>\s*\(S(\d+)\)", text):
                for key, value, kind in ((f"S{speaker}", subject, "speaker"), (f"Subject {subject}", speaker, "subject")):
                    if pairs.setdefault(key, value) != value:
                        errors.append(f"{where}: {kind} label {key} is paired inconsistently across segments")
            lines = _dialogue(text)
            spoken.extend(lines)
            words = sum(len(line.split()) for line in lines)
            budget = option["segments"][number - 1]["speech_words"]
            message = (f"{where}: {words} spoken words exceed its speech budget of about {budget} "
                       f"({option['segments'][number - 1]['usable_seconds']:.2f}s usable); move lines or "
                       "choose another timing option")
            if words > budget * SPEECH_ERROR_RATIO + SPEECH_TOLERANCE_WORDS:
                (errors if budget_errors else warnings).append(message)
            elif words > budget * SPEECH_WARN_RATIO + SPEECH_TOLERANCE_WORDS:
                warnings.append(message)
            for picture in re.findall(r"<Picture (\d+)>", text):
                if not 1 <= int(picture) <= len(slots):
                    errors.append(f"{where}: <Picture {picture}> does not exist; there are {len(slots)} reference slots")
        expected = quoted_lines(concept)
        if expected:
            joined = "   ".join(_normal(line) for line in spoken)
            position = 0
            for line in expected:
                needle = quote_needle(line)
                count = joined.count(needle)
                found = joined.find(needle, position)
                if count != 1 or found < 0:
                    errors.append(f"{label}: the concept's quoted line \"{line}\" must be spoken verbatim, "
                                  f"in order, exactly once (found {count})")
                else:
                    position = found + len(needle)
    for slot in slots:
        path = plan["references"].get(slot["node_id"])
        if not path:
            errors.append(f"Reference slot {slot['node_id']} ({slot['title']}) has no image")
            continue
        problem = reference_problem(path, roots)
        if problem:
            errors.append(f"Reference for slot {slot['node_id']} ({slot['title']}) {problem}: {path}")
    extra = sorted(set(plan["references"]) - {s["node_id"] for s in slots})
    if extra:
        errors.append("references names unknown slot node_id(s): " + ", ".join(extra))
    return list(dict.fromkeys(errors))


def check_plan(plan: dict, info: dict, *, concept: str, roots: list[Path], quantity: int,
               warnings: list | None = None, budget_errors: bool = True) -> dict:
    option = check_shape(plan, info, quantity)
    errors = lint(plan, info, option, concept=concept, roots=roots, quantity=quantity, warnings=warnings,
                  budget_errors=budget_errors)
    if errors:
        more = f" (and {len(errors) - 12} more)" if len(errors) > 12 else ""
        raise ValueError("Creative plan needs changes: " + "; ".join(errors[:12]) + more)
    return option


# 4. Compiler


def compile_workflow(template: dict, plan: dict, settings, run_id: str, *, object_info: dict,
                     segments: list | None = None) -> dict:
    """Build the native workflow for a creative plan. Same inputs, same bytes.

    The template's own cards are reused for the first segments (their IDs, report
    previews and wiring stay as authored); extra segments are clones of the last
    card with fresh node and link IDs; surplus cards and their report previews
    are dropped. Subgraph definitions, loaders and all other nodes are unchanged.
    """
    info = analyze(template, object_info, settings, run_id)
    option = _option_for(info, plan)
    segments = plan["segments"] if segments is None else segments
    if len(segments) != option["segment_count"]:
        raise ValueError(f"{option['id']} needs exactly {option['segment_count']} segments")
    setup = info["setup"]
    workflow = copy.deepcopy(template)
    raw_links = workflow.get("links", [])
    links = _links(workflow)
    sample = raw_links[0] if raw_links else None
    nodes = {n["id"]: n for n in workflow["nodes"]}
    chain, count = info["chain"], option["segment_count"]
    keep, drop = chain[:count], set(chain[count:])
    drop |= {info["previews"][card] for card in chain[count:] if card in info["previews"]}
    final = nodes[setup["final"]["id"]]
    video = _slot(final, "video")
    final_link = final["inputs"][video].get("link")
    removed = {link[0] for link in links if link[1] in drop or link[3] in drop}
    link_types = {(link[3], link[4]): link[5] for link in links}
    final_type = link_types.get((final["id"], video), "VIDEO")
    removed.add(final_link)
    workflow["links"] = [raw for raw, link in zip(raw_links, links) if link[0] not in removed]
    workflow["nodes"] = [n for n in workflow["nodes"] if n["id"] not in drop]
    for node in workflow["nodes"]:
        for inp in node.get("inputs", []):
            if inp.get("link") in removed:
                inp["link"] = None
        for output in node.get("outputs", []):
            if output.get("links"):
                output["links"] = [x for x in output["links"] if x not in removed]
    next_node = max([workflow.get("last_node_id", 0)] + [n["id"] for n in template["nodes"]]) + 1
    next_link = max([workflow.get("last_link_id", 0)] + [link[0] for link in links]) + 1

    def connect(origin: dict, origin_slot: int, target: dict, target_slot: int, kind) -> None:
        nonlocal next_link
        workflow["links"].append(_make_link(sample, (next_link, origin["id"], origin_slot, target["id"],
                                                     target_slot, kind)))
        if "outputs" in origin:  # Nodes without output records keep none (validator skips them).
            output = origin["outputs"][origin_slot]
            output["links"] = [*(output.get("links") or []), next_link]
        target["inputs"][target_slot]["link"] = next_link
        next_link += 1

    template_last = nodes[chain[-1]]
    project = nodes[setup["project"]["id"]]
    project_type = link_types.get((chain[0], _slot(nodes[chain[0]], "project")), "H3RC_PROJECT")
    previous_type = (link_types.get((chain[1], _slot(nodes[chain[1]], "previous")))
                     if len(chain) > 1 else None) or (nodes[chain[0]].get("outputs") or [{}])[0].get("type", "*")
    step = ([nodes[chain[-1]]["pos"][0] - nodes[chain[-2]]["pos"][0], nodes[chain[-1]]["pos"][1] - nodes[chain[-2]]["pos"][1]]
            if len(chain) > 1 and all("pos" in nodes[c] for c in chain[-2:]) else [480, 0])
    cards = [nodes[card] for card in keep]
    for index in range(len(keep), count):
        card = copy.deepcopy(template_last)
        card["id"] = next_node
        next_node += 1
        card["title"] = f"SEGMENT {index + 1:02d}"
        if "pos" in card and "pos" in template_last:
            offset = index - len(chain) + 1
            card["pos"] = [template_last["pos"][0] + step[0] * offset, template_last["pos"][1] + step[1] * offset]
        for inp in card.get("inputs", []):
            inp["link"] = None
        for output in card.get("outputs", []):
            output["links"] = []
        workflow["nodes"].append(card)
        connect(project, 0, card, _slot(card, "project"), project_type)
        connect(cards[-1], 0, card, _slot(card, "previous"), previous_type)
        cards.append(card)
    connect(cards[-1], 1, final, video, final_type)

    base_seed = dict(zip(_widget_names(nodes[chain[0]]), nodes[chain[0]].get("widgets_values", []))).get("seed")
    for index, (card, segment) in enumerate(zip(cards, segments)):
        names = _widget_names(card)
        values = list(card["widgets_values"])
        updates = {"duration_override": option["segments"][index]["duration_override"], "steps_override": 0,
                   "prompt": render_prompt(segment["sections"], info["sections"], info["raw_prompt"])}
        if isinstance(base_seed, int) and not isinstance(base_seed, bool):
            updates["seed"] = base_seed + index
        for name, value in updates.items():
            if name in names:
                values[names.index(name)] = value
                if isinstance(card.get("widgets_values_named"), dict):
                    card["widgets_values_named"][name] = value
        card["widgets_values"] = values

    mapping = {m["input"]: m["widget_index"] for m in _mapping(project, object_info, info["segment_type"])}
    values = project["widgets_values"]
    for name, value in (("width", setup["width"]), ("height", setup["height"]), ("size_mode", "width_height"),
                        ("default_steps", settings.steps)):
        if name in mapping and value is not None and mapping[name] < len(values):
            values[mapping[name]] = value
    for slot in info["reference_slots"]:
        loader = nodes[int(slot["node_id"])]
        loader["widgets_values"][0] = plan["references"][slot["node_id"]]
    workflow["last_node_id"] = max(n["id"] for n in workflow["nodes"])
    workflow["last_link_id"] = max(_normal_link(link)[0] for link in workflow["links"])
    return workflow


def compile_variations(template: dict, plan: dict, settings, run_ids: list[str], *, object_info: dict) -> list[dict]:
    """Qty > 1: one workflow per variation (shared layout and references); seeds per variations.py."""
    from .variations import variant_workflow
    info = analyze(template, object_info, settings, run_ids[0])
    groups = [plan["segments"]] + [variant["segments"] for variant in plan.get("variations", [])]
    if len(groups) != len(run_ids):
        raise ValueError("Variation count differs from the saved sequence")
    return [variant_workflow(compile_workflow(template, plan, settings, run_id, object_info=object_info,
                                              segments=segments), [], info["segment_type"], run_id)
            for segments, run_id in zip(groups, run_ids)]


def dumps(workflow: dict) -> bytes:
    """Canonical bytes for a compiled workflow."""
    return json.dumps(workflow, indent=2, ensure_ascii=False, allow_nan=False).encode("utf-8")


def support_problem(template: dict, object_info: dict, settings) -> str | None:
    """Why the compiler cannot build this template for these settings, or None.

    A dry run with placeholder content: every timing option is compiled and
    validated, so a template that passes cannot fail at compile time after
    agent work has been spent.
    """
    from .workflow_edits import validate_edited_workflow
    try:
        info = analyze(template, object_info, settings, "preflight")
        for option in info["options"]:
            plan = {"script": "", "used_references": [], "timing_option": option["id"],
                    "references": {slot["node_id"]: "/preflight/placeholder.png" for slot in info["reference_slots"]},
                    "segments": [{"sections": {name: f"Preflight placeholder {name} {index + 1}."
                                               for name in info["sections"]}}
                                 for index in range(option["segment_count"])]}
            workflow = compile_workflow(template, plan, settings, "preflight", object_info=object_info)
            validate_edited_workflow(template, workflow, object_info, megapixels=settings.megapixels,
                                     duration=settings.duration, aspect_ratio=settings.aspect_ratio,
                                     steps=settings.steps, run_id="preflight", check_model_availability=False)
    except (ValueError, KeyError, TypeError, IndexError, StopIteration, AttributeError, ZeroDivisionError) as error:
        return str(error) or type(error).__name__
    return None
