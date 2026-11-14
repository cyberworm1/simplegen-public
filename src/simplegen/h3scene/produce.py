"""H3 Scene Chain agent message and plan checks (dialogue, pictures, card constraints)."""
import math
import re
from pathlib import Path
from . import plan
from .room import RoomError, normalize_line, plan_lines


COMPONENT = re.compile(r"[A-Za-z0-9_-]+")


OWNER_DIALOGUE = re.compile(r"<d>\s*(?:\[[^\]\n]+\]\s*)?(.*?)</d>", re.S)


NAMED_LINE = re.compile(r"^\s*([A-Z][\w' -]*):\s*(.*)$")


HEADERS = {"TITLE", "CONCEPT", "DURATION", "PACE", "CAST", "PLACE", "NOTE", "NOTES"}


SCREENPLAY_CUE = re.compile(r"([A-Z][A-Z'’ -]*?)(?:\s*\([^()]*\))?")


SCENE_HEADING = re.compile(r"^(?:INT\.|EXT\.)", re.I)


TRANSITION = re.compile(r"^(?:[A-Z][A-Z -]* TO:|FADE (?:IN|OUT)[.:]?|"
                        r"FADE TO BLACK[.:]?|(?:CUT|DISSOLVE|WIPE|IRIS)[.:])$")


def build_messages(master, owner, seconds, notes, cid, library):
    root = Path(library).resolve()
    paths = [root / "INDEX.md"]
    for category in ("characters", "places", "props"):
        paths.extend(sorted((root / category).rglob("card.md")))
    library = "\n\n".join(
        f"=== {path.relative_to(root).as_posix()} ===\n{path.read_text(encoding='utf-8')}"
        for path in paths)
    expectations = (f"# Expectations\nConcept id: {cid}\nRequested seconds: {seconds:g}\n"
                    + "".join(f"Note: {note}\n" for note in notes))
    return [{"role": "system", "content": master},
            {"role": "user", "content": expectations + "\n# Owner text (verbatim)\n"
             + owner + "\n\n# Library\n" + library}]


def owner_dialogue(owner):
    """Return (line, strict) pairs; screenplay paragraphs need owner verification."""
    if "<d>" in owner or "</d>" in owner:
        matches = list(OWNER_DIALOGUE.finditer(owner))
        if owner.count("<d>") != len(matches) or owner.count("</d>") != len(matches):
            raise ValueError("owner text has malformed <d> tags")
        lines = [(normalize_line(match[1]), True) for match in matches]
    else:
        lines = []
        in_block = False
        paragraph = []

        def flush():
            if paragraph:
                lines.append((normalize_line(" ".join(paragraph)), False))
                paragraph.clear()

        for raw in owner.splitlines():
            line = raw.strip()
            cue = SCREENPLAY_CUE.fullmatch(line)
            named = NAMED_LINE.fullmatch(line)
            if SCENE_HEADING.match(line) or TRANSITION.fullmatch(line):
                flush()
                in_block = False
            elif named:
                flush()
                in_block = False
                if named[1].upper() not in HEADERS:
                    lines.append((normalize_line(named[2]), True))
            elif cue:
                flush()
                in_block = cue[1] not in HEADERS
            elif not line:
                flush()
            elif in_block and not (line.startswith("(") and line.endswith(")")):
                paragraph.append(line)
        flush()
    if any(not line for line, _ in lines):
        raise ValueError("owner text has an empty spoken line")
    return lines


def owner_lines(owner):
    """Extract explicit dialogue and screenplay paragraphs, never arbitrary quotes."""
    return [line for line, _ in owner_dialogue(owner)]


def dialogue_checks(owner, candidate):
    try:
        expected, actual = owner_dialogue(owner), plan_lines(candidate)
    except (ValueError, RoomError, KeyError, TypeError, AttributeError) as exc:
        return [f"verbatim: {exc}"], []
    # Connective dialogue is allowed; every owner line must survive in order,
    # including repeated lines. A consumed occurrence cannot match twice.
    cursor = 0
    errors = []
    warnings = []
    for i, (line, strict) in enumerate(expected, 1):
        try:
            cursor = actual.index(line, cursor) + 1
        except ValueError:
            prefix = "verbatim" if strict else "owner_dialogue_unverified"
            (errors if strict else warnings).append(
                f"{prefix}: owner line {i} missing, changed or out of order: {line!r}")
    if not expected and owner.strip():
        warnings.append("owner_dialogue_unverified: no recognized owner dialogue; "
                        "coverage could not be checked")
    source = normalize_line(owner)
    for i, line in enumerate(actual, 1):
        if line not in source:
            warnings.append(f"connective_lines: plan line {i} not found verbatim in owner text: {line!r}")
    return errors, warnings


def verbatim_errors(owner, candidate):
    """Only strict dialogue errors may instruct the provider to retry."""
    return dialogue_checks(owner, candidate)[0]


def picture_errors(candidate, root):
    """Pictures must be existing files inside the chosen library folder."""
    errors = []
    for pic in candidate.get("pictures") or []:
        path = pic.get("path") if isinstance(pic, dict) else None
        if not isinstance(path, str) or not path:
            continue
        found = (root / path).resolve()
        if not found.is_relative_to(root) or not found.is_file():
            errors.append(f"picture not found: {path}")
    return errors


VERBATIM_HEADING = re.compile(r"^## Verbatim constraints[ \t]*$", re.M)


def verbatim_block(card):
    """Text under '## Verbatim constraints' (to the next heading), whitespace-normalised."""
    found = VERBATIM_HEADING.search(card)
    if not found:
        return ""
    rest = card[found.end():]
    following = re.search(r"^#{1,6} ", rest, re.M)
    return " ".join((rest[:following.start()] if following else rest).split())


def constraint_errors(candidate, root):
    """Each picture's card block must appear verbatim wherever its subject is defined."""
    errors = []
    pictures, segments = candidate.get("pictures"), candidate.get("segments")
    if not isinstance(pictures, list) or not isinstance(segments, list):
        return errors
    for n, pic in enumerate(pictures, 1):
        path = pic.get("path") if isinstance(pic, dict) else None
        if not isinstance(path, str) or not path:
            continue
        card = (root / path).resolve().parent / "card.md"
        if not card.is_relative_to(root) or not card.is_file():
            continue
        block = verbatim_block(card.read_text(encoding="utf-8"))
        if not block:
            continue
        for i, segment in enumerate(segments, 1):
            sections = segment.get("sections") if isinstance(segment, dict) else None
            text = sections.get("subject_definitions") if isinstance(sections, dict) else None
            if not isinstance(text, str):
                continue
            text = " ".join(text.split())
            if f"<Picture {n}>" in text and block not in text:
                errors.append(f"seg {i}: subject_definitions for <Picture {n}> ({path}) must contain "
                              f"its card's Verbatim constraints word for word: {block!r}")
    return errors


def check_candidate(candidate, owner, seconds, library):
    """Aggregate plan, dialogue and duration errors; compute the review flag in code."""
    if not isinstance(candidate, dict):
        return ["plan must be an object"], {}
    errors = [f"missing {k}" for k in ("concept_id", "title", "script", "pictures", "segments")
              if k not in candidate]
    pictures, segments = candidate.get("pictures"), candidate.get("segments")
    # Shape failures must not prevent the existing checker from reporting
    # independent path, join, length and block failures in the same response.
    checked = dict(candidate)
    for key in ("concept_id", "title", "script"):
        checked.setdefault(key, "")
    if not isinstance(pictures, list) or any(
            not isinstance(p, dict) or not isinstance(p.get("path"), str) for p in pictures):
        errors.append("pictures must be a list of objects with text paths")
    checked["pictures"] = [p if isinstance(p, dict) and isinstance(p.get("path"), str)
                           else {"path": ""} for p in pictures] if isinstance(pictures, list) else []
    if not isinstance(segments, list) or not segments:
        errors.append("segments must be a non-empty list")
        segments = []
    checked["segments"] = []
    total = 0.0
    duration_valid = bool(segments)
    for i, segment in enumerate(segments, 1):
        if not isinstance(segment, dict):
            errors.append(f"seg {i}: must be an object")
            duration_valid = False
            checked["segments"].append({"sections": {}})
            continue
        checked_segment = dict(segment)
        checked["segments"].append(checked_segment)
        value = segment.get("seconds")
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value)):
            errors.append(f"seg {i}: seconds must be a finite number")
            duration_valid = False
            checked_segment["seconds"] = 0
        else:
            total += value
        sections = segment.get("sections")
        if not isinstance(sections, dict) or any(not isinstance(v, str) for v in sections.values()):
            errors.append(f"seg {i}: sections must contain text blocks")
            checked_segment["sections"] = {k: v if isinstance(v, str) else ""
                                           for k, v in sections.items()} if isinstance(sections, dict) else {}
    root = Path(library).resolve()
    errors.extend(plan.check_plan(checked))
    errors.extend(picture_errors(checked, root))
    errors.extend(constraint_errors(checked, root))
    if candidate.get("input_mode") not in ("prompt", "script", "concept", "mixed"):
        errors.append("input_mode must be prompt, script, concept or mixed")
    warnings = candidate.get("warnings", [])
    if not isinstance(warnings, list) or any(not isinstance(w, str) for w in warnings):
        errors.append("warnings must be a list of strings")
    dialogue_errors, dialogue_warnings = dialogue_checks(owner, checked)
    errors.extend(dialogue_errors)
    if isinstance(warnings, list) and all(isinstance(w, str) for w in warnings):
        candidate["warnings"] = warnings + [w for w in dialogue_warnings if w not in warnings]
    overflow = candidate.get("overflow", {})
    if not isinstance(overflow, dict):
        errors.append("overflow must be an object")
        overflow = {}
    if "needs_review" in overflow and not isinstance(overflow["needs_review"], bool):
        errors.append("overflow.needs_review must be a boolean")
    duration = {}
    if duration_valid:
        extra = max(0.0, total - seconds)
        needs_review = extra > 15 or overflow.get("needs_review") is True
        duration = {"requested_seconds": seconds, "total_seconds": total,
                    "extra_seconds": extra, "needs_review": needs_review}
        overflow = dict(overflow, extra_seconds=extra, needs_review=needs_review)
        if needs_review:
            overflow.setdefault("reason", f"Plan requires {total:g}s for a {seconds:g}s request.")
            overflow["options"] = "approve / edit / deny"
        candidate["overflow"] = overflow
    return errors, duration
