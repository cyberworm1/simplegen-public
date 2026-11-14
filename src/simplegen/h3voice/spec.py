"""Shot spec: the single source of truth a user or agent edits.

A spec is plain JSON (see shots/*.json). Everything else — reference numbering,
the six-block prompt, frame counts and the ComfyUI graph — is derived from it.
Refer to any picture, video, voice or subject in free text as [[id]]; the
renderer replaces it with the right <Picture N> / <Video N> / <Audio N> /
<Subject N> tag, so reordering references never breaks a prompt.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

RETENTION_MODES = ("reference", "partially_copy", "fully_copy")


class SpecError(ValueError):
    pass


@dataclass
class Picture:
    id: str
    file: str
    note: str = ""  # what this picture shows, used in subject_definitions


@dataclass
class Video:
    id: str
    file: str
    note: str = ""
    use_soundtrack: bool = False  # its audio becomes an <Audio j> right before <Video k>
    soundtrack_mode: str = "reference"
    soundtrack_note: str = ""


@dataclass
class Voice:
    """A standalone reference audio clip (ref_audios)."""

    id: str
    file: str
    subject: str = ""  # subject whose voice this is; empty for non-voice audio (BGM, ambience)
    mode: str = "reference"  # reference = timbre only; *_copy = keep the waveform
    start: float = 0.0  # trim start (seconds, negative counts from the end)
    duration: float | None = None  # trim length; None keeps the rest of the file
    language: str = "English"
    delivery: str = "measured"  # adjective for how the reference sounds
    note: str = ""  # replaces the default description when set


@dataclass
class Subject:
    id: str
    who: str  # "the woman", "the older man in the grey coat"
    pictures: list[str] = field(default_factory=list)
    videos: list[str] = field(default_factory=list)
    keep: str = "same face, hair, and wardrobe"
    retention: str = "face, hair, and clothing stay consistent"


@dataclass
class Line:
    speaker: str
    text: str
    language: str = ""  # defaults to the speaker's voice language, else English
    verb: str = "says"  # everything between "(S1)" and <d>, e.g. "says quietly,"


@dataclass
class Shot:
    description: str  # camera + action; may use [[id]]
    lines: list[Line] = field(default_factory=list)
    after: str = ""  # text after the dialogue, e.g. a reaction


@dataclass
class Guide:
    """Pin a picture (already in `pictures`) and/or a voice to an output time."""

    at_seconds: float
    picture: str = ""
    voice: str = ""


@dataclass
class Spec:
    name: str
    pictures: list[Picture] = field(default_factory=list)
    videos: list[Video] = field(default_factory=list)
    voices: list[Voice] = field(default_factory=list)
    subjects: list[Subject] = field(default_factory=list)
    shots: list[Shot] = field(default_factory=list)
    guides: list[Guide] = field(default_factory=list)
    style: str = "realistic cinematic"
    soundscape: str = "Quiet room tone under the spoken line."
    no_extra_speech: bool = True
    music: str = "none"
    duration: float | None = None  # seconds; None = estimate from dialogue
    seed: int = 1
    # Per-shot overrides of the profile (width/height/megapixels/aspect/steps/...).
    render: dict[str, Any] = field(default_factory=dict)
    # Replace (blocks) or extend (append) any prompt block by name.
    blocks: dict[str, str] = field(default_factory=dict)
    append: dict[str, str] = field(default_factory=dict)
    prompt_override: str = ""  # full hand-written prompt; [[id]] still resolved
    base_dir: Path = field(default=Path("."), repr=False)

    def resolve_file(self, name: str) -> Path:
        path = Path(name).expanduser()
        return path if path.is_absolute() else (self.base_dir / path).resolve()


def _build(cls, data: Any, where: str):
    if not isinstance(data, dict):
        raise SpecError(f"{where}: expected an object")
    known = {f.name for f in fields(cls)} - {"base_dir"}
    unknown = set(data) - known
    if unknown:
        raise SpecError(f"{where}: unknown keys {sorted(unknown)}; allowed {sorted(known)}")
    try:
        return cls(**data)
    except TypeError as exc:
        raise SpecError(f"{where}: {exc}") from None


def from_dict(data: dict, base_dir: Path | str = ".") -> Spec:
    data = dict(data)
    data.pop("$schema", None)
    data.pop("_comment", None)
    nested = {
        "pictures": Picture, "videos": Video, "voices": Voice,
        "subjects": Subject, "guides": Guide,
    }
    for key, cls in nested.items():
        data[key] = [_build(cls, item, f"{key}[{i}]") for i, item in enumerate(data.get(key, []))]
    shots = []
    for i, raw in enumerate(data.get("shots", [])):
        raw = dict(raw) if isinstance(raw, dict) else raw
        if isinstance(raw, dict):
            raw["lines"] = [_build(Line, x, f"shots[{i}].lines[{j}]") for j, x in enumerate(raw.get("lines", []))]
        shots.append(_build(Shot, raw, f"shots[{i}]"))
    data["shots"] = shots
    spec = _build(Spec, data, "spec")
    spec.base_dir = Path(base_dir)
    _check_ids(spec)
    _check_refs(spec)
    return spec


def load(path: Path | str) -> Spec:
    path = Path(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SpecError(f"{path}: invalid JSON: {exc}") from None
    return from_dict(data, path.parent.resolve())


def _check_ids(spec: Spec) -> None:
    seen: set[str] = set()
    for group in (spec.pictures, spec.videos, spec.voices, spec.subjects):
        for item in group:
            if not item.id or not item.id.replace("_", "").replace("-", "").isalnum():
                raise SpecError(f"invalid id {item.id!r}: use letters, digits, _ or -")
            if item.id in seen:
                raise SpecError(f"duplicate id {item.id!r}")
            seen.add(item.id)
    for v in spec.voices:
        if v.mode not in RETENTION_MODES:
            raise SpecError(f"voice {v.id}: mode must be one of {RETENTION_MODES}")
    for v in spec.videos:
        if v.soundtrack_mode not in RETENTION_MODES:
            raise SpecError(f"video {v.id}: soundtrack_mode must be one of {RETENTION_MODES}")


def _check_refs(spec: Spec) -> None:
    """Dangling ids fail at load time, before any prompt text is rendered."""
    pictures = {p.id for p in spec.pictures}
    videos = {v.id for v in spec.videos}
    voices = {v.id for v in spec.voices}
    subjects = {s.id for s in spec.subjects}

    def need(value: str, pool: set, what: str, where: str) -> None:
        if value and value not in pool:
            raise SpecError(f"{where}: {value!r} is not a {what} id (known: {sorted(pool) or 'none'})")

    for s in spec.subjects:
        for p in s.pictures:
            need(p, pictures, "picture", f"subject {s.id}")
        for v in s.videos:
            need(v, videos, "video", f"subject {s.id}")
    for v in spec.voices:
        need(v.subject, subjects, "subject", f"voice {v.id}")
    for i, shot in enumerate(spec.shots, 1):
        for line in shot.lines:
            if not line.speaker:
                raise SpecError(f"shot {i}: every line needs a speaker")
            need(line.speaker, subjects, "subject", f"shot {i} line")
    for g in spec.guides:
        need(g.picture, pictures, "picture", f"guide at {g.at_seconds}s")
        need(g.voice, voices, "voice", f"guide at {g.at_seconds}s")
