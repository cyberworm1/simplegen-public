"""Render the six-block Ref2VA prompt from a spec.

Block renderers are registered in BLOCKS, in output order. To add or change a
block, write a function `(spec, refs) -> str` and put it in BLOCKS; users can
also replace any block (spec.blocks[name]) or append to it (spec.append[name])
without touching code.
"""

from __future__ import annotations

from collections.abc import Callable

from .refs import Ref, resolve_tokens, speaker_code
from .spec import Spec, Voice

COPY_WORDING = {
    "partially_copy": "keep part of the original signal in the soundtrack",
    "fully_copy": "keep the original signal in the soundtrack",
}


def _join_tags(tags: list[str]) -> str:
    if len(tags) <= 1:
        return "".join(tags)
    return ", ".join(tags[:-1]) + (", and " if len(tags) > 2 else " and ") + tags[-1]


def _voices_for(spec: Spec, subject_id: str) -> list[Voice]:
    return [v for v in spec.voices if v.subject == subject_id]


def _audio_items(spec: Spec, refs: dict[str, Ref]):
    """(tag, mode, subject_id, description, delivery) for every reference audio, in label order."""
    items = []
    for v in spec.videos:
        if v.use_soundtrack:
            note = v.soundtrack_note or f"the soundtrack of {refs[v.id].tag}"
            items.append((refs[f"{v.id}#audio"].tag, v.soundtrack_mode, "", note, ""))
    for v in spec.voices:
        items.append((refs[v.id].tag, v.mode, v.subject, v.note, v.delivery))
    return items


def subject_definitions(spec: Spec, refs: dict[str, Ref]) -> str:
    out = []
    for s in spec.subjects:
        tags = [refs[p].tag for p in s.pictures] + [refs[v].tag for v in s.videos]
        seen = f" in {_join_tags(tags)}" if tags else ""
        keep = f", {s.keep}" if s.keep else ""
        out.append(f"{refs[s.id].tag} is {s.who}{seen}{keep}.")
    for p in spec.pictures:
        if p.note:
            out.append(f"{refs[p.id].tag} shows {p.note}.")
    for tag, mode, subject, note, _ in _audio_items(spec, refs):
        if subject and mode == "reference":
            voice = next(v for v in spec.voices if refs[v.id].tag == tag)
            desc = note or f"containing a spoken {voice.language} vocal layer"
            out.append(
                f"{tag} is the voice-timbre reference for {refs[subject].tag} "
                f"({speaker_code(refs, subject)}), {desc}."
            )
        elif subject:
            out.append(f"{tag} is a recording of {refs[subject].tag}'s voice{', ' + note if note else ''}.")
        else:
            out.append(f"{tag} is {note or 'a reference audio clip'}.")
    return "\n".join(out)


def summary(spec: Spec, refs: dict[str, Ref]) -> str:
    modes = {mode for _, mode, *_ in _audio_items(spec, refs)}
    task = ["reference generation"]
    if "reference" in modes:
        task.append("audio reference")
    if modes & set(COPY_WORDING):
        task.append("audio reuse")
    speakers = []
    for shot in spec.shots:
        for line in shot.lines:
            if refs[line.speaker].tag not in speakers:
                speakers.append(refs[line.speaker].tag)
    if speakers:
        body = f"The target video shows {_join_tags(speakers)} speaking the lines below."
    else:
        body = "The target video shows the scene described below."
    notes = []
    for tag, mode, subject, *_ in _audio_items(spec, refs):
        if mode == "reference" and subject:
            notes.append(f"{tag} is used only as a voice-timbre reference for {refs[subject].tag}.")
        elif mode == "reference":
            notes.append(f"{tag} is used only as a style reference.")
        else:
            notes.append(f"{tag} is reused: {COPY_WORDING[mode]}.")
    return " ".join([f"[{' + '.join(task)}] {body}"] + notes)


def retention_analysis(spec: Spec, refs: dict[str, Ref]) -> str:
    out = []
    for s in spec.subjects:
        shots = [f"[Shot {i}]" for i, shot in enumerate(spec.shots, 1)
                 if f"[[{s.id}]]" in shot.description or any(l.speaker == s.id for l in shot.lines)]
        where = f" (appears in {', '.join(shots)})" if shots else ""
        out.append(f"{refs[s.id].tag}{where}: fully_preserved - {s.retention}.")
    for tag, mode, subject, _, delivery in _audio_items(spec, refs):
        if mode == "reference":
            who = f" guide {refs[subject].tag}" if subject else " guide the soundtrack"
            quality = f"vocal timbre and {delivery} delivery" if subject and delivery else "timbre and delivery"
            out.append(f"{tag}: reference - {quality}{who} without copying the original signal.")
        else:
            out.append(f"{tag}: {mode} - {COPY_WORDING[mode]}.")
    return "\n".join(out)


def _line(spec: Spec, refs: dict[str, Ref], line) -> str:
    language = line.language or next(
        (v.language for v in _voices_for(spec, line.speaker)), "English"
    )
    text = line.text.strip()
    return (f"{refs[line.speaker].tag} ({speaker_code(refs, line.speaker)}) {line.verb.strip()} "
            f"<d>[{language}] {text}</d>")


def detailed_description(spec: Spec, refs: dict[str, Ref]) -> str:
    out = [f"The target video is {spec.style}."]
    for i, shot in enumerate(spec.shots, 1):
        parts = [f"[Shot {i}]", resolve_tokens(shot.description.strip(), refs)]
        parts += [_line(spec, refs, line) for line in shot.lines]
        if shot.after:
            parts.append(resolve_tokens(shot.after.strip(), refs))
        out.append(" ".join(p for p in parts if p))
    return "\n".join(out)


def overall_soundscape(spec: Spec, refs: dict[str, Ref]) -> str:
    text = resolve_tokens(spec.soundscape.strip(), refs)
    if spec.no_extra_speech and any(shot.lines for shot in spec.shots):
        text = f"{text} No extra speech after the last line ends.".strip()
    return text


def non_diegetic_music(spec: Spec, refs: dict[str, Ref]) -> str:
    return resolve_tokens(spec.music.strip() or "none", refs)


BLOCKS: list[tuple[str, Callable[[Spec, dict[str, Ref]], str]]] = [
    ("subject_definitions", subject_definitions),
    ("summary", summary),
    ("retention_analysis", retention_analysis),
    ("detailed_description", detailed_description),
    ("overall_soundscape", overall_soundscape),
    ("non_diegetic_music", non_diegetic_music),
]


def render(spec: Spec, refs: dict[str, Ref]) -> str:
    if spec.prompt_override:
        return resolve_tokens(spec.prompt_override.strip(), refs) + "\n"
    names = [name for name, _ in BLOCKS]
    for key in list(spec.blocks) + list(spec.append):
        if key not in names:
            raise ValueError(f"unknown prompt block {key!r}; blocks are {names}")
    sections = []
    for name, fn in BLOCKS:
        body = resolve_tokens(spec.blocks[name], refs) if name in spec.blocks else fn(spec, refs)
        if name in spec.append:
            body = f"{body}\n{resolve_tokens(spec.append[name], refs)}".strip()
        sections.append(f"{name}:\n{body.strip()}")
    return "\n\n".join(sections) + "\n"
