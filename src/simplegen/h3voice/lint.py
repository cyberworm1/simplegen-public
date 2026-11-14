"""Deterministic checks for a spec and its rendered prompt.

Each check is a function `(ctx) -> None` that calls ctx.error / ctx.warn.
Add a check by writing one and appending it to CHECKS. Errors block building;
warnings are printed and recorded. A check that cannot run (missing ffprobe,
missing file) says so instead of passing silently.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from . import refs as R
from .spec import Spec


@dataclass
class Issue:
    level: str  # error | warning | unverified
    code: str
    message: str

    def __str__(self) -> str:
        return f"{self.level.upper():10} {self.code}: {self.message}"


@dataclass
class Ctx:
    spec: Spec
    refs: dict
    prompt: str
    check_files: bool = True
    issues: list[Issue] = field(default_factory=list)
    audio_seconds: dict[str, float | None] = field(default_factory=dict)

    def error(self, code: str, msg: str) -> None:
        self.issues.append(Issue("error", code, msg))

    def warn(self, code: str, msg: str) -> None:
        self.issues.append(Issue("warning", code, msg))

    def unverified(self, code: str, msg: str) -> None:
        self.issues.append(Issue("unverified", code, msg))


def probe_seconds(path: Path) -> float | None:
    if not shutil.which("ffprobe"):
        return None
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(path)],
            capture_output=True, text=True, timeout=30, check=True,
        )
        return float(json.loads(out.stdout)["format"]["duration"])
    except (subprocess.SubprocessError, KeyError, ValueError):
        return None


def check_counts(c: Ctx) -> None:
    s = c.spec
    n_audio = len(s.voices) + sum(v.use_soundtrack for v in s.videos)
    if len(s.pictures) > R.MAX_IMAGES:
        c.error("too_many_pictures", f"{len(s.pictures)} pictures; Ref2VA takes at most {R.MAX_IMAGES}")
    if len(s.videos) > R.MAX_VIDEOS:
        c.error("too_many_videos", f"{len(s.videos)} videos; at most {R.MAX_VIDEOS}")
    if len(s.voices) > R.MAX_AUDIOS:
        c.error("too_many_audios", f"{len(s.voices)} standalone audio clips; at most {R.MAX_AUDIOS}")
    if len(s.pictures) + len(s.videos) + len(s.voices) > R.MAX_FILES:
        c.error("too_many_files", f"more than {R.MAX_FILES} reference files")
    if n_audio and not (s.pictures or s.videos):
        c.error("audio_alone", "reference audio must travel with at least one picture or video")


def check_links(c: Ctx) -> None:
    s = c.spec
    kinds = {k: r.kind for k, r in c.refs.items()}
    for subj in s.subjects:
        for p in subj.pictures:
            if kinds.get(p) != "picture":
                c.error("bad_ref", f"subject {subj.id}: {p!r} is not a picture id")
        for v in subj.videos:
            if kinds.get(v) != "video":
                c.error("bad_ref", f"subject {subj.id}: {v!r} is not a video id")
        if not subj.pictures and not subj.videos:
            c.warn("unanchored_subject", f"subject {subj.id} has no picture or video; identity is text-only")
    for v in s.voices:
        if v.subject and kinds.get(v.subject) != "subject":
            c.error("bad_ref", f"voice {v.id}: subject {v.subject!r} is not a subject id")
    for g in s.guides:
        if g.picture and kinds.get(g.picture) != "picture":
            c.error("bad_ref", f"guide at {g.at_seconds}s: {g.picture!r} is not a picture id")
        if g.voice and kinds.get(g.voice) != "voice":
            c.error("bad_ref", f"guide at {g.at_seconds}s: {g.voice!r} is not a voice id")
        if not g.picture and not g.voice:
            c.error("empty_guide", f"guide at {g.at_seconds}s needs a picture or a voice")
        if not 0 <= g.at_seconds * R.FPS < R.frame_count(s):
            c.error("guide_outside", f"guide at {g.at_seconds}s is outside the {R.target_seconds(s)}s clip")
    for i, shot in enumerate(s.shots, 1):
        for line in shot.lines:
            if kinds.get(line.speaker) != "subject":
                c.error("bad_speaker", f"shot {i}: speaker {line.speaker!r} is not a subject id")
            elif not any(v.subject == line.speaker for v in s.voices):
                c.warn("no_voice", f"shot {i}: {line.speaker} speaks but has no voice reference; the voice will be invented")


def check_dialogue(c: Ctx) -> None:
    for i, shot in enumerate(c.spec.shots, 1):
        for line in shot.lines:
            text = line.text.strip()
            if not text:
                c.error("empty_line", f"shot {i}: empty dialogue line")
                continue
            if re.search(r"[<>]", text):
                c.error("tag_in_line", f"shot {i}: dialogue text must not contain < or >: {text!r}")
            if text[-1] not in ".!?…":
                c.warn("line_end", f"shot {i}: end spoken lines with a period (fewer mumbled endings): {text!r}")
    opens, closes = c.prompt.count("<d>"), c.prompt.count("</d>")
    if opens != closes:
        c.error("unbalanced_d", f"prompt has {opens} <d> and {closes} </d>")


def check_timing(c: Ctx) -> None:
    s = c.spec
    frames = R.frame_count(s)
    seconds = frames / R.FPS
    lo, hi = R.TRAINED_FRAMES
    if not lo <= frames <= hi:
        c.warn("untrained_length", f"{frames} frames ({seconds:.2f}s) is outside the trained {lo}-{hi} frame range")
    words = R.spoken_words(s)
    budget = seconds * R.WORDS_PER_SECOND
    if words > budget * 1.5 + 3:
        c.error("speech_budget", f"{words} words in {seconds:.1f}s; budget is ~{budget:.0f} (2.5 words/s)")
    elif words > budget * 1.2 + 3:
        c.warn("speech_budget", f"{words} words in {seconds:.1f}s is tight; budget ~{budget:.0f}")


def check_audio(c: Ctx) -> None:
    s = c.spec
    total, unknown = 0.0, []
    speech = max(R.spoken_words(s) / R.WORDS_PER_SECOND, 1.0)
    for v in s.voices:
        path = s.resolve_file(v.file)
        length = probe_seconds(path) if path.exists() else None
        if length is not None:
            start = v.start if v.start >= 0 else max(0.0, length + v.start)
            if start >= length:
                c.error("trim_past_end", f"voice {v.id} starts at {start:.1f}s but the file is only {length:.1f}s long")
            length = max(0.0, length - start)
            if v.duration is not None:
                length = min(length, v.duration)
        elif v.duration is not None:
            length = v.duration
        c.audio_seconds[v.id] = length
        if length is None:
            unknown.append(v.id)
            continue
        total += length
        if length < 2.0:
            c.warn("short_audio", f"voice {v.id} is {length:.1f}s; clips under ~2s give a weak timbre reference")
        if v.mode == "reference" and length > speech * 1.6 + 2:
            c.warn("long_audio", f"voice {v.id} is {length:.1f}s for ~{speech:.1f}s of speech; long clips leak "
                   f"extra speech — set duration near the line length")
    for v in s.videos:
        if v.use_soundtrack:
            path = s.resolve_file(v.file)
            length = probe_seconds(path) if path.exists() else None
            if length is None:
                unknown.append(v.id)
            else:
                total += min(length, R.target_seconds(s))
    if total > R.MAX_AUDIO_SECONDS + 1e-6:
        c.error("audio_total", f"reference audio totals {total:.1f}s; the limit is {R.MAX_AUDIO_SECONDS:.0f}s")
    if unknown:
        c.unverified("audio_length", f"could not measure {', '.join(unknown)} (missing file or ffprobe); "
                     f"set voices[].duration to make the 15s check exact")


def check_files(c: Ctx) -> None:
    if not c.check_files:
        return
    s = c.spec
    for item in [*s.pictures, *s.videos, *s.voices]:
        if not s.resolve_file(item.file).is_file():
            c.error("missing_file", f"{item.id}: {s.resolve_file(item.file)} does not exist")


def check_prompt(c: Ctx) -> None:
    p = c.prompt
    counts = {
        "Picture": len(c.spec.pictures),
        "Video": len(c.spec.videos),
        "Audio": len(c.spec.voices) + sum(v.use_soundtrack for v in c.spec.videos),
        "Subject": len(c.spec.subjects),
    }
    for kind, n in re.findall(r"<(Picture|Video|Audio|Subject) (\d+)>", p):
        if int(n) > counts[kind]:
            c.error("tag_out_of_range", f"<{kind} {n}> used but only {counts[kind]} {kind.lower()}(s) are connected")
    if R.TOKEN.search(p):
        c.error("unresolved_token", "prompt still contains [[...]] tokens")
    has_ref_voice = any(v.mode == "reference" for v in c.spec.voices)
    if has_ref_voice and "without copying" not in p and "do not copy" not in p:
        c.warn("copy_leak", "voice references should say 'without copying the original signal'; "
               "otherwise H3 often replays the clip")
    if "<d>" in p and "overall_soundscape:" not in p:
        c.warn("no_soundscape", "write an overall_soundscape block, or the model tends to add leftover speech")


CHECKS = [check_counts, check_links, check_dialogue, check_timing, check_audio, check_files, check_prompt]


def run(spec: Spec, refs: dict, prompt: str, check_files: bool = True) -> Ctx:
    ctx = Ctx(spec, refs, prompt, check_files)
    for check in CHECKS:
        check(ctx)
    return ctx
