"""Reference numbering and timing — deterministic, never left to prompt authors.

MiniMaxH3ReferenceToVideo labels references in a fixed order: images, then
videos (a video's soundtrack takes an <Audio j> label right before its
<Video k>), then standalone audio. Socket names are 0-based
(ref_image_0 -> <Picture 1>).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from .spec import Spec, SpecError

FPS = 24
MAX_IMAGES, MAX_VIDEOS, MAX_AUDIOS, MAX_FILES = 9, 3, 3, 12
MAX_AUDIO_SECONDS = 15.0
TRAINED_FRAMES = (124, 362)  # ~5.2 s to ~15 s
WORDS_PER_SECOND = 2.5  # same speech budget simpleGen uses
TOKEN = re.compile(r"\[\[([A-Za-z0-9_-]+)\]\]")


@dataclass(frozen=True)
class Ref:
    kind: str  # picture | video | video_audio | voice | subject
    id: str
    tag: str  # <Picture 1>, <Audio 2>, <Subject 1> ...
    socket: str = ""  # MiniMaxH3ReferenceToVideo input name, e.g. ref_images.ref_image_0
    index: int = 0  # 1-based ordinal within its tag family


def assign(spec: Spec) -> dict[str, Ref]:
    """Map every id (and 'video_id#audio' for soundtracks) to its reference tag."""
    refs: dict[str, Ref] = {}
    for i, p in enumerate(spec.pictures, 1):
        refs[p.id] = Ref("picture", p.id, f"<Picture {i}>", f"ref_images.ref_image_{i - 1}", i)
    audio_n = 0
    for k, v in enumerate(spec.videos, 1):
        if v.use_soundtrack:
            audio_n += 1
            refs[f"{v.id}#audio"] = Ref(
                "video_audio", v.id, f"<Audio {audio_n}>", f"ref_video_audios.ref_video_audio_{k - 1}", audio_n
            )
        refs[v.id] = Ref("video", v.id, f"<Video {k}>", f"ref_videos.ref_video_{k - 1}", k)
    for n, a in enumerate(spec.voices):
        audio_n += 1
        refs[a.id] = Ref("voice", a.id, f"<Audio {audio_n}>", f"ref_audios.ref_audio_{n}", audio_n)
    for n, s in enumerate(spec.subjects, 1):
        refs[s.id] = Ref("subject", s.id, f"<Subject {n}>", "", n)
    return refs


def speaker_code(refs: dict[str, Ref], subject_id: str) -> str:
    return f"S{refs[subject_id].index}"


def resolve_tokens(text: str, refs: dict[str, Ref]) -> str:
    def swap(m: re.Match) -> str:
        key = m.group(1)
        if key not in refs:
            raise SpecError(f"[[{key}]] does not name a picture, video, voice or subject")
        return refs[key].tag

    return TOKEN.sub(swap, text)


def align_frames(n: int) -> int:
    """Snap up to H3's 17k+5 frame grid (same rule as the native node)."""
    n = max(5, n)
    while n % 17 != 5:
        n += 1
    return n


def spoken_words(spec: Spec) -> int:
    return sum(len(line.text.split()) for shot in spec.shots for line in shot.lines)


def estimate_seconds(spec: Spec) -> float:
    """Speech at 2.5 words/s plus ~1.5 s of lead-in/tail, clamped to the trained range."""
    speech = spoken_words(spec) / WORDS_PER_SECOND
    low, high = TRAINED_FRAMES[0] / FPS, TRAINED_FRAMES[1] / FPS
    return min(high, max(low, math.ceil((speech + 1.5) * 2) / 2))


def target_seconds(spec: Spec) -> float:
    return spec.duration if spec.duration else estimate_seconds(spec)


def frame_count(spec: Spec) -> int:
    return align_frames(round(target_seconds(spec) * FPS))
