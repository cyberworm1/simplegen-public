"""Aspect choices and deterministic H3 output dimensions."""

import math

ASPECT_RATIOS = (
    ("1:1", "Square"),
    ("2:3", "Portrait Photo"),
    ("3:2", "Photo"),
    ("3:4", "Portrait Standard"),
    ("4:3", "Standard"),
    ("9:16", "Portrait Widescreen"),
    ("16:9", "Widescreen"),
    ("21:9", "Ultrawide"),
)


def resolve_dimensions(megapixels: float, width: float, height: float,
                       aspect_ratio: str = "") -> tuple[int, int]:
    """Round target area/ratio to H3's 32-pixel grid; empty ratio preserves legacy sizing."""
    if aspect_ratio:
        if aspect_ratio not in dict(ASPECT_RATIOS):
            raise ValueError("Choose a supported aspect ratio.")
        width, height = (int(part) for part in aspect_ratio.split(":"))
    if not all(math.isfinite(v) and v > 0 for v in (megapixels, width, height)):
        raise ValueError("Megapixels and dimensions must be positive finite numbers.")
    scale = math.sqrt(megapixels * 1_000_000 / (width * height))
    return tuple(max(32, round(value * scale / 32) * 32) for value in (width, height))
