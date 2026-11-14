"""Stable video manifest identities, without copying media or trusting filenames alone."""

from __future__ import annotations

import hashlib
import json
from pathlib import PurePosixPath

VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}


def digest(value) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def video_outputs(prompt_id: str, history: dict) -> list[dict]:
    files = {}
    for outputs in history.get("outputs", {}).values():
        for values in outputs.values():
            if not isinstance(values, list):
                continue
            for value in values:
                if not isinstance(value, dict) or not isinstance(
                    value.get("filename"), str
                ):
                    continue
                name = value["filename"]
                if PurePosixPath(name).suffix.lower() not in VIDEO_SUFFIXES:
                    continue
                subfolder, kind = (
                    value.get("subfolder", ""),
                    value.get("type", "output"),
                )
                if (
                    not isinstance(subfolder, str)
                    or PurePosixPath(name).name != name
                    or "\\" in name
                    or "\\" in subfolder
                    or PurePosixPath(subfolder).is_absolute()
                    or ".." in PurePosixPath(subfolder).parts
                    or kind != "output"
                ):
                    raise ValueError(
                        "All required videos must have safe persistent output paths"
                    )
                item = {
                    "prompt_id": prompt_id,
                    "filename": name,
                    "subfolder": subfolder,
                    "type": kind,
                }
                key = digest(item)
                files[key] = {"id": key, **item}
    return [files[key] for key in sorted(files)]


def manifest(jobs: dict) -> list[dict]:
    files = {}
    for job in jobs.values():
        for item in job.get("outputs", []):
            files[item["id"]] = item
    return [files[key] for key in sorted(files)]
