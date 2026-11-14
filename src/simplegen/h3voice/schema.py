"""Target node schemas (/object_info): load, trim, complete inputs, validate.

Each profile names a schema snapshot in schemas/. Snapshots are trimmed to the
node classes this package uses, so they are small enough to keep in the folder
and let every check run offline. `h3voice schema <profile>` refreshes one from
a live server (local) or from simpleGen's cached RunPod schema.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

SCALARS = {"INT", "FLOAT", "STRING", "BOOLEAN", "COMBO"}
DYNAMIC = "COMFY_DYNAMICCOMBO_V3"
AUTOGROW = "COMFY_AUTOGROW_V3"
# Classes the builder can emit. Add a class here when graph.py starts using it.
USED_CLASSES = [
    "UNETLoader", "UNETLoaderMultiGPU", "CLIPLoader", "CLIPLoaderMultiGPU", "VAELoader", "VAELoaderMultiGPU",
    "VRAM_Debug",
    "LoraLoaderModelOnly", "LoadImage", "LoadAudio", "LoadVideo", "GetVideoComponents",
    "TrimAudioDuration", "MiniMaxH3ReferenceToVideo", "MiniMaxH3AddGuide", "MiniMaxH3SigmaShift",
    "RandomNoise", "KSamplerSelect", "BasicScheduler", "BasicGuider", "SamplerCustomAdvanced",
    "VAEDecode", "VAEDecodeAudio", "CreateVideo", "SaveVideo", "SaveAudio",
]


# Loader inputs whose options are the server's input folder. Snapshots drop these
# lists: they go stale immediately, and `run` uploads the files it needs.
INPUT_FILES = {"LoadImage": "image", "LoadAudio": "audio", "LoadVideo": "file"}


def trim(object_info: dict) -> dict:
    out = json.loads(json.dumps({k: object_info[k] for k in USED_CLASSES if k in object_info}))
    for cls, name in INPUT_FILES.items():
        spec = out.get(cls, {}).get("input", {}).get("required", {}).get(name)
        if spec and isinstance(spec[0], list):
            spec[0] = []
        elif spec and len(spec) > 1 and "options" in spec[1]:
            spec[1]["options"] = []
    return out


def load(path: Path) -> dict:
    return json.loads(Path(path).expanduser().read_text(encoding="utf-8"))


def _options(spec: list) -> list | None:
    kind = spec[0]
    if isinstance(kind, list):
        return kind
    if kind == "COMBO" and len(spec) > 1:
        return spec[1].get("options", [])
    return None


def _is_widget(spec: list) -> bool:
    kind, opts = spec[0], (spec[1] if len(spec) > 1 else {})
    return not opts.get("forceInput") and (isinstance(kind, list) or kind in SCALARS or kind == DYNAMIC)


def _default(spec: list) -> Any:
    opts = spec[1] if len(spec) > 1 else {}
    if "default" in opts:
        return opts["default"]
    if spec[0] == DYNAMIC:
        return opts["options"][0]["key"]
    choices = _options(spec)
    if choices is not None:
        return choices[0] if choices else None
    return {"INT": 0, "FLOAT": 0.0, "STRING": "", "BOOLEAN": False}.get(spec[0])


def walk(info: dict, values: dict):
    """Yield (name, spec, required) for every input, expanding the chosen dynamic-combo branch."""

    def expand(name: str, spec: list, required: bool):
        yield name, spec, required
        if spec[0] == DYNAMIC:
            chosen = values.get(name, _default(spec))
            branch = next((o for o in spec[1]["options"] if o["key"] == chosen), None)
            if branch:
                for section in ("required", "optional"):
                    for child, cspec in branch.get("inputs", {}).get(section, {}).items():
                        yield from expand(f"{name}.{child}", cspec, required and section == "required")

    for section in ("required", "optional"):
        for name, spec in info.get("input", {}).get(section, {}).items():
            yield from expand(name, spec, section == "required")


def complete(graph: dict, object_info: dict) -> dict:
    """Fill every omitted widget with the target's default, so API and UI exports are explicit."""
    for node in graph.values():
        info = object_info.get(node["class_type"])
        if not info:
            continue
        for name, spec, _required in walk(info, node["inputs"]):
            if name not in node["inputs"] and _is_widget(spec):
                node["inputs"][name] = _default(spec)
    return graph


def _is_link(value: Any) -> bool:
    return isinstance(value, list) and len(value) == 2 and isinstance(value[0], str) and isinstance(value[1], int)


def validate(graph: dict, object_info: dict, uploaded: set = frozenset()) -> list[str]:
    """Return human-readable problems; empty means the target can accept this graph.

    uploaded: (node_id, input) pairs holding files uploaded for this run, which
    the schema's file lists (fetched earlier) cannot contain yet.
    """
    problems = []
    for nid, node in graph.items():
        cls = node["class_type"]
        info = object_info.get(cls)
        if info is None:
            problems.append(f"{nid}: node class {cls} is not installed on this target")
            continue
        inputs = node["inputs"]
        specs = {name: (spec, req) for name, spec, req in walk(info, inputs)}
        for name, (spec, req) in specs.items():
            if spec[0] == AUTOGROW:
                continue
            if name not in inputs:
                if req:
                    problems.append(f"{nid}: required input {name} is missing")
                continue
            value = inputs[name]
            if _is_link(value):
                src = graph.get(value[0])
                if src is None:
                    problems.append(f"{nid}.{name}: links to missing node {value[0]}")
                    continue
                src_info = object_info.get(src["class_type"])
                if src_info:
                    outs = src_info.get("output", [])
                    if value[1] >= len(outs):
                        problems.append(f"{nid}.{name}: {value[0]} has no output {value[1]}")
                    elif not _types_match(outs[value[1]], spec[0]):
                        problems.append(f"{nid}.{name}: expects {spec[0]}, gets {outs[value[1]]} from {value[0]}")
                continue
            choices = _options(spec)
            if choices is not None and choices and value not in choices and (nid, name) not in uploaded:
                shown = ", ".join(map(str, choices[:12])) + (" ..." if len(choices) > 12 else "")
                problems.append(f"{nid}.{name}: {value!r} is not available here. Options: {shown}")
            opts = spec[1] if len(spec) > 1 else {}
            numeric = spec[0] in ("INT", "FLOAT") and isinstance(value, (int, float))
            if numeric and (("min" in opts and value < opts["min"]) or ("max" in opts and value > opts["max"])):
                problems.append(f"{nid}.{name}: {value} is outside {opts.get('min')}..{opts.get('max')}")
        groups = {n: s for n, (s, _) in specs.items() if s[0] == AUTOGROW}
        for name, value in inputs.items():
            if name in specs:
                continue
            group, _, slot = name.partition(".")
            template = groups.get(group, [None, {}])[1].get("template") if group in groups else None
            if not template:
                problems.append(f"{nid}: unknown input {name}")
                continue
            prefix, tmax = template["prefix"], template.get("max", 0)
            index = slot[len(prefix):] if slot.startswith(prefix) else ""
            if not index.isdigit() or int(index) >= tmax:
                problems.append(f"{nid}: {name} is not a valid {group} slot (max {tmax})")
                continue
            (_tname, tspec), = template["input"].get("required", template["input"].get("optional", {})).items()
            if _is_link(value) and value[0] in graph:
                src_info = object_info.get(graph[value[0]]["class_type"], {})
                outs = src_info.get("output", [])
                if value[1] < len(outs) and not _types_match(outs[value[1]], tspec[0]):
                    problems.append(f"{nid}.{name}: expects {tspec[0]}, gets {outs[value[1]]}")
    return problems


def _types_match(out_type: Any, in_type: Any) -> bool:
    if isinstance(out_type, list) or isinstance(in_type, list) or "*" in (out_type, in_type):
        return True
    return set(str(out_type).split(",")) & set(str(in_type).split(",")) != set()
