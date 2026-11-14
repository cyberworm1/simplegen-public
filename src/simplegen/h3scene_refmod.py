"""RefMod variant of the H3 Scene Chain: the approved graph plus RefMod loader/apply nodes.

The vendored builder stays untouched. Its graph is extended in code: one shared
``MiniMaxH3RefModsLoader`` and, in every segment, one ``MiniMaxH3RefModApply``
between the Ref2VA conditioning and its consumer (the Extender or the guider).
RefMods come from the asset cards' ``## RefMod`` sections; the agent only picks them.
"""
from __future__ import annotations

import re
from pathlib import Path

from .h3scene import build as vendored
from .h3scene import produce

SKILL = "h3-scene-refmod-production"
SLOTS = 8
# 0 disables the node pack's combined RefMod token budget. Tested runs used limits of
# 16,384 and 24,576 for packages of 13,402 and 19,226 tokens.
MAX_TOKENS = 0
WORK_PREFIX = "refmodBuilder_work/"
LOADER, APPLY_NODE = "MiniMaxH3RefModsLoader", "MiniMaxH3RefModApply"
# Apply and per-slot values from tested RefMod workflows.
APPLY = {"override": False, "retention": 1.0, "curve_direction": "constant", "scramble_seed": -1,
         "curve_shape": "linear", "curve_value": 1.0, "graph_preset": "(none)", "scramble_mode": "shuffle",
         "scramble_keep": 1, "max_total_tokens": MAX_TOKENS, "save_preset_as": ""}
SLOT = {"strength": 1.0, "copies": 1, "components": "Visual", "visual_strength": 1.0, "audio_strength": 1.0}
PROFILE = {
    "node_pack": "ComfyUI-MiniMaxH3Mod", "node_pack_revision": "f9462081e28794389b5a6c5067eb327412ad8ee7",
    "slots": SLOTS, "slot": SLOT, "apply": APPLY, "show_info": True,
    "placement": "every segment: Ref2VA conditioning -> Apply -> Extender or guider",
    "card_section": "## RefMod", "replaces_card_pictures": True, "tested": "single 5 s shots only",
}

SECTION = re.compile(r"^## RefMods?[ \t]*$", re.MULTILINE)
PART = r"[A-Za-z0-9_][A-Za-z0-9_. -]*"
NAME = re.compile(rf"{PART}(?:/{PART})*\Z")


def card_names(text):
    """Loader names listed under a card's ``## RefMod`` heading (one per line)."""
    found = SECTION.search(text)
    if not found:
        return []
    rest = text[found.end():]
    following = re.search(r"^#{1,6} ", rest, re.MULTILINE)
    names = []
    for line in (rest[:following.start()] if following else rest).splitlines():
        line = line.strip().lstrip("-*").strip().strip("`").strip()
        line = line.removesuffix(".safetensors")
        if line:
            names.append(line)
    return names


def library(root):
    """Every card RefMod in the asset folder: loader name -> card folder (relative)."""
    root = Path(root).resolve()
    found = {}
    for category in ("characters", "places", "props"):
        for card in sorted((root / category).rglob("card.md")):
            if not card.resolve().is_relative_to(root):
                continue
            for name in card_names(card.read_text(encoding="utf-8")):
                found.setdefault(name, card.parent.relative_to(root).as_posix())
    return found


def name_problem(name):
    if not NAME.fullmatch(name) or ".." in name.split("/"):
        return f"RefMod {name!r} is not a RefMod loader name"
    if name.startswith(WORK_PREFIX):
        return f"RefMod {name!r} is a refmodBuilder work file; name the exported package instead"
    return None


def available(object_info):
    """Loader choices ComfyUI reports, or None when the schema does not say."""
    spec = (object_info or {}).get(LOADER, {}).get("input", {}).get("required", {}).get("mod_1")
    return set(spec[0]) if spec and isinstance(spec[0], list) else None


def availability(root, object_info):
    """(problems, names, verified) for the cards' RefMods against ComfyUI's loader list."""
    names = library(root)
    problems = [f"{card}: {problem}" for name, card in names.items() if (problem := name_problem(name))]
    choices = available(object_info)
    if choices is not None:
        problems += [f"{card}: RefMod {name!r} is not installed in ComfyUI" for name, card in names.items()
                     if not name_problem(name) and name not in choices]
    return problems, names, choices is not None


def check(plan, root, errors):
    """Mechanical RefMod checks; the agent's choices are never altered."""
    root = Path(root).resolve()
    refmods = plan.get("refmods")
    if not isinstance(refmods, list) or any(
            not isinstance(r, dict) or not isinstance(r.get("name"), str) or not r["name"].strip()
            or not isinstance(r.get("subject"), str) or not r["subject"].strip() for r in refmods):
        errors.append("refmods must be a list of objects with name and subject text")
        return
    pictures = plan.get("pictures") if isinstance(plan.get("pictures"), list) else []
    names = [r["name"].strip() for r in refmods]
    if not names and not pictures:
        errors.append("need at least one picture or RefMod")
    if len(names) > SLOTS:
        errors.append(f"use at most {SLOTS} RefMods")
    if len(set(names)) != len(names):
        errors.append("each RefMod may be listed once")
    known = library(root)
    segments = plan.get("segments") if isinstance(plan.get("segments"), list) else []
    for name in dict.fromkeys(names):
        problem = name_problem(name)
        if problem:
            errors.append(problem)
            continue
        if name not in known:
            errors.append(f"RefMod {name!r} is not named in any card's ## RefMod section")
            continue
        folder = (root / known[name]).resolve()
        for picture in pictures:
            path = picture.get("path") if isinstance(picture, dict) else None
            if isinstance(path, str) and path and (root / path).resolve().is_relative_to(folder):
                errors.append(f"picture {path} belongs to {known[name]}, which uses RefMod {name!r}; "
                              f"drop the picture (the RefMod replaces it)")
        block = produce.verbatim_block((folder / "card.md").read_text(encoding="utf-8"))
        for i, segment in enumerate(segments, 1):
            sections = segment.get("sections") if isinstance(segment, dict) else None
            text = sections.get("subject_definitions") if isinstance(sections, dict) else None
            if block and isinstance(text, str) and block not in " ".join(text.split()):
                errors.append(f"seg {i}: subject_definitions for RefMod {name!r} ({known[name]}) must contain "
                              f"its card's Verbatim constraints word for word: {block!r}")


def loader_widgets(names):
    slots = list(names) + ["(none)"] * (SLOTS - len(names))
    widgets = [("show_info", PROFILE["show_info"])]
    for i, name in enumerate(slots, 1):
        widgets += [(f"mod_{i}", name), (f"strength_{i}", SLOT["strength"]), (f"copies_{i}", SLOT["copies"])]
    widgets.append(("max_total_tokens", MAX_TOKENS))
    for i in range(1, SLOTS + 1):
        widgets += [(f"components_{i}", SLOT["components"]), (f"visual_strength_{i}", SLOT["visual_strength"]),
                    (f"audio_strength_{i}", SLOT["audio_strength"])]
    return widgets


def add_nodes(graph, names):
    """Insert the loader and a per-segment Apply after each Ref2VA conditioning."""
    if len(names) > SLOTS:
        raise ValueError(f"use at most {SLOTS} RefMods")
    loader = graph.add(LOADER, f"RefMods ({len(names)}) - Visual only", loader_widgets(names),
                       [("mods", "H3_REF_MODS"), ("info", "STRING")], (0, -420), (330, 380))
    for r2v in [n for n in graph.nodes if n["type"] == "MiniMaxH3ReferenceToVideo"]:
        segment = r2v["title"].split(" ", 1)[0]
        x, y = r2v["pos"]
        apply = graph.add(APPLY_NODE, f"{segment} Apply RefMods", list(APPLY.items()),
                          [("conditioning", "CONDITIONING"), ("curve graph", "IMAGE")], (x, y + 720), (330, 300))
        # Everything that read the Ref2VA conditioning now reads it with the RefMods applied.
        for link in graph.links:
            if link[1] == r2v["id"] and link[2] == 0:
                link[1] = apply["id"]
        graph.link(r2v, 0, apply, "conditioning", "CONDITIONING")
        graph.link(loader, 0, apply, "mods", "H3_REF_MODS")
    return graph


def drop_pictures(graph):
    """Remove the builder's placeholder picture when a plan uses RefMods only."""
    gone = {n["id"] for n in graph.nodes if n["type"] == "LoadImage"}
    dead = {link[0] for link in graph.links if link[1] in gone}
    graph.nodes = [n for n in graph.nodes if n["id"] not in gone]
    graph.links = [link for link in graph.links if link[0] not in dead]
    by_id = {link[0]: link for link in graph.links}
    for node in graph.nodes:
        node["ins"] = [i for i in node["ins"] if i["link"] not in dead]
        for index, item in enumerate(node["ins"]):
            by_id[item["link"]][4] = index


def build(spec, warn=print):
    """The approved builder's graph with RefMods; returns the vendored Graph object."""
    pictures = list(spec["pictures"])
    placeholder = not pictures
    names = pictures or ["placeholder.png"]
    graph, _ = vendored.build({**spec, "pictures": names}, names, warn=warn)
    add_nodes(graph, list(spec.get("refmods", [])))
    if placeholder:
        drop_pictures(graph)
    return graph


def verify(graph):
    """Exactly one loader with fixed slot settings and one fixed Apply per Ref2VA."""
    loaders = [k for k, n in graph.items() if n["class_type"] == LOADER]
    if len(loaders) != 1:
        raise ValueError("H3 Scene Chain RefMod workflow must have exactly one RefMod loader")
    inputs = graph[loaders[0]]["inputs"]
    expected = dict(loader_widgets([]))
    for key, value in expected.items():
        if not key.startswith("mod_") and inputs.get(key) != value:
            raise ValueError(f"H3 Scene Chain RefMod loader changed approved setting {key}")
    names = [inputs.get(f"mod_{i}") for i in range(1, SLOTS + 1)]
    if set(inputs) != set(expected) or any(not isinstance(n, str) for n in names):
        raise ValueError("H3 Scene Chain RefMod loader inputs differ from the approved layout")
    for name in names:
        if name != "(none)" and name_problem(name):
            raise ValueError(name_problem(name))
    r2vs = {k for k, n in graph.items() if n["class_type"] == "MiniMaxH3ReferenceToVideo"}
    applies = {k: n for k, n in graph.items() if n["class_type"] == APPLY_NODE}
    if sorted(n["inputs"]["conditioning"][0] for n in applies.values()) != sorted(r2vs):
        raise ValueError("Every H3 Scene Chain segment needs exactly one RefMod Apply on its conditioning")
    for key, node in applies.items():
        fixed = {k: v for k, v in node["inputs"].items() if k not in ("conditioning", "mods")}
        if fixed != APPLY or node["inputs"]["mods"] != [loaders[0], 0]:
            raise ValueError(f"H3 Scene Chain RefMod Apply {key} changed approved settings")
    for key, node in graph.items():
        for name, value in node["inputs"].items():
            if isinstance(value, list) and value[:1] and value[0] in r2vs and value[1] == 0 and key not in applies:
                raise ValueError(f"H3 Scene Chain {key}.{name} bypasses the RefMod Apply")
