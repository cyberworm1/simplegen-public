"""Validate agent edits to a native workflow without choosing its shot topology.

Only a run's copy is accepted here. The source and native subgraph definitions
remain immutable. Code provides frame math and adds transport-only shot saves.
"""

from __future__ import annotations

import copy
import math
import re
from pathlib import Path

from .dimensions import resolve_dimensions
from .workflows import _duration_frames, _flatten, _specs, _widget, _widgets, validate_graph

_SEGMENT_FIELDS = {"prompt", "duration_override", "steps_override", "seed", "seed_mode"}
_PROJECT_FIELDS = {
    "width",
    "height",
    "size_mode",
    "megapixels",
    "default_seconds",
    "default_steps",
    "scheduler",
    "reference_size",
}
_LAYOUT_FIELDS = {"pos", "size", "order", "title"}


def _edges(container: dict) -> list[tuple]:
    result = []
    for link in container.get("links", []):
        if isinstance(link, list) and len(link) in (5, 6):
            values = link[:5]
        elif isinstance(link, dict) and set(link) <= {
            "id",
            "origin_id",
            "origin_slot",
            "target_id",
            "target_slot",
            "type",
        }:
            values = [
                link[key] for key in ("id", "origin_id", "origin_slot", "target_id", "target_slot")
            ]
        else:
            raise ValueError("Unsupported native link representation")
        if any(not isinstance(value, int) or isinstance(value, bool) for value in values):
            raise ValueError("Native node/link IDs and socket indices must be integers")
        result.append(tuple(values))
    return result


def _native_integrity(container: dict) -> None:
    nodes = container.get("nodes")
    if not isinstance(nodes, list) or not nodes:
        raise ValueError("Expected a native workflow with nodes")
    ids = [node.get("id") for node in nodes]
    if any(not isinstance(i, int) or isinstance(i, bool) or i < 0 for i in ids) or len(
        set(ids)
    ) != len(ids):
        raise ValueError("Native workflow has invalid or duplicate node IDs")
    by_id = {node["id"]: node for node in nodes}
    edges = _edges(container)
    if len({edge[0] for edge in edges}) != len(edges) or any(edge[0] < 0 for edge in edges):
        raise ValueError("Native workflow has invalid or duplicate link IDs")
    incoming, outgoing = {}, {}
    input_boundary = container.get("inputNode", {}).get("id", -10)
    output_boundary = container.get("outputNode", {}).get("id", -20)
    for edge_id, origin, source_slot, target, target_slot in edges:
        if source_slot < 0 or target_slot < 0:
            raise ValueError("Negative native socket index")
        if (target, target_slot) in incoming:
            raise ValueError("Multiple native links feed the same input")
        incoming[target, target_slot] = edge_id
        outgoing.setdefault((origin, source_slot), []).append(edge_id)
        if origin == input_boundary and "inputs" in container:
            if source_slot >= len(container["inputs"]):
                raise ValueError("Invalid subgraph input boundary")
        elif origin not in by_id:
            raise ValueError("Native link source node is missing")
        elif "outputs" in by_id[origin] and source_slot >= len(by_id[origin]["outputs"]):
            raise ValueError("Invalid native output socket")
        if target == output_boundary and "outputs" in container:
            if target_slot >= len(container["outputs"]):
                raise ValueError("Invalid subgraph output boundary")
        elif target not in by_id:
            raise ValueError("Native link target node is missing")
        elif target_slot >= len(by_id[target].get("inputs", [])):
            raise ValueError("Invalid native input socket")
    for node in nodes:
        for index, inp in enumerate(node.get("inputs", [])):
            if inp.get("link") != incoming.get((node["id"], index)):
                raise ValueError(f"Native node {node['id']} input has a nonreciprocal link")
        for index, output in enumerate(node.get("outputs", [])):
            links = output.get("links") or []
            if sorted(links) != sorted(outgoing.get((node["id"], index), [])):
                raise ValueError(f"Native node {node['id']} output has nonreciprocal links")
    for index, inp in enumerate(container.get("inputs", [])):
        if "linkIds" in inp and sorted(inp["linkIds"]) != sorted(
            outgoing.get((input_boundary, index), [])
        ):
            raise ValueError("Subgraph input linkIds are nonreciprocal")
    for index, output in enumerate(container.get("outputs", [])):
        if "linkIds" in output and output["linkIds"] != (
            [incoming[output_boundary, index]] if (output_boundary, index) in incoming else []
        ):
            raise ValueError("Subgraph output linkIds are nonreciprocal")


def _setup(
    template: dict, object_info: dict, megapixels: float, duration: float, run_id: str,
    aspect_ratio: str = "",
) -> dict:
    if not isinstance(template, dict) or "nodes" not in template:
        raise ValueError("Direct editing requires a native ComfyUI workflow JSON")
    if (
        not isinstance(megapixels, (int, float))
        or not math.isfinite(megapixels)
        or not 0.01 <= megapixels <= 256
    ):
        raise ValueError("Megapixels must be between 0.01 and 256")
    if not isinstance(duration, (int, float)) or not math.isfinite(duration):
        raise ValueError("Duration must be finite")
    if not isinstance(run_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,120}", run_id):
        raise ValueError("Invalid run ID")
    _native_integrity(template)
    definitions = template.get("definitions", {}).get("subgraphs", [])
    if len({d["id"] for d in definitions}) != len(definitions):
        raise ValueError("Duplicate native subgraph definitions")
    for definition in definitions:
        _native_integrity(definition)
    types = [
        d["id"]
        for d in definitions
        if sum(n["type"] == "H3RCResolveSegment" for n in d["nodes"]) == 1
        and sum(n["type"] == "H3RCState" for n in d["nodes"]) == 1
    ]
    if len(types) != 1:
        raise ValueError("Direct editing currently supports one H3RC native segment definition")
    nodes = {n["id"]: n for n in template["nodes"]}
    segments = [n for n in nodes.values() if n["type"] == types[0]]
    projects = [n for n in nodes.values() if n["type"] == "H3RCProject"]
    finals = [n for n in nodes.values() if n["type"] == "SaveVideo"]
    if len(finals) > 1:
        finals = [n for n in finals if "final" in n.get("title", "").lower()]
    if not segments or len(projects) != 1 or len(finals) != 1:
        raise ValueError(
            "Expected one H3RC project, a segment chain, and one identifiable final SaveVideo"
        )
    project = projects[0]
    values = _widgets(project, object_info["H3RCProject"])
    width, height = resolve_dimensions(megapixels, values["width"], values["height"], aspect_ratio)
    frames = _duration_frames(
        duration, float(values["default_seconds"]), int(values["context_frames"])
    )
    return {
        "nodes": nodes,
        "segments": segments,
        "segment_type": types[0],
        "project": project,
        "values": values,
        "final": finals[0],
        "width": width,
        "height": height,
        "frames": frames,
    }


def _mapping(node: dict, info: dict, segment_type: str) -> list[dict]:
    if node["type"] == segment_type:
        names = [inp["name"] for inp in node.get("inputs", []) if inp.get("widget")]
    else:
        names = [name for name, spec in _specs(info[node["type"]]).items() if _widget(spec)]
    return [
        {"input": name, "widget_index": i}
        for i, name in enumerate(names[: len(node.get("widgets_values", []))])
    ]


def edit_contract(
    template: dict, object_info: dict, *, megapixels: float, duration: float, run_id: str,
    check_model_availability: bool = True,
    aspect_ratio: str = "",
    steps: int | None = None,
) -> dict:
    """Small JSON-serializable editing instructions; contains no example prompts."""
    try:
        setup = _setup(template, object_info, megapixels, duration, run_id, aspect_ratio)
        if steps is not None:
            spec = _specs(object_info["H3RCProject"]).get("default_steps")
            if not spec or spec[0] != "INT":
                raise ValueError("This workflow does not expose default sampling steps.")
            bounds = spec[1] if len(spec) > 1 else {}
            if type(steps) is not int or not bounds.get("min", 1) <= steps <= bounds.get("max", 100):
                raise ValueError("Requested steps are outside this workflow's supported bounds.")
        baseline_graph, _ = _flatten(copy.deepcopy(template), object_info)
        validate_graph(baseline_graph, object_info, check_model_availability=check_model_availability)
        context = setup["values"]["context_frames"]
        refs = [n for n in template["nodes"] if n["type"] == "LoadImage"]
        sample_values = setup["segments"][0].get("widgets_values", [])
        section_names = sorted(
            {
                match.group(1)
                for value in sample_values
                if isinstance(value, str)
                for match in re.finditer(r"(?m)^([a-z][a-z_]{1,60}):", value)
            }
        )
        return {
            "native_workflow": True,
            "aspect_ratio": aspect_ratio or "template",
            "quality": {"default_steps": steps, "steps_override": 0} if steps is not None else {},
            "segment_type": setup["segment_type"],
            "width": setup["width"],
            "height": setup["height"],
            "legal_frame_counts": setup["frames"],
            "project_node_id": setup["project"]["id"],
            "final_node_id": setup["final"]["id"],
            "segment_node_ids": [n["id"] for n in setup["segments"]],
            "prompt_sections": section_names,
            "project_widgets": [
                m
                for m in _mapping(setup["project"], object_info, setup["segment_type"])
                if m["input"] in _PROJECT_FIELDS
            ],
            "segment_widgets": _mapping(setup["segments"][0], object_info, setup["segment_type"]),
            "reference_nodes": [
                {
                    "node_id": n["id"],
                    "title": n.get("title", "Reference image"),
                    "input": "image",
                    "widget_index": 0,
                }
                for n in refs
            ],
            "resolution": {
                "width": setup["width"],
                "height": setup["height"],
                "megapixels": megapixels,
                "size_mode": "width_height",
            },
            "timing": {
                "requested_total_seconds": duration,
                "fps": 24,
                "context_frames": context,
                "recommended_segments": [
                    {
                        "frames": f,
                        "duration_override": min(15.0, f / 24),
                        "usable_seconds": (f - (context if i else 0)) / 24,
                    }
                    for i, f in enumerate(setup["frames"])
                ],
                "actual_total_seconds": (
                    sum(setup["frames"]) - (len(setup["frames"]) - 1) * context
                )
                / 24,
                "rounding_tolerance_seconds": 17 / 24,
            },
            "allowed_segment_fields": sorted(_SEGMENT_FIELDS),
            "allowed_project_fields": sorted(_PROJECT_FIELDS),
            "instructions": [
                "Edit the native workflow.json directly. Do not replace it with an API graph.",
                "When quality is supplied, set H3RCProject.default_steps to that exact value and every segment steps_override to 0 so each segment inherits it. Do not choose a different step count.",
                "Write a new nonempty prompt on EVERY segment; do not retain source sample prompts.",
                "Clone/delete segment cards to implement the duration; retain their type, properties, socket definitions and subgraph definitions exactly.",
                "Use unique integer node/link IDs. Update links and reciprocal input.link/output.links arrays, including project output fanout.",
                "Every segment project input comes from the original project output 0. First previous is unconnected; subsequent previous inputs come from predecessor next_segment output 0. No branching.",
                "Final SaveVideo.video must connect to the last segment assembled_video output 1. Retain all other nonsegment nodes, model selections, and wiring.",
                "PreviewAny nodes connected to deleted segments may be deleted. No new executable classes or metadata/properties are permitted.",
                "Set each reference image widget to an existing absolute LOCAL file path under a linked folder. Do not copy assets or upload them yourself.",
                "Apply resolution and timing.recommended_segments exactly in order. Use each supplied duration_override and usable_seconds; do not recalculate frames, overlap or divide the requested duration. Code validates the total after context trimming.",
                "Do not change output prefixes: the transport scopes them and saves raw shots automatically without changing creative topology.",
            ],
        }
    except (
        KeyError,
        TypeError,
        IndexError,
        ZeroDivisionError,
        AttributeError,
        OverflowError,
    ) as error:
        raise ValueError(f"Unsupported or malformed native template: {error}") from None


def _node_structure(node: dict) -> dict:
    result = copy.deepcopy(node)
    # Named widget copies are informational; validate the actual widgets below.
    for key in _LAYOUT_FIELDS | {"id", "widgets_values", "widgets_values_named"}:
        result.pop(key, None)
    for inp in result.get("inputs", []):
        inp.pop("link", None)
    for output in result.get("outputs", []):
        output.pop("links", None)
    return result


def _input_edges(container: dict) -> dict:
    return {
        (target, slot): (origin, origin_slot)
        for _, origin, origin_slot, target, slot in _edges(container)
    }


def _validate(
    template: dict, edited: dict, info: dict, megapixels: float, duration: float, run_id: str,
    check_model_availability: bool = True,
    aspect_ratio: str = "",
    steps: int | None = None,
) -> dict:
    setup = _setup(template, info, megapixels, duration, run_id, aspect_ratio)
    if not isinstance(edited, dict):
        raise TypeError("Edited workflow must be an object")
    _native_integrity(edited)
    mutable = {"nodes", "links", "last_node_id", "last_link_id", "revision"}
    if {k: v for k, v in edited.items() if k not in mutable} != {
        k: v for k, v in template.items() if k not in mutable
    }:
        raise ValueError(
            "Native definitions, metadata, configuration, and groups must remain unchanged"
        )
    original, nodes = setup["nodes"], {n["id"]: n for n in edited["nodes"]}
    kind = setup["segment_type"]
    old_edges, new_edges = _input_edges(template), _input_edges(edited)
    segment_ids = {i for i, n in nodes.items() if n["type"] == kind}
    old_segment_ids = {n["id"] for n in setup["segments"]}
    if not segment_ids:
        raise ValueError("The edited workflow must contain at least one segment")
    for node_id, node in original.items():
        if node_id in nodes or node_id in old_segment_ids:
            continue
        source = old_edges.get((node_id, 0))
        if (
            node["type"] != "PreviewAny"
            or not source
            or source[0] not in old_segment_ids - segment_ids
        ):
            raise ValueError(f"Required original node {node_id} was removed")
    source_prompts = {
        value
        for n in setup["segments"]
        for key, value in zip(
            [i["name"] for i in n.get("inputs", []) if i.get("widget")], n.get("widgets_values", [])
        )
        if key == "prompt" and isinstance(value, str) and value.strip()
    }
    for node_id, node in nodes.items():
        if node_id in original:
            baseline = original[node_id]
        elif node["type"] == kind:
            baseline = setup["segments"][-1]
        else:
            raise ValueError("Only existing segment cards may be cloned")
        if _node_structure(node) != _node_structure(baseline):
            raise ValueError(
                f"Node {node_id}: class, socket definitions, properties or execution metadata changed"
            )
        values, before = node.get("widgets_values", []), baseline.get("widgets_values", [])
        if type(values) is not type(before) or len(values) != len(before):
            raise ValueError(
                f"Node {node_id}: widget representation and count must remain unchanged"
            )
        if isinstance(values, dict) and values.keys() != before.keys():
            raise ValueError(f"Node {node_id}: unapproved widget keys")
        if node["type"] == kind:
            names = [i["name"] for i in node.get("inputs", []) if i.get("widget")]
            values, before = node.get("widgets_values", []), baseline.get("widgets_values", [])
            if not isinstance(values, list) or len(values) != len(names):
                raise ValueError("Unsupported promoted segment widgets")
            for name, value, previous in zip(names, values, before):
                if name not in _SEGMENT_FIELDS and value != previous:
                    raise ValueError(f"Unapproved segment widget {name}")
            prompt = dict(zip(names, values)).get("prompt")
            if not isinstance(prompt, str) or not prompt.strip() or prompt in source_prompts:
                raise ValueError(
                    f"Segment {node_id}: write a new nonempty prompt; source sample prompts cannot remain"
                )
            if prompt.count("<d>") != prompt.count("</d>"):
                raise ValueError("Unbalanced dialogue tags in edited prompt")
        elif node["type"] in {"H3RCProject", "LoadImage"}:
            old_values, values = (
                _widgets(baseline, info[node["type"]]),
                _widgets(node, info[node["type"]]),
            )
            allowed = _PROJECT_FIELDS if node["type"] == "H3RCProject" else {"image"}
            if {k: v for k, v in values.items() if k not in allowed} != {
                k: v for k, v in old_values.items() if k not in allowed
            }:
                raise ValueError(f"Node {node_id}: unapproved setting change")
            if node["type"] == "LoadImage" and (
                not isinstance(values.get("image"), str) or not Path(values["image"]).is_absolute()
            ):
                raise ValueError("Reference image widgets must contain absolute local file paths")
        elif node.get("widgets_values", []) != baseline.get("widgets_values", []):
            raise ValueError(
                f"Node {node_id}: original model selections and fixed widgets must remain unchanged"
            )
    project_id, final_id = setup["project"]["id"], setup["final"]["id"]
    previous = {}
    for node_id, node in nodes.items():
        for slot, inp in enumerate(node.get("inputs", [])):
            edge = new_edges.get((node_id, slot))
            if node_id in segment_ids:
                name = inp["name"]
                if name == "project":
                    if edge != (project_id, 0):
                        raise ValueError("Every segment must use the original shared project")
                elif name == "previous":
                    if edge is not None and (edge[0] not in segment_ids or edge[1] != 0):
                        raise ValueError(
                            "Segment previous must use another segment next_segment output"
                        )
                    previous[node_id] = edge[0] if edge else None
                elif edge is not None:
                    raise ValueError("Promoted segment widgets may not be rewired")
            elif node_id == final_id and inp["name"] == "video":
                continue
            elif edge != old_edges.get((node_id, slot)):
                raise ValueError(f"Unapproved rewiring of fixed node {node_id}.{inp['name']}")
    ordered = []
    cursor = None
    while len(ordered) < len(segment_ids):
        matches = [
            node_id
            for node_id in segment_ids - set(ordered)
            if previous.get(node_id, object()) == cursor
        ]
        if len(matches) != 1:
            raise ValueError("Segments must form one linear chain without branches or cycles")
        cursor = matches[0]
        ordered.append(cursor)
    final_slot = next(
        i for i, inp in enumerate(nodes[final_id]["inputs"]) if inp["name"] == "video"
    )
    if new_edges.get((final_id, final_slot)) != (ordered[-1], 1):
        raise ValueError("Final assembly must use the last segment assembled_video output")
    graph, groups = _flatten(copy.deepcopy(edited), info)
    project = graph[str(project_id)]["inputs"]
    if steps is not None and project.get("default_steps") != steps:
        raise ValueError("Edited default_steps does not match the selected Steps setting.")
    width, height = project["width"], project["height"]
    if project.get("size_mode", "width_height") == "megapixels":
        mp = project.get("megapixels", 1.0)
        if not isinstance(mp, (int, float)) or not math.isfinite(mp) or mp <= 0:
            raise ValueError("Invalid edited megapixel setting")
        scale = math.sqrt(mp * 1_000_000 / (width * height))
        width, height = (max(32, round(value * scale / 32) * 32) for value in (width, height))
    if (width, height) != (setup["width"], setup["height"]):
        raise ValueError(
            "Edited resolution does not match the requested megapixels and aspect ratio"
        )
    default = float(project["default_seconds"])
    if not math.isfinite(default) or not 4 <= default <= 15:
        raise ValueError("Default H3 generation window must be 4–15 seconds")
    frames, prompt_slots = [], []
    final = str(final_id)
    for index, native_id in enumerate(ordered):
        group = groups[f"{native_id}:"]
        resolvers = [i for i in group["nodes"] if graph[i]["class_type"] == "H3RCResolveSegment"]
        states = [i for i in group["nodes"] if graph[i]["class_type"] == "H3RCState"]
        if len(resolvers) != 1 or len(states) != 1:
            raise ValueError("Unsupported H3 segment internals")
        if steps is not None and graph[resolvers[0]]["inputs"].get("steps_override", 0) != 0:
            raise ValueError("Set every segment steps_override to 0 to inherit the selected Steps setting.")
        seconds = float(graph[resolvers[0]]["inputs"].get("duration_override", 0) or default)
        if not math.isfinite(seconds) or not 4 <= seconds <= 15:
            raise ValueError("Each H3 generation window must be 4–15 seconds")
        count = round(seconds * 24)
        count += (5 - count % 17) % 17
        frames.append(count)
        for node_id in sorted(group["nodes"]):
            node = graph[node_id]
            if node["class_type"] == "PrimitiveStringMultiline":
                prompt_slots.append(
                    {
                        "node_id": node_id,
                        "input": "value",
                        "text": node["inputs"]["value"],
                        "duration_seconds": (count - (project["context_frames"] if index else 0))
                        / 24,
                    }
                )
        shot = group["outputs"][2]
        state_inputs = graph[states[0]]["inputs"]
        if "raw_frames" in state_inputs:
            if graph[shot[0]]["class_type"] != "CreateVideo":
                raise ValueError("Raw-shot saving requires the native CreateVideo output")
            raw_id = f"simplegen_raw_{native_id}"
            if raw_id in graph:
                raise ValueError("Reserved transport node ID collision")
            graph[raw_id] = copy.deepcopy(graph[shot[0]])
            graph[raw_id]["inputs"]["images"] = copy.deepcopy(state_inputs["raw_frames"])
            if "raw_audio" in state_inputs:
                graph[raw_id]["inputs"]["audio"] = copy.deepcopy(state_inputs["raw_audio"])
            else:
                graph[raw_id]["inputs"].pop("audio", None)
            shot = [raw_id, 0]
        save_id = f"simplegen_shot_{native_id}"
        graph[save_id] = copy.deepcopy(graph[final])
        graph[save_id]["inputs"].update(
            video=shot, filename_prefix=f"simpleGen/{run_id}/shots/shot_{index + 1:04d}"
        )
    actual = (sum(frames) - (len(frames) - 1) * project["context_frames"]) / 24
    if not duration <= actual < duration + 17 / 24 + 1e-9:
        raise ValueError(
            f"Edited chain produces {actual:.3f}s; requested {duration:g}s within one H3 frame-lattice step"
        )
    graph[final]["inputs"]["filename_prefix"] = f"simpleGen/{run_id}/final"
    project["checkpoint_prefix"] = f"simpleGen/{run_id}/latents/segment"
    for node_id, node in graph.items():
        if (
            node["class_type"] == "SaveVideo"
            and node_id != final
            and not node_id.startswith("simplegen_shot_")
        ):
            node["inputs"]["filename_prefix"] = (
                f"simpleGen/{run_id}/shots/source_{node_id.replace(':', '_')}"
            )
    references = [
        {
            "node_id": node_id,
            "input": "image",
            "value": node["inputs"]["image"],
            "title": node.get("_meta", {}).get("title", "Reference image"),
        }
        for node_id, node in graph.items()
        if node["class_type"] == "LoadImage"
    ]
    validate_graph(graph, info, check_model_availability=check_model_availability)
    return {
        "graph": graph,
        "final_node": final,
        "prompt_slots": prompt_slots,
        "reference_slots": references,
        "summary": f"Agent-edited H3 chain: {len(ordered)} segments; requested {duration:g}s, actual {actual:.3f}s at 24 fps; {width} × {height} ({width * height / 1_000_000:.3f} MP). Final and raw shots saved.",
    }


def validate_edited_workflow(
    template: dict,
    edited: dict,
    object_info: dict,
    *,
    megapixels: float,
    duration: float,
    run_id: str,
    check_model_availability: bool = True,
    aspect_ratio: str = "",
    steps: int | None = None,
) -> dict:
    """Flatten an approved native edit; never clone/remove creative segments."""
    try:
        return _validate(template, edited, object_info, megapixels, duration, run_id,
                         check_model_availability=check_model_availability, aspect_ratio=aspect_ratio,
                         steps=steps)
    except (
        KeyError,
        TypeError,
        IndexError,
        ZeroDivisionError,
        StopIteration,
        OverflowError,
        AttributeError,
    ) as error:
        raise ValueError(f"Malformed or unsupported edited workflow: {error}") from None
