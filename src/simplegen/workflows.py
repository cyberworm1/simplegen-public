"""Deterministic ComfyUI graph preparation; never executes a workflow.

Native subgraphs and promoted inputs are expanded from the linked template.
H3 Reference Chain timing is solved on its 24 fps / 17-frame lattice.
"""

from __future__ import annotations

import copy
import math
import re
from typing import Any

from .dimensions import resolve_dimensions

_MISSING = object()
_NOTES = {"MarkdownNote", "Note", "PreviewAny"}
_SCALARS = {"INT", "FLOAT", "STRING", "BOOLEAN", "COMBO", "COMFY_DYNAMICCOMBO_V3"}


def _link(value: Any) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 2
        and isinstance(value[0], str)
        and isinstance(value[1], int)
    )


def _specs(info: dict) -> dict:
    return {
        **info.get("input", {}).get("required", {}),
        **info.get("input", {}).get("optional", {}),
    }


def _widget(spec: list) -> bool:
    kind, options = spec[0], spec[1] if len(spec) > 1 else {}
    return not options.get("forceInput") and (isinstance(kind, list) or kind in _SCALARS)


def _widgets(node: dict, info: dict) -> dict:
    """Decode the frontend's ordered widgets, including dynamic nested choices."""
    values = node.get("widgets_values", []) or []
    if isinstance(values, dict):
        return {k: v for k, v in values.items() if k in _specs(info)}
    result: dict = {}
    index = 0

    def consume(name: str, spec: list) -> None:
        nonlocal index
        if not _widget(spec):
            return
        options = spec[1] if len(spec) > 1 else {}
        if index >= len(values):
            if "default" in options:
                result[name] = options["default"]
            return
        value = values[index]
        index += 1
        result[name] = value
        if spec[0] == "COMFY_DYNAMICCOMBO_V3":
            selected = next((x for x in options.get("options", []) if x["key"] == value), None)
            if selected is None:
                raise ValueError(f"Node {node['id']}: unsupported {name} choice {value!r}")
            for section in ("required", "optional"):
                for child, child_spec in selected.get("inputs", {}).get(section, {}).items():
                    consume(f"{name}.{child}", child_spec)
        # Frontend-only seed controls and image upload button state are not API inputs.
        if (
            options.get("control_after_generate")
            and index < len(values)
            and values[index] in ("fixed", "randomize", "increment", "decrement")
        ):
            index += 1
        if (
            options.get("image_upload")
            and index < len(values)
            and values[index] in ("image", "choose file to upload")
        ):
            index += 1

    for name, spec in _specs(info).items():
        consume(name, spec)
    if index != len(values):
        raise ValueError(
            f"Node {node['id']} ({node['type']}): cannot map {len(values) - index} frontend widgets; export an API workflow or update the node package"
        )
    return result


def _links(container: dict) -> dict:
    result = {}
    for item in container.get("links", []):
        if isinstance(item, list):
            key, origin, slot, target, target_slot, *_ = item
        else:
            key, origin, slot, target, target_slot = (
                item[k] for k in ("id", "origin_id", "origin_slot", "target_id", "target_slot")
            )
        result[key] = (str(origin), slot, str(target), target_slot)
    return result


def _flatten(template: dict, object_info: dict) -> tuple[dict, dict]:
    definitions = {d["id"]: d for d in template.get("definitions", {}).get("subgraphs", [])}
    graph: dict = {}
    groups: dict = {}

    def expand(container: dict, prefix: str, external: dict, ancestry: tuple = ()) -> list:
        nodes = {str(n["id"]): n for n in container.get("nodes", [])}
        links = _links(container)
        expanded: dict = {}
        visiting: set = set()
        broadcasts: dict = {}
        for n in nodes.values():
            if "Everywhere" not in n["type"]:
                continue
            if n["type"] != "Anything Everywhere" or n.get("mode", 0) != 0:
                raise ValueError(
                    "Only plain active Anything Everywhere nodes are supported; export an API graph for regex, bypass, or scoped broadcasts"
                )
            props = n.get("properties", {})
            ue_props = props.get("ue_properties", {})
            if any(
                props.get(k) or ue_props.get(k)
                for k in ("group_restricted", "color_restricted", "title_regex", "input_regex")
            ):
                raise ValueError("Scoped Anything Everywhere broadcasts require an API export")
            for inp in n.get("inputs", []):
                if inp.get("link") is not None:
                    edge = links[inp["link"]]
                    kind = inp.get("type")
                    if kind in broadcasts:
                        raise ValueError(f"Ambiguous Anything Everywhere broadcast for {kind}")
                    broadcasts[kind] = edge[:2]

        def source(origin: str, slot: int) -> Any:
            if origin == str(container.get("inputNode", {}).get("id", -10)):
                return external.get(slot, _MISSING)
            if origin not in nodes:
                raise ValueError(f"Broken workflow link: node {prefix}{origin} is missing")
            n = nodes[origin]
            if n["type"] in definitions:
                build(origin)
                outputs = expanded[origin]
                if slot >= len(outputs) or outputs[slot] is _MISSING:
                    raise ValueError(f"Subgraph {prefix}{origin} output {slot} is disconnected")
                return outputs[slot]
            if n["type"] == "Reroute":
                inputs = n.get("inputs", [])
                if len(inputs) != 1 or inputs[0].get("link") not in links:
                    raise ValueError("Disconnected reroute")
                return source(*links[inputs[0]["link"]][:2])
            if n["type"] == "PrimitiveNode":
                return n.get("widgets_values", [_MISSING])[0]
            if n["type"] in _NOTES or n.get("mode", 0) != 0:
                raise ValueError(
                    f"Executable link uses disabled or unsupported node {prefix}{origin}"
                )
            return [prefix + origin, slot]

        def inputs_for(n: dict, values: dict) -> dict:
            for inp in n.get("inputs", []):
                edge_id = inp.get("link")
                if edge_id is not None:
                    if edge_id not in links:
                        raise ValueError(f"Node {n['id']} references missing link {edge_id}")
                    value = source(*links[edge_id][:2])
                    if value is _MISSING:
                        values.pop(inp["name"], None)
                    else:
                        values[inp["name"]] = value
                elif (
                    not inp.get("widget")
                    and inp.get("type") in broadcasts
                    and not n.get("properties", {})
                    .get("ue_properties", {})
                    .get("input_ue_unconnectable", {})
                    .get(inp["name"])
                ):
                    values[inp["name"]] = source(*broadcasts[inp["type"]])
            return values

        def build(node_id: str) -> None:
            if node_id in expanded:
                return
            if node_id in visiting:
                raise ValueError("Cyclic native subgraph connections")
            visiting.add(node_id)
            n = nodes[node_id]
            kind = n["type"]
            if kind in definitions:
                if kind in ancestry:
                    raise ValueError("Recursive native subgraph definitions")
                definition = definitions[kind]
                values = n.get("widgets_values", []) or []
                widget_names = [i["name"] for i in n.get("inputs", []) if i.get("widget")]
                if len(values) != len(widget_names):
                    raise ValueError(f"Subgraph {node_id}: unsupported promoted widget layout")
                bound = inputs_for(n, dict(zip(widget_names, values)))
                boundary = {
                    i: bound[inp["name"]]
                    for i, inp in enumerate(definition.get("inputs", []))
                    if inp["name"] in bound
                }
                child_prefix = f"{prefix}{node_id}:"
                before = set(graph)
                outputs = expand(definition, child_prefix, boundary, ancestry + (kind,))
                groups[child_prefix] = {
                    "nodes": set(graph) - before,
                    "outputs": outputs,
                    "definition": kind,
                }
                expanded[node_id] = outputs
            elif kind in _NOTES or kind in {"Reroute", "PrimitiveNode"} or "Everywhere" in kind:
                expanded[node_id] = []
            else:
                if n.get("mode", 0) != 0:
                    raise ValueError(
                        f"Node {node_id} is muted/bypassed; export an active API graph"
                    )
                if kind not in object_info:
                    raise ValueError(
                        f"ComfyUI is missing node class {kind}; install the template's required node package"
                    )
                graph[prefix + node_id] = {
                    "class_type": kind,
                    "inputs": inputs_for(n, _widgets(n, object_info[kind])),
                    "_meta": {"title": n.get("title", kind)},
                }
                expanded[node_id] = []
            visiting.remove(node_id)

        for node_id in nodes:
            build(node_id)
        outputs = [_MISSING] * len(container.get("outputs", []))
        output_id = str(container.get("outputNode", {}).get("id", -20))
        for origin, slot, target, target_slot in links.values():
            if target == output_id:
                outputs[target_slot] = source(origin, slot)
        return outputs

    expand(template, "", {})
    return graph, groups


def _duration_frames(duration: float, default_seconds: float, context: int) -> list[int]:
    if not 4 <= duration <= 86400:
        raise ValueError("H3 total duration must be between 4 seconds and 24 hours")
    if context not in (5, 22, 39):
        raise ValueError("Unsupported H3 context length; expected 5, 22, or 39 frames")
    target = math.ceil(duration * 24)
    base = round(default_seconds * 24)
    base += (5 - base % 17) % 17
    count = max(1, math.ceil((target - context) / max(1, base - context)))
    while count > 1 and count * 107 - (count - 1) * context > target:
        count -= 1
    while count * 362 - (count - 1) * context < target:
        count += 1
    steps = max(count * 6, math.ceil((target + (count - 1) * context - count * 5) / 17))
    quotient, remainder = divmod(steps, count)
    return [5 + 17 * (quotient + (i < remainder)) for i in range(count)]


def _prepare_chain(
    graph: dict, groups: dict, project_id: str, duration: float, run_id: str, final: str
) -> tuple[dict, float]:
    project = graph[project_id]["inputs"]
    frames = _duration_frames(
        duration, float(project["default_seconds"]), int(project["context_frames"])
    )
    segments = []
    for prefix, group in groups.items():
        resolvers = [i for i in group["nodes"] if graph[i]["class_type"] == "H3RCResolveSegment"]
        states = [i for i in group["nodes"] if graph[i]["class_type"] == "H3RCState"]
        previous = [i for i in group["nodes"] if graph[i]["class_type"] == "H3RCPreviousState"]
        if len(resolvers) == len(states) == len(previous) == 1:
            if graph[resolvers[0]]["inputs"].get("project") != [project_id, 0]:
                raise ValueError("Multiple project settings in one H3 chain are unsupported")
            segments.append(
                {
                    **group,
                    "prefix": prefix,
                    "resolve": resolvers[0],
                    "state": states[0],
                    "previous": previous[0],
                }
            )
    if not segments:
        raise ValueError(
            "H3 duration expansion requires the native clonable Reference Chain segment template"
        )
    ordered = []
    last = None
    while segments:
        found = [s for s in segments if graph[s["previous"]]["inputs"].get("previous") == last]
        if len(found) != 1:
            raise ValueError("H3 segments must form one unambiguous linear continuation chain")
        segment = found[0]
        ordered.append(segment)
        segments.remove(segment)
        last = [segment["state"], 0]
    if graph[final]["inputs"].get("video") != ordered[-1]["outputs"][1]:
        raise ValueError("Final SaveVideo must use the final H3 segment's assembled video")
    originals = copy.deepcopy(graph)
    for segment in ordered:
        for node_id in segment["nodes"]:
            del graph[node_id]
    previous_state = None
    prompt_durations = {}
    for index, frame_count in enumerate(frames):
        source = ordered[min(index, len(ordered) - 1)]
        mapping = {
            node_id: f"sg{index + 1}:{node_id.removeprefix(source['prefix'])}"
            for node_id in source["nodes"]
        }

        def remap(value: Any, mapping=mapping) -> Any:
            return (
                [mapping.get(value[0], value[0]), value[1]]
                if _link(value)
                else copy.deepcopy(value)
            )

        for old, new in mapping.items():
            graph[new] = copy.deepcopy(originals[old])
            graph[new]["inputs"] = {k: remap(v) for k, v in graph[new]["inputs"].items()}
            if graph[new]["class_type"] == "H3RCNoise":
                graph[new]["inputs"]["seed"] = int(graph[new]["inputs"]["seed"]) + index
        resolver = graph[mapping[source["resolve"]]]["inputs"]
        resolver["duration_override"] = min(15.0, frame_count / 24)
        previous = graph[mapping[source["previous"]]]["inputs"]
        if previous_state is None:
            previous.pop("previous", None)
        else:
            previous["previous"] = previous_state
        previous_state = [mapping[source["state"]], 0]
        assembled = remap(source["outputs"][1])
        shot = remap(source["outputs"][2])
        state_inputs = graph[mapping[source["state"]]]["inputs"]
        if "raw_frames" in state_inputs:
            # Preserve the actual generated shot, including its motion-context
            # frames. The template still owns trimming and final assembly.
            if not _link(shot) or graph[shot[0]]["class_type"] != "CreateVideo":
                raise ValueError(
                    "H3 raw-shot saving requires the segment's native CreateVideo output"
                )
            raw_id = f"sg_raw_{index + 1}"
            graph[raw_id] = copy.deepcopy(graph[shot[0]])
            graph[raw_id]["inputs"]["images"] = copy.deepcopy(state_inputs["raw_frames"])
            if "raw_audio" in state_inputs:
                graph[raw_id]["inputs"]["audio"] = copy.deepcopy(state_inputs["raw_audio"])
            else:
                graph[raw_id]["inputs"].pop("audio", None)
            shot = [raw_id, 0]
        shot_id = f"sg_shot_{index + 1}"
        graph[shot_id] = copy.deepcopy(originals[final])
        graph[shot_id]["inputs"].update(
            video=shot, filename_prefix=f"simpleGen/{run_id}/shots/shot_{index + 1:04d}"
        )
        for node_id in mapping.values():
            if graph[node_id]["class_type"] == "PrimitiveStringMultiline":
                prompt_durations[node_id] = (
                    frame_count - (project["context_frames"] if index else 0)
                ) / 24
    graph[final]["inputs"]["video"] = assembled
    project["checkpoint_prefix"] = f"simpleGen/{run_id}/latents/segment"
    actual = (sum(frames) - project["context_frames"] * (len(frames) - 1)) / 24
    return prompt_durations, actual


def prepare_workflow(
    template: dict, object_info: dict, *, megapixels: float, duration: float, run_id: str,
    aspect_ratio: str = "",
) -> dict:
    """Prepare a copied graph and compact agent-editable slots; no I/O or rendering."""
    if not math.isfinite(megapixels) or not 0.01 <= megapixels <= 256:
        raise ValueError("Megapixels must be between 0.01 and 256")
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("Duration must be a positive finite number")
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,120}", run_id):
        raise ValueError("Run ID must contain only letters, digits, underscores, and hyphens")
    if "nodes" in template:
        graph, groups = _flatten(copy.deepcopy(template), object_info)
    else:
        graph, groups = copy.deepcopy(template.get("prompt", template)), {}
    finals = [i for i, n in graph.items() if n.get("class_type") in {"SaveVideo", "H3CombineChain"}]
    preferred = [i for i in finals if "final" in graph[i].get("_meta", {}).get("title", "").lower()]
    if len(preferred) == 1:
        final = preferred[0]
    elif len(finals) == 1:
        final = finals[0]
    else:
        raise ValueError(
            "Template needs one explicit final video output (title its SaveVideo 'FINAL')"
        )
    projects = [i for i, n in graph.items() if n.get("class_type") == "H3RCProject"]
    prompt_durations = {}
    actual = duration
    if len(projects) == 1:
        p = graph[projects[0]]["inputs"]
        p["width"], p["height"] = resolve_dimensions(
            megapixels, p["width"], p["height"], aspect_ratio)
        p.update(size_mode="width_height", megapixels=megapixels)
        prompt_durations, actual = _prepare_chain(
            graph, groups, projects[0], duration, run_id, final
        )
    elif projects:
        raise ValueError("Multiple H3 projects in one template are unsupported")
    else:
        raise ValueError(
            "Automatic duration and megapixel settings currently require an H3 Reference Chain native-subgraph template"
        )
    graph[final]["inputs"]["filename_prefix"] = f"simpleGen/{run_id}/final"
    prompt_slots = []
    reference_slots = []
    for node_id, node in graph.items():
        kind, inputs = node["class_type"], node["inputs"]
        for name, value in inputs.items():
            if not isinstance(value, str):
                continue
            if name in {"prompt", "text", "positive_prompt", "negative_prompt"} or (
                name == "value" and kind == "PrimitiveStringMultiline"
            ):
                slot = {"node_id": node_id, "input": name, "text": value}
                if node_id in prompt_durations:
                    slot["duration_seconds"] = prompt_durations[node_id]
                    slot["overlap_seconds"] = (
                        0.0 if node_id.startswith("sg1:") else p["context_frames"] / 24
                    )
                    slot["raw_duration_seconds"] = (
                        slot["duration_seconds"] + slot["overlap_seconds"]
                    )
                prompt_slots.append(slot)
            if (kind.startswith("Load") or "LoadVideo" in kind) and name in {
                "image",
                "video",
                "audio",
            }:
                reference_slots.append(
                    {
                        "node_id": node_id,
                        "input": name,
                        "value": value,
                        "title": node.get("_meta", {}).get("title", kind),
                    }
                )
    return {
        "graph": graph,
        "final_node": final,
        "prompt_slots": prompt_slots,
        "reference_slots": reference_slots,
        "summary": f"H3 Reference Chain: {len(prompt_durations)} segments, requested {duration:g}s, actual {actual:.3f}s at 24 fps; {p['width']} × {p['height']} ({p['width'] * p['height'] / 1_000_000:.3f} MP). Final and all segment videos are saved.",
    }


def _expanded_specs(info: dict, inputs: dict) -> tuple[dict, set]:
    specs = _specs(info)
    required = set(info.get("input", {}).get("required", {}))

    def expand(name: str, spec: list) -> None:
        options = spec[1] if len(spec) > 1 else {}
        if spec[0] == "COMFY_DYNAMICCOMBO_V3":
            selected = next(
                (x for x in options.get("options", []) if x["key"] == inputs.get(name)), None
            )
            if selected:
                for section in ("required", "optional"):
                    for child, child_spec in selected.get("inputs", {}).get(section, {}).items():
                        key = f"{name}.{child}"
                        specs[key] = child_spec
                        if section == "required":
                            required.add(key)
                        expand(key, child_spec)
        elif spec[0] == "COMFY_AUTOGROW_V3":
            template = options.get("template", {})
            prefix = template.get("prefix", options.get("prefix", ""))
            nested = template.get("input", {}).get("required", {})
            if len(nested) != 1:
                raise ValueError(f"Unsupported dynamic input schema {name}")
            child_spec = next(iter(nested.values()))
            for key in inputs:
                if key.startswith(f"{name}.{prefix}"):
                    suffix = key[len(name) + 1 + len(prefix) :]
                    if not suffix.isdigit() or not 0 <= int(suffix) < template.get(
                        "max", options.get("max", 1000)
                    ):
                        raise ValueError(f"Invalid dynamic input {key}")
                    specs[key] = child_spec

    for name, spec in list(specs.items()):
        expand(name, spec)
    return specs, required


MATCHTYPE = "COMFY_MATCHTYPE_V3"


def _matchtype_allowed(spec: list) -> str:
    """A MatchType input accepts its template's allowed types (comma-separated)."""
    options = spec[1] if len(spec) > 1 else {}
    return str(options.get("template", {}).get("allowed_types", "*")) or "*"


def _output_type(graph: dict, object_info: dict, node_id: str, slot: int, seen: frozenset = frozenset()):
    """Resolve a MatchType output to the type linked into its matching input."""
    info = object_info[graph[node_id]["class_type"]]
    actual = info.get("output", [])[slot]
    if isinstance(actual, list) or actual != MATCHTYPE:
        return actual
    template_id = (info.get("output_matchtypes") or [None] * (slot + 1))[slot]
    allowed = "*"
    for name, spec in _specs(info).items():
        options = spec[1] if len(spec) > 1 else {}
        if spec[0] != MATCHTYPE or options.get("template", {}).get("template_id") != template_id:
            continue
        allowed = _matchtype_allowed(spec)
        value = graph[node_id].get("inputs", {}).get(name)
        if _link(value) and value[0] in graph and node_id not in seen:
            source = graph[value[0]].get("class_type")
            if source in object_info and 0 <= value[1] < len(object_info[source].get("output", [])):
                return _output_type(graph, object_info, value[0], value[1], seen | {node_id})
    return allowed


def validate_graph(graph: dict, object_info: dict, *, check_model_availability: bool = True) -> None:
    """Reject missing nodes, invalid sockets, incompatible types, values and cycles."""
    if not graph:
        raise ValueError("Workflow graph is empty")
    dependencies = {str(i): set() for i in graph}
    for node_id, node in graph.items():
        kind = node.get("class_type")
        if kind not in object_info:
            raise ValueError(f"ComfyUI is missing node class {kind}")
        inputs = node.get("inputs", {})
        specs, required = _expanded_specs(object_info[kind], inputs)
        missing = required - inputs.keys()
        if missing:
            raise ValueError(
                f"Node {node_id} ({kind}) is missing required inputs: {', '.join(sorted(missing))}"
            )
        for name, value in inputs.items():
            if name not in specs:
                raise ValueError(f"Node {node_id}: unsupported input {name}")
            spec = specs[name]
            expected, options = spec[0], spec[1] if len(spec) > 1 else {}
            if _link(value):
                source_id, slot = value
                if source_id not in graph:
                    raise ValueError(f"Node {node_id}.{name}: missing source {source_id}")
                source_kind = graph[source_id].get("class_type")
                if source_kind not in object_info:
                    raise ValueError(f"ComfyUI is missing node class {source_kind}")
                outputs = object_info[source_kind].get("output", [])
                if slot < 0 or slot >= len(outputs):
                    raise ValueError(f"Node {node_id}.{name}: invalid output slot {slot}")
                actual = _output_type(graph, object_info, source_id, slot)
                actual = "COMBO" if isinstance(actual, list) else actual
                expected = "COMBO" if isinstance(expected, list) else expected
                expected = _matchtype_allowed(spec) if expected == MATCHTYPE else expected
                if (
                    actual != "*"
                    and expected != "*"
                    and not set(str(actual).split(",")) & set(str(expected).split(","))
                ):
                    raise ValueError(
                        f"Node {node_id}.{name}: incompatible connection {actual} → {expected}"
                    )
                dependencies[node_id].add(source_id)
                continue
            choices = (
                expected
                if isinstance(expected, list)
                else options.get("options")
                if expected == "COMBO"
                else None
            )
            if expected == "COMFY_DYNAMICCOMBO_V3":
                choices = [x["key"] for x in options.get("options", [])]
            if choices is not None:
                # Creative preparation preserves model names even when this
                # machine lacks them. Execution always uses the strict default.
                deferred_model = not check_model_availability and name in {
                    "model", "model_name", "ckpt_name", "unet_name", "vae_name",
                    "clip_name", "clip_name1", "clip_name2", "clip_name3", "lora_name",
                } and isinstance(value, str) and bool(value.strip())
                # Image/audio/video names are supplied by upload and /object_info may predate it.
                uploaded = any(options.get(k) for k in ("image_upload", "audio_upload", "video_upload"))
                if value not in choices and not uploaded and not deferred_model:
                    raise ValueError(
                        f"Node {node_id}.{name}: unavailable choice {value!r}; check models or template compatibility"
                    )
            elif expected == "INT":
                if not isinstance(value, int) or isinstance(value, bool):
                    raise ValueError(f"Node {node_id}.{name}: expected integer")
            elif expected == "FLOAT":
                if (
                    not isinstance(value, (int, float))
                    or isinstance(value, bool)
                    or not math.isfinite(value)
                ):
                    raise ValueError(f"Node {node_id}.{name}: expected finite number")
            elif expected == "STRING":
                if not isinstance(value, str):
                    raise ValueError(f"Node {node_id}.{name}: expected text")
            elif expected == "BOOLEAN":
                if not isinstance(value, bool):
                    raise ValueError(f"Node {node_id}.{name}: expected boolean")
            elif expected != "*":
                raise ValueError(f"Node {node_id}.{name}: expected a {expected} connection")
            if (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and (
                    not math.isfinite(value)
                    or ("min" in options and value < options["min"])
                    or ("max" in options and value > options["max"])
                )
            ):
                raise ValueError(f"Node {node_id}.{name}: value outside supported bounds")
    dependents: dict[str, list[str]] = {i: [] for i in dependencies}
    for target, sources in dependencies.items():
        for source in sources:
            dependents[source].append(target)
    ready = [i for i, dependencies_for_node in dependencies.items() if not dependencies_for_node]
    visited = set()
    while ready:
        current = ready.pop()
        if current in visited:
            continue
        visited.add(current)
        for target in dependents[current]:
            dependencies[target].remove(current)
            if not dependencies[target]:
                ready.append(target)
    if len(visited) != len(graph):
        raise ValueError("Workflow graph contains a cycle")
