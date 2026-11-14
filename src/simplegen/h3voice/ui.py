"""Convert a built API graph into a ComfyUI frontend workflow (drag-and-drop JSON).

Widget order and dynamic-combo children come from the target's schema, so the
same graph exports correctly for local and RunPod even where node versions
differ (e.g. SaveVideo's codec layout).
"""

from __future__ import annotations

import uuid

from . import schema as S

LANE_TITLES = ["Models", "References (pictures, voices)", "Conditioning", "Sampling", "Decode + save"]
LANE_X = [0, 420, 900, 1420, 1840]
LANE_W = [380, 400, 480, 380, 380]


def _widgets_values(node: dict, info: dict) -> list:
    values = []
    for name, spec, _ in S.walk(info, node["inputs"]):
        if not S._is_widget(spec):
            continue
        value = node["inputs"].get(name, S._default(spec))
        if S._is_link(value):
            value = S._default(spec)
        values.append(value)
        opts = spec[1] if len(spec) > 1 else {}
        if opts.get("control_after_generate"):
            values.append("fixed")
        if opts.get("image_upload"):
            values.append("image")
    return values


def _height(cls: str, n_widgets: int, n_inputs: int) -> int:
    if cls == "MiniMaxH3ReferenceToVideo":
        return 700
    if cls == "LoadImage":
        return 330
    return 60 + 26 * (n_widgets + n_inputs)


TEMPLATE_FORMAT = 1  # bump when extra.h3voice changes shape (simpleGen reads it)


def template_meta(profile: dict) -> dict:
    """What a consumer (simpleGen single-shot mode) needs to rebuild this recipe."""
    return {"format": TEMPLATE_FORMAT, "kind": "ref2va_voice_single_shot",
            "profile": {k: profile.get(k) for k in ("name", "models", "lora", "render")}}


def export(graph: dict, object_info: dict, notes: list[tuple[str, str]] | None = None,
           lanes: dict[str, int] | None = None, profile: dict | None = None) -> dict:
    lanes = lanes or {}
    ids = {nid: i for i, nid in enumerate(graph, 1)}
    nodes, links = [], []
    outputs: dict[str, list] = {}
    y = [120] * len(LANE_X)
    link_id = 0

    for nid, node in graph.items():
        info = object_info[node["class_type"]]
        outputs[nid] = [
            {"name": name, "type": typ, "links": [], "localized_name": name}
            for typ, name in zip(info.get("output", []), info.get("output_name", info.get("output", [])))
        ]

    for nid, node in graph.items():
        cls, info = node["class_type"], object_info[node["class_type"]]
        inputs = []
        order = [name for name, spec, _ in S.walk(info, node["inputs"]) if not S._is_widget(spec)]
        linked = [(k, v) for k, v in node["inputs"].items() if S._is_link(v)]
        linked.sort(key=lambda kv: (order.index(kv[0].split(".")[0]) if kv[0].split(".")[0] in order else 99, kv[0]))
        for name, (src, slot) in linked:
            link_id += 1
            typ = outputs[src][slot]["type"]
            entry = {"name": name, "localized_name": name, "type": typ, "link": link_id}
            if "." in name:
                entry.update(label=name.split(".", 1)[1], shape=7)
            inputs.append(entry)
            outputs[src][slot]["links"].append(link_id)
            links.append([link_id, ids[src], slot, ids[nid], len(inputs) - 1, typ])
        widgets = _widgets_values(node, info)
        lane = lanes.get(nid, 0)
        h = _height(cls, len(widgets), len(inputs))
        nodes.append({
            "id": ids[nid], "type": cls, "pos": [LANE_X[lane], y[lane]], "size": [LANE_W[lane], h],
            "flags": {}, "order": ids[nid] - 1, "mode": 0, "title": node["_meta"]["title"],
            "inputs": inputs, "outputs": outputs[nid],
            "properties": {"Node name for S&R": cls, "h3voice_id": nid},
            "widgets_values": widgets,
        })
        y[lane] += h + 60

    for nid in graph:
        for out in outputs[nid]:
            out["links"] = out["links"] or None

    groups = []
    for lane, title in enumerate(LANE_TITLES):
        if y[lane] > 120:
            groups.append({"id": lane + 1, "title": title, "bounding": [LANE_X[lane] - 20, 60, LANE_W[lane] + 40, y[lane] - 40],
                           "color": "#3f789e", "font_size": 22, "flags": {}})
    next_id = len(ids) + 1
    for i, (title, text) in enumerate(notes or []):
        nodes.append({"id": next_id + i, "type": "MarkdownNote", "pos": [-520, 120 + i * 560], "size": [480, 520],
                      "flags": {}, "order": next_id + i - 1, "mode": 0, "title": title, "inputs": [], "outputs": [],
                      "properties": {}, "widgets_values": [text]})
    return {
        "id": str(uuid.uuid5(uuid.NAMESPACE_URL, "h3voice/" + ",".join(graph))),
        "revision": 0, "last_node_id": next_id + len(notes or []) - 1, "last_link_id": link_id,
        "nodes": nodes, "links": links, "groups": groups, "config": {},
        "extra": {"ds": {"scale": 0.6, "offset": [620, 40]},
                  "h3voice": template_meta(profile) if profile else {"format": TEMPLATE_FORMAT}},
        "version": 0.4,
    }
