"""Synthetic graphs only: no production templates or sample prompts are bundled."""

import copy
import json
import os
from pathlib import Path

import pytest

from simplegen.workflows import _duration_frames, _flatten, prepare_workflow, validate_graph


def schema(required, outputs=(), optional=None):
    return {"input": {"required": required, "optional": optional or {}}, "output": list(outputs)}


@pytest.fixture
def info():
    return {
        "H3RCProject": schema(
            {
                "width": ["INT"],
                "height": ["INT"],
                "default_seconds": ["FLOAT"],
                "context_frames": ["INT"],
                "checkpoint_prefix": ["STRING"],
            },
            ["H3RC_PROJECT"],
            {"size_mode": [["width_height", "megapixels"]], "megapixels": ["FLOAT"]},
        ),
        "H3RCResolveSegment": schema(
            {"project": ["H3RC_PROJECT"], "duration_override": ["FLOAT"]}, ["INT"]
        ),
        "H3RCPreviousState": schema(
            {"project": ["H3RC_PROJECT"]}, ["INT"], {"previous": ["H3RC_STATE"]}
        ),
        "H3RCState": schema(
            {"project": ["H3RC_PROJECT"], "previous": ["INT"], "prompt": ["STRING"]},
            ["H3RC_STATE", "VIDEO", "VIDEO"],
        ),
        "PrimitiveStringMultiline": schema({"value": ["STRING", {"multiline": True}]}, ["STRING"]),
        "SaveVideo": schema(
            {"video": ["VIDEO"], "filename_prefix": ["STRING"], "format": [["mp4"]]}, []
        ),
    }


@pytest.fixture
def workflow():
    # A small independently authored analogue of the native H3RC subgraph boundary.
    segment = {
        "id": "segment",
        "inputs": [
            {"name": "project"},
            {"name": "previous"},
            {"name": "duration_override"},
            {"name": "prompt"},
        ],
        "outputs": [
            {"name": "next_segment"},
            {"name": "assembled_video"},
            {"name": "new_segment_video"},
        ],
        "nodes": [
            {
                "id": 1,
                "type": "H3RCResolveSegment",
                "widgets_values": [0.0],
                "inputs": [
                    {"name": "project", "link": 1},
                    {"name": "duration_override", "link": 2},
                ],
            },
            {
                "id": 2,
                "type": "H3RCPreviousState",
                "inputs": [{"name": "project", "link": 3}, {"name": "previous", "link": 4}],
            },
            {
                "id": 3,
                "type": "PrimitiveStringMultiline",
                "widgets_values": ["placeholder"],
                "inputs": [{"name": "value", "link": 5}],
            },
            {
                "id": 4,
                "type": "H3RCState",
                "inputs": [
                    {"name": "project", "link": 6},
                    {"name": "previous", "link": 7},
                    {"name": "prompt", "link": 8},
                ],
            },
        ],
        "links": [
            [1, -10, 0, 1, 0],
            [2, -10, 2, 1, 1],
            [3, -10, 0, 2, 0],
            [4, -10, 1, 2, 1],
            [5, -10, 3, 3, 0],
            [6, -10, 0, 4, 0],
            [7, 2, 0, 4, 1],
            [8, 3, 0, 4, 2],
            [9, 4, 0, -20, 0],
            [10, 4, 1, -20, 1],
            [11, 4, 2, -20, 2],
        ],
    }
    return {
        "definitions": {"subgraphs": [segment]},
        "nodes": [
            {"id": 1, "type": "H3RCProject", "widgets_values": [1344, 768, 8.0, 22, "old"]},
            {
                "id": 2,
                "type": "segment",
                "widgets_values": [0.0, "write here"],
                "inputs": [
                    {"name": "project", "link": 1},
                    {"name": "previous", "link": None},
                    {"name": "duration_override", "widget": {"name": "duration_override"}},
                    {"name": "prompt", "widget": {"name": "prompt"}},
                ],
            },
            {
                "id": 3,
                "type": "SaveVideo",
                "title": "FINAL",
                "widgets_values": ["original", "mp4"],
                "inputs": [{"name": "video", "link": 2}],
            },
        ],
        "links": [[1, 1, 0, 2, 0], [2, 2, 1, 3, 0]],
    }


def test_primary_chain_expands_and_saves_shots_without_mutating_source(workflow, info):
    original = copy.deepcopy(workflow)
    result = prepare_workflow(workflow, info, megapixels=0.3, duration=60, run_id="run_1")
    validate_graph(result["graph"], info)
    assert workflow == original
    assert len(result["prompt_slots"]) == 9
    assert len([n for n in result["graph"].values() if n["class_type"] == "SaveVideo"]) == 10
    assert result["graph"]["1"]["inputs"]["width"] == 736
    assert result["graph"]["1"]["inputs"]["height"] == 416
    assert result["graph"]["3"]["inputs"]["video"] == ["sg9:4", 1]
    assert result["graph"]["sg1:2"]["inputs"].get("previous") is None
    assert result["graph"]["sg2:2"]["inputs"]["previous"] == ["sg1:4", 0]
    assert sum(s["duration_seconds"] for s in result["prompt_slots"]) == pytest.approx(60.4166667)


def test_small_duration_has_single_segment(workflow, info):
    result = prepare_workflow(workflow, info, megapixels=1, duration=4, run_id="short")
    validate_graph(result["graph"], info)
    assert len(result["prompt_slots"]) == 1
    assert result["prompt_slots"][0]["duration_seconds"] == 107 / 24


def test_existing_chain_shortens_and_moves_final_connection(workflow, info):
    second = copy.deepcopy(workflow["nodes"][1])
    second["id"] = 4
    second["inputs"][0]["link"] = 3
    second["inputs"][1]["link"] = 4
    workflow["nodes"].append(second)
    workflow["links"][1][1] = 4
    workflow["links"].extend([[3, 1, 0, 4, 0], [4, 2, 0, 4, 1]])
    result = prepare_workflow(workflow, info, megapixels=1, duration=4, run_id="short")
    validate_graph(result["graph"], info)
    assert result["graph"]["3"]["inputs"]["video"] == ["sg1:4", 1]
    assert len(result["prompt_slots"]) == 1


@pytest.mark.parametrize("context", [5, 22, 39])
@pytest.mark.parametrize("duration", [4, 8, 15, 22, 60, 300])
def test_timing_lattice_and_overlap(context, duration):
    frames = _duration_frames(duration, 8, context)
    actual = (sum(frames) - (len(frames) - 1) * context) / 24
    assert all(107 <= count <= 362 and count % 17 == 5 for count in frames)
    assert duration <= actual < duration + 17 / 24


def test_promoted_widget_propagates(workflow, info):
    graph, _ = _flatten(workflow, info)
    assert graph["2:3"]["inputs"]["value"] == "write here"
    assert graph["2:1"]["inputs"]["duration_override"] == 0
    assert "previous" not in graph["2:2"]["inputs"]


def test_dynamic_widgets_and_seed_control():
    info = {
        "Node": schema(
            {
                "seed": ["INT", {"control_after_generate": True}],
                "format": [
                    "COMFY_DYNAMICCOMBO_V3",
                    {
                        "options": [
                            {
                                "key": "mp4",
                                "inputs": {"required": {"codec": [["h264"]], "crf": ["FLOAT"]}},
                            }
                        ]
                    },
                ],
            },
            ["VIDEO"],
        )
    }
    graph, _ = _flatten(
        {
            "nodes": [
                {"id": 1, "type": "Node", "widgets_values": [4, "randomize", "mp4", "h264", 18.0]}
            ]
        },
        info,
    )
    assert graph["1"]["inputs"] == {
        "seed": 4,
        "format": "mp4",
        "format.codec": "h264",
        "format.crf": 18.0,
    }
    validate_graph(graph, info)


def test_missing_node_class(workflow, info):
    del info["H3RCState"]
    with pytest.raises(ValueError, match="missing node class"):
        prepare_workflow(workflow, info, megapixels=1, duration=8, run_id="x")


@pytest.mark.parametrize(
    "mutation, message",
    [
        (lambda g: g["b"]["inputs"].clear(), "missing required"),
        (lambda g: g["b"]["inputs"].update(value=["absent", 0]), "missing source"),
        (lambda g: g["b"]["inputs"].update(value=["a", 1]), "invalid output slot"),
        (lambda g: g["a"]["inputs"].update(value=["b", 0]), "cycle"),
        (lambda g: g["a"]["inputs"].update(value=3), "expected text"),
        (lambda g: g["a"]["inputs"].update(unknown="x"), "unsupported input"),
    ],
)
def test_validation_failure_paths(mutation, message):
    info = {"Text": schema({"value": ["STRING"]}, ["STRING"])}
    graph = {
        "a": {"class_type": "Text", "inputs": {"value": "hello"}},
        "b": {"class_type": "Text", "inputs": {"value": ["a", 0]}},
    }
    mutation(graph)
    with pytest.raises(ValueError, match=message):
        validate_graph(graph, info)


def test_missing_model_and_type_mismatch():
    info = {
        "Loader": schema({"model": [["installed"]]}, ["MODEL"]),
        "Consumer": schema({"input": ["IMAGE"]}),
    }
    graph = {"a": {"class_type": "Loader", "inputs": {"model": "missing"}}}
    with pytest.raises(ValueError, match="unavailable choice"):
        validate_graph(graph, info)
    graph["a"]["inputs"]["model"] = "installed"
    graph["b"] = {"class_type": "Consumer", "inputs": {"input": ["a", 0]}}
    with pytest.raises(ValueError, match="incompatible"):
        validate_graph(graph, info)


def test_ambiguous_everywhere_fails():
    template = {
        "nodes": [
            {
                "id": i,
                "type": "Anything Everywhere",
                "inputs": [{"name": "anything", "type": "MODEL", "link": i}],
            }
            for i in [1, 2]
        ],
        "links": [[1, 3, 0, 1, 0], [2, 4, 0, 2, 0]],
    }
    with pytest.raises(ValueError, match="Ambiguous"):
        _flatten(template, {})


def test_external_user_template_optional():
    """Opt-in read-only local compatibility check; never submits a GPU job."""
    template_path, info_path = (
        os.getenv("SIMPLEGEN_TEST_TEMPLATE"),
        os.getenv("SIMPLEGEN_TEST_OBJECT_INFO"),
    )
    if not template_path or not info_path:
        pytest.skip("Set external template and object_info paths for compatibility check")
    template = json.loads(Path(template_path).read_text())
    info = json.loads(Path(info_path).read_text())
    result = prepare_workflow(
        template, info, megapixels=0.3, duration=60, run_id="compatibility_test"
    )
    # Local models differ from the BF16 handoff template. Preserve and separately
    # check choices, then validate all topology using an isolated schema copy.
    for node in result["graph"].values():
        for key, value in node["inputs"].items():
            spec = info[node["class_type"]]["input"].get("required", {}).get(key)
            if (
                spec
                and isinstance(spec[0], list)
                and isinstance(value, str)
                and value not in spec[0]
            ):
                spec[0].append(value)
    validate_graph(result["graph"], info)
    assert len(result["prompt_slots"]) > 3
    assert len(result["reference_slots"]) >= 4
    assert "sg_raw_1" in result["graph"]
    assert result["graph"]["sg_shot_1"]["inputs"]["video"] == ["sg_raw_1", 0]
