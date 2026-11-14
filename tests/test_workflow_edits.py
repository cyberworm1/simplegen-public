"""Agent-authored native edits; synthetic fixtures, optional external read-only checks."""

import copy
import json
import os
from pathlib import Path

import pytest
import test_workflows as source_fixtures

from simplegen.workflow_edits import edit_contract, validate_edited_workflow


@pytest.fixture(name="info")
def _info():
    return source_fixtures.info.__wrapped__()


@pytest.fixture(name="workflow")
def _workflow():
    return source_fixtures.workflow.__wrapped__()


def author_edit(source, contract):
    """Simulate an agent directly editing the native cards and their links."""
    result = copy.deepcopy(source)
    nodes = {n["id"]: n for n in result["nodes"]}
    old_ids = contract["segment_node_ids"]
    old_nodes = [nodes[i] for i in old_ids]
    count = len(contract["legal_frame_counts"])
    next_id = max(nodes) + 1
    new_ids = old_ids[:count]
    for index in range(len(new_ids), count):
        clone = copy.deepcopy(old_nodes[-1])
        clone["id"] = next_id
        nodes[next_id] = clone
        new_ids.append(next_id)
        next_id += 1
    deleted = set(old_ids) - set(new_ids)
    old_edges = [edge[:5] for edge in result["links"]]
    for _, origin, _, target, _ in old_edges:
        if origin in deleted and nodes[target]["type"] == "PreviewAny":
            deleted.add(target)
    nodes = {i: n for i, n in nodes.items() if i not in deleted}
    for key, value in contract["resolution"].items():
        mapping = next((m for m in contract["project_widgets"] if m["input"] == key), None)
        if mapping:
            nodes[contract["project_node_id"]]["widgets_values"][mapping["widget_index"]] = value
    for index, node_id in enumerate(new_ids):
        for mapping in contract["segment_widgets"]:
            name = mapping["input"]
            if name == "duration_override":
                nodes[node_id]["widgets_values"][mapping["widget_index"]] = contract["timing"][
                    "recommended_segments"
                ][index]["duration_override"]
            elif name == "prompt":
                nodes[node_id]["widgets_values"][mapping["widget_index"]] = (
                    f"New concept scene {index + 1}: a paper kite travels across a quiet field."
                )
    for ref in contract["reference_nodes"]:
        nodes[ref["node_id"]]["widgets_values"][ref["widget_index"]] = (
            f"/tmp/reference-{ref['node_id']}.png"
        )
    final_id = contract["final_node_id"]
    edges = [
        (o, s, t, ts)
        for _, o, s, t, ts in old_edges
        if o not in deleted and t not in deleted and t not in old_ids and t != final_id
    ]
    for index, node_id in enumerate(new_ids):
        inputs = nodes[node_id]["inputs"]
        project_slot = next(i for i, inp in enumerate(inputs) if inp["name"] == "project")
        previous_slot = next(i for i, inp in enumerate(inputs) if inp["name"] == "previous")
        edges.append((contract["project_node_id"], 0, node_id, project_slot))
        if index:
            edges.append((new_ids[index - 1], 0, node_id, previous_slot))
    final_slot = next(
        i for i, inp in enumerate(nodes[final_id]["inputs"]) if inp["name"] == "video"
    )
    edges.append((new_ids[-1], 1, final_id, final_slot))
    for node in nodes.values():
        for inp in node.get("inputs", []):
            inp["link"] = None
        for output in node.get("outputs", []):
            output["links"] = []
    result["links"] = []
    for link_id, (origin, slot, target, target_slot) in enumerate(edges, 1):
        result["links"].append([link_id, origin, slot, target, target_slot, "*"])
        nodes[target]["inputs"][target_slot]["link"] = link_id
        if "outputs" in nodes[origin]:
            nodes[origin]["outputs"][slot]["links"].append(link_id)
    result["nodes"] = list(nodes.values())
    return result


def prepare(source, info, duration=22):
    options = {"megapixels": 0.3, "duration": duration, "run_id": "native_test"}
    contract = edit_contract(source, info, **options)
    return author_edit(source, contract), contract, options


def test_selected_default_steps_must_be_honored_by_agent(workflow, info):
    info["H3RCProject"]["input"]["optional"]["default_steps"] = ["INT", {"default": 25, "min": 1, "max": 100}]
    project = next(n for n in workflow["nodes"] if n["type"] == "H3RCProject")
    project["widgets_values"].extend(["width_height", 1.0, 25])
    options = {"megapixels": 0.3, "duration": 15, "run_id": "steps-test", "steps": 37}
    contract = edit_contract(workflow, info, **options)
    assert contract["quality"] == {"default_steps": 37, "steps_override": 0}
    edited = author_edit(workflow, contract)
    with pytest.raises(ValueError, match="default_steps"):
        validate_edited_workflow(workflow, edited, info, **options)
    edited_project = next(n for n in edited["nodes"] if n["type"] == "H3RCProject")
    edited_project["widgets_values"][-1] = 37
    result = validate_edited_workflow(workflow, edited, info, **options)
    assert result["graph"]["1"]["inputs"]["default_steps"] == 37
    # Old accepted runs do not acquire a newly imposed steps requirement.
    edited_project["widgets_values"][-1] = 25
    validate_edited_workflow(workflow, edited, info, **{**options, "steps": None})


def test_agent_authored_clones_are_retained_without_automatic_topology(workflow, info):
    source = copy.deepcopy(workflow)
    edited, contract, options = prepare(source, info, duration=60)
    original_edit = copy.deepcopy(edited)
    result = validate_edited_workflow(source, edited, info, **options)
    assert source == workflow
    assert edited == original_edit
    assert len(result["prompt_slots"]) == len(contract["legal_frame_counts"]) == 9
    assert result["graph"]["3"]["inputs"]["video"][0].startswith(str(edited["links"][-1][1]) + ":")
    assert sum(n["class_type"] == "SaveVideo" for n in result["graph"].values()) == 10
    assert all(slot["text"].startswith("New concept") for slot in result["prompt_slots"])
    assert "placeholder" not in json.dumps(contract)
    json.dumps(result)


@pytest.mark.parametrize("duration", [4, 8, 15, 22, 60, 300])
def test_duration_contract_windows_are_valid(workflow, info, duration):
    edited, _, options = prepare(workflow, info, duration)
    validate_edited_workflow(workflow, edited, info, **options)


def test_agent_can_shorten_an_existing_native_chain(workflow, info):
    long, _, _ = prepare(workflow, info, 60)
    # The long native copy becomes an independent input template for this test.
    contract = edit_contract(long, info, megapixels=0.3, duration=4, run_id="short")
    edited = author_edit(long, contract)
    # Ensure this run writes fresh text rather than carrying over prior text.
    for n in edited["nodes"]:
        if n["type"] == contract["segment_type"]:
            n["widgets_values"][-1] = "A newly authored short scene."
    result = validate_edited_workflow(
        long, edited, info, megapixels=0.3, duration=4, run_id="short"
    )
    assert len(result["prompt_slots"]) == 1


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda w: w["nodes"].append(copy.deepcopy(w["nodes"][0])), "duplicate node"),
        (lambda w: w["links"].append(copy.deepcopy(w["links"][0])), "duplicate link"),
        (lambda w: w["nodes"][1]["inputs"][0].update(link=999), "nonreciprocal"),
        (lambda w: w["nodes"][0].update(type="InjectedNode"), "class, socket"),
        (lambda w: w["nodes"][0].update(properties={"injected": True}), "properties"),
        (lambda w: w.update(extra={"injected": True}), "metadata"),
        (
            lambda w: w["definitions"]["subgraphs"][0]["nodes"][0].update(type="InjectedNode"),
            "definitions",
        ),
        (lambda w: w["nodes"][1]["widgets_values"].__setitem__(-1, "write here"), "source sample"),
        (lambda w: w["nodes"][1]["widgets_values"].__setitem__(-1, ""), "nonempty"),
        (lambda w: w["nodes"][0]["widgets_values"].__setitem__(0, 1024), "resolution"),
        (lambda w: w["nodes"][1]["widgets_values"].__setitem__(0, 3), "4–15"),
        (lambda w: w["nodes"][0]["widgets_values"].__setitem__(3, 5), "unapproved setting"),
        (
            lambda w: w["nodes"][2]["widgets_values"].__setitem__(0, "another/location"),
            "fixed widgets",
        ),
        (
            lambda w: w["nodes"][0].update(widgets_values={"width": 736, "injected": True}),
            "widget representation",
        ),
    ],
)
def test_invalid_native_edits_fail(workflow, info, mutation, match):
    edited, _, options = prepare(workflow, info)
    mutation(edited)
    with pytest.raises(ValueError, match=match):
        validate_edited_workflow(workflow, edited, info, **options)


def test_wrong_duration_fails_instead_of_rewriting_segments(workflow, info):
    edited, _, options = prepare(workflow, info, 4)
    with pytest.raises(ValueError, match="Edited chain produces"):
        validate_edited_workflow(workflow, edited, info, **{**options, "duration": 60})


def test_wrong_final_segment_is_rejected(workflow, info):
    edited, _, options = prepare(workflow, info, 60)
    edited["links"][-1][1] = 2
    with pytest.raises(ValueError, match="last segment"):
        validate_edited_workflow(workflow, edited, info, **options)


def test_branch_is_rejected(workflow, info):
    edited, _, options = prepare(workflow, info, 60)
    previous_edges = [edge for edge in edited["links"] if edge[2] == 0 and edge[4] == 1]
    previous_edges[-1][1] = previous_edges[0][1]
    with pytest.raises(ValueError, match="linear chain"):
        validate_edited_workflow(workflow, edited, info, **options)


def test_cycle_is_rejected(workflow, info):
    edited, _, options = prepare(workflow, info, 60)
    previous_edges = [edge for edge in edited["links"] if edge[2] == 0 and edge[4] == 1]
    previous_edges[0][1] = previous_edges[0][3]
    with pytest.raises(ValueError, match="linear chain"):
        validate_edited_workflow(workflow, edited, info, **options)


def test_output_reciprocity_is_checked(workflow, info):
    workflow["nodes"][0]["outputs"] = [{"name": "project", "type": "H3RC_PROJECT", "links": [1]}]
    edited, _, options = prepare(workflow, info)
    edited["nodes"][0]["outputs"][0]["links"] = []
    with pytest.raises(ValueError, match="nonreciprocal"):
        validate_edited_workflow(workflow, edited, info, **options)


def test_contract_rejects_unavailable_models_before_agent(workflow, info):
    workflow["nodes"].append({"id": 20, "type": "Loader", "widgets_values": ["missing"]})
    info["Loader"] = {"input": {"required": {"model": [["installed"]]}}, "output": ["MODEL"]}
    with pytest.raises(ValueError, match="unavailable choice"):
        edit_contract(workflow, info, megapixels=0.3, duration=8, run_id="preflight")


def test_preparation_defers_only_model_availability(workflow, info):
    workflow["nodes"].append({"id": 20, "type": "Loader", "widgets_values": ["missing"]})
    info["Loader"] = {"input": {"required": {"model": [["installed"]]}}, "output": ["MODEL"]}
    edit_contract(workflow, info, megapixels=0.3, duration=8, run_id="prepare",
                  check_model_availability=False)
    info["Loader"]["input"]["required"] = {"precision": [["fp16", "bf16"]]}
    with pytest.raises(ValueError, match="unavailable choice"):
        edit_contract(workflow, info, megapixels=0.3, duration=8, run_id="prepare",
                      check_model_availability=False)


def test_reference_paths_must_be_absolute(workflow, info):
    workflow["nodes"].append(
        {"id": 20, "type": "LoadImage", "widgets_values": ["source.png", "image"]}
    )
    info["LoadImage"] = {
        "input": {"required": {"image": [["source.png"], {"image_upload": True}]}},
        "output": ["IMAGE"],
    }
    edited, _, options = prepare(workflow, info)
    reference = next(n for n in edited["nodes"] if n["type"] == "LoadImage")
    reference["widgets_values"][0] = "relative.png"
    with pytest.raises(ValueError, match="absolute local"):
        validate_edited_workflow(workflow, edited, info, **options)
    reference["widgets_values"][0] = "/tmp/chosen.png"
    result = validate_edited_workflow(workflow, edited, info, **options)
    assert result["reference_slots"][0]["value"] == "/tmp/chosen.png"


def test_external_native_template_optional():
    path, info_path = os.getenv("SIMPLEGEN_TEST_TEMPLATE"), os.getenv("SIMPLEGEN_TEST_OBJECT_INFO")
    if not path or not info_path:
        pytest.skip("Set external template/object_info paths for read-only native validation")
    source = json.loads(Path(path).read_text())
    info = json.loads(Path(info_path).read_text())
    # The template deliberately targets different models from this workstation.
    # Modify an isolated metadata copy only, never its native model selections.
    from simplegen.workflows import _widgets

    for node in source["nodes"]:
        if node["type"] not in info:
            continue
        for key, value in _widgets(node, info[node["type"]]).items():
            spec = info[node["type"]]["input"].get("required", {}).get(key)
            if spec and isinstance(spec[0], list) and value not in spec[0]:
                spec[0].append(value)
    for duration in (4, 22, 60):
        edited, _, options = prepare(source, info, duration)
        result = validate_edited_workflow(source, edited, info, **options)
        assert result["reference_slots"][0]["value"].startswith("/tmp/")
        assert any(i.startswith("simplegen_raw_") for i in result["graph"])
