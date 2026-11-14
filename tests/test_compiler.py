"""Phase 3: the agent writes content, code builds the workflow (schema 5).

Real template fixtures (copied to tmp before mutation), a fake agent CLI
process and a fake ComfyUI. No real services.
"""

import copy
import hashlib
import json
import os
import shutil
import sys
from dataclasses import replace
from pathlib import Path
from threading import Event

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from test_workflows import info as _synthetic_info  # noqa: F401 -- fixture "info"
from test_workflows import workflow as _synthetic_workflow  # noqa: F401 -- fixture "workflow"

from simplegen import agents, compiler, editing, engine, refcache, repair
from simplegen.compiler import analyze, compile_workflow, dumps, lint, segment_options
from simplegen.preflight import run_saved_preflight
from simplegen.settings import Settings
from simplegen.workflow_edits import validate_edited_workflow
from simplegen.workflows import _duration_frames

REAL = Path(__file__).parent / "fixtures" / "real"
BUNDLED = Path(__file__).parents[1] / "src" / "simplegen" / "data" / "workflows"
BF16 = "H3_Persistent_References_Clonable_Chain.json"
INT8 = "H3_Persistent_References_Clonable_Chain_LOCAL_INT8_NVFP4.json"
PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32


def tools(name):
    return f"/usr/bin/{name}"


@pytest.fixture(scope="module")
def schemas():
    cloud = json.loads((REAL / "schemas" / "cloud_global_object_info.json").read_text())
    return {BF16: cloud.get("object_info", cloud),
            INT8: json.loads((REAL / "schemas" / "local_object_info.json").read_text())}


@pytest.fixture(scope="module")
def templates():
    return {name: json.loads((BUNDLED / name).read_text()) for name in (BF16, INT8)}


def good_plan(info, *, option=None, refs=None, dialogue="Hello there.", prefix="Original"):
    option = option or next(o for o in info["options"] if o["recommended"])
    refs = refs or {slot["node_id"]: f"/refs/{slot['node_id']}.png" for slot in info["reference_slots"]}
    segments = []
    for index in range(option["segment_count"]):
        sections = {name: f"{prefix} {name} for segment {index + 1}." for name in info["sections"]}
        sections[info["sections"][-2]] += f" <Subject 1> (S1) says: <d>[English] {dialogue}</d>"
        segments.append({"sections": sections})
    return {"script": "A script.", "used_references": [], "timing_option": option["id"], "references": refs,
            "segments": segments}


def settings_for(duration, **changes):
    return replace(Settings(duration=float(duration), steps=30, aspect_ratio="9:16", megapixels=0.7), **changes)


# 1 and 3. Every option compiles to a valid, deterministic workflow; options match the lattice


@pytest.mark.parametrize("name", [BF16, INT8])
def test_every_timing_option_compiles_and_validates(name, schemas, templates):
    template, info_schema = templates[name], schemas[name]
    before = json.dumps(template, sort_keys=True)
    for duration in range(5, 121):
        settings = settings_for(duration)
        info = analyze(template, info_schema, settings, "run1")
        for option in info["options"]:
            plan = good_plan(info, option=option)
            workflow = compile_workflow(template, plan, settings, "run1", object_info=info_schema)
            assert dumps(workflow) == dumps(compile_workflow(template, plan, settings, "run1",
                                                             object_info=info_schema))
            validate_edited_workflow(template, workflow, info_schema, megapixels=settings.megapixels,
                                     duration=settings.duration, run_id="run1", aspect_ratio=settings.aspect_ratio,
                                     steps=settings.steps, check_model_availability=False)
    assert json.dumps(template, sort_keys=True) == before  # The template is never mutated.


def test_timing_options_match_the_existing_lattice():
    for duration in range(4, 241):
        for default, context in ((8.0, 22), (6.0, 5), (10.0, 39)):
            options = segment_options(float(duration), default, context)
            recommended = [o for o in options if o["recommended"]]
            assert len(recommended) == 1 and 1 <= len(options) <= 3
            assert [s["frames"] for s in recommended[0]["segments"]] == _duration_frames(duration, default, context)
            counts = [o["segment_count"] for o in options]
            assert counts == sorted(counts) and all(abs(c - recommended[0]["segment_count"]) <= 1 for c in counts)
            for option in options:
                frames = [s["frames"] for s in option["segments"]]
                assert all(107 <= f <= 362 and (f - 5) % 17 == 0 for f in frames)
                actual = (sum(frames) - (len(frames) - 1) * context) / 24
                assert duration <= actual < duration + 17 / 24 + 1e-9
                assert option["actual_total_seconds"] == round(actual, 3)
                for index, segment in enumerate(option["segments"]):
                    overlap = context if index else 0
                    assert segment["usable_seconds"] == round((segment["frames"] - overlap) / 24, 3)
                    assert segment["overlap_seconds"] == round(overlap / 24, 3)
                    assert segment["speech_words"] == int((segment["frames"] - overlap) / 24 * 2.5)
                    assert segment["duration_override"] == min(15.0, segment["frames"] / 24)
    assert len(segment_options(45.0, 8.0, 22)) == 3


# 2. Structure


@pytest.mark.parametrize(("duration", "expected"), [(5, 1), (20, 3), (45, 7)])
def test_compiled_structure(schemas, templates, duration, expected):
    template, info_schema = templates[INT8], schemas[INT8]
    settings = settings_for(duration)
    info = analyze(template, info_schema, settings, "run1")
    plan = good_plan(info)
    workflow = compile_workflow(template, plan, settings, "run1", object_info=info_schema)
    nodes = {n["id"]: n for n in workflow["nodes"]}
    kind = info["segment_type"]
    cards = [n for n in workflow["nodes"] if n["type"] == kind]
    assert len(cards) == expected == len(plan["segments"])
    links = {link[0]: link for link in workflow["links"]}
    ordered, cursor = [], None
    while len(ordered) < len(cards):
        (card,) = [c for c in cards if c["id"] not in ordered and (
            links[c["inputs"][1]["link"]][1] if c["inputs"][1]["link"] else None) == cursor]
        ordered.append(card["id"])
        cursor = card["id"]
        assert links[card["inputs"][0]["link"]][1:3] == [6, 0]  # Project fan-out.
    final = nodes[19]
    assert links[final["inputs"][0]["link"]][1:3] == [ordered[-1], 1]
    reports = [n for n in workflow["nodes"] if n["type"] == "PreviewAny" and n["id"] != 12]
    assert len(reports) == min(expected, 3)
    option = next(o for o in info["options"] if o["id"] == plan["timing_option"])
    for index, card_id in enumerate(ordered):
        values = dict(zip([i["name"] for i in nodes[card_id]["inputs"] if i.get("widget")],
                          nodes[card_id]["widgets_values"]))
        assert values["duration_override"] == option["segments"][index]["duration_override"]
        assert values["steps_override"] == 0 and values["seed"] == 1001 + index
        assert values["prompt"] == compiler.render_prompt(plan["segments"][index]["sections"], info["sections"])
        assert values["prompt"].startswith("subject_definitions:\nOriginal subject_definitions")
    for slot in info["reference_slots"]:
        assert nodes[int(slot["node_id"])]["widgets_values"][0] == plan["references"][slot["node_id"]]
    original = {n["id"]: n for n in template["nodes"]}
    for loader in (1, 2, 3, 4, 5):
        assert nodes[loader] == original[loader]  # Model loaders byte-for-byte.
    assert workflow["definitions"] == template["definitions"]
    project = dict(zip(["width", "height", "default_seconds", "default_steps"], nodes[6]["widgets_values"][:4]))
    assert project["default_steps"] == 30 and (project["width"], project["height"]) == (
        info["setup"]["width"], info["setup"]["height"])
    ids = [n["id"] for n in workflow["nodes"]]
    assert len(ids) == len(set(ids)) and workflow["last_node_id"] == max(ids)
    assert workflow["last_link_id"] == max(links)


# 4. Lint


@pytest.fixture
def lint_env(tmp_path, schemas, templates):
    assets = tmp_path / "assets"
    assets.mkdir()
    (assets / "ref.png").write_bytes(PNG)
    (assets / "notes.txt").write_text("not an image")
    (assets / "fake.png").write_text("not an image either")
    settings = settings_for(20)
    info = analyze(templates[INT8], schemas[INT8], settings, "run1")
    refs = {slot["node_id"]: str(assets / "ref.png") for slot in info["reference_slots"]}
    option = next(o for o in info["options"] if o["recommended"])

    def check(plan, concept="A short scene"):
        return lint(plan, info, option, concept=concept, roots=[assets.resolve()], quantity=1)
    return info, refs, check, assets, templates[INT8]


def test_lint_passes_a_good_plan(lint_env):
    info, refs, check, _, _ = lint_env
    assert check(good_plan(info, refs=refs)) == []


@pytest.mark.parametrize("breakage", [
    "missing_section", "empty_section", "sample_prompt", "sample_section", "unbalanced", "speakers", "budget",
    "picture", "unfilled_slot", "missing_file", "not_image", "fake_image", "outside", "quoted_missing",
    "quoted_twice", "quoted_order"])
def test_each_lint_rule_fires(lint_env, breakage):
    info, refs, check, assets, _ = lint_env
    plan = good_plan(info, refs=refs)
    first = plan["segments"][0]["sections"]
    concept = "A short scene"
    lines = ["Hello, thanks for coming in.", "The weather is fine this morning."]
    if breakage.startswith("sample"):
        # The bundled templates carry headings only; a user's own template may carry sample prose.
        info["sample_prompts"] = [compiler.render_prompt(
            {name: f"Sample {name} prose that a user template might carry for segment {i + 1}, long enough to count."
             for name in info["sections"]}, info["sections"]) for i in range(3)]
    if breakage == "missing_section":
        del first[info["sections"][0]]
    elif breakage == "empty_section":
        first[info["sections"][1]] = "  "
    elif breakage == "sample_prompt":
        for segment, sample in zip(plan["segments"], info["sample_prompts"]):
            segment["sections"] = compiler.parse_sections(sample, info["sections"])
    elif breakage == "sample_section":
        sample = compiler.parse_sections(info["sample_prompts"][0], info["sections"])
        name = next(n for n in info["sections"] if len(sample.get(n, "")) >= compiler.SAMPLE_MIN_CHARS)
        first[name] = sample[name]
    elif breakage == "unbalanced":
        first[info["sections"][0]] += " <d>[English] dangling"
    elif breakage == "speakers":
        plan["segments"][1]["sections"][info["sections"][0]] += " <Subject 2> (S1) whispers."
    elif breakage == "budget":
        first[info["sections"][0]] += " <d>[English] " + " ".join(["word"] * 60) + "</d>"
    elif breakage == "picture":
        first[info["sections"][0]] += " <Picture 9> appears."
    elif breakage == "unfilled_slot":
        del plan["references"][info["reference_slots"][0]["node_id"]]
    elif breakage == "missing_file":
        plan["references"][info["reference_slots"][0]["node_id"]] = str(assets / "absent.png")
    elif breakage == "not_image":
        plan["references"][info["reference_slots"][0]["node_id"]] = str(assets / "notes.txt")
    elif breakage == "fake_image":
        plan["references"][info["reference_slots"][0]["node_id"]] = str(assets / "fake.png")
    elif breakage == "outside":
        outside = assets.parent / "elsewhere.png"
        outside.write_bytes(PNG)
        plan["references"][info["reference_slots"][0]["node_id"]] = str(outside)
    else:
        concept = f'Say "{lines[0]}" and then say “{lines[1]}”'
        spoken = {"quoted_missing": [lines[0]], "quoted_twice": [lines[0], lines[0], lines[1]],
                  "quoted_order": [lines[1], lines[0]]}[breakage]
        for index, line in enumerate(spoken):
            section = plan["segments"][index]["sections"][info["sections"][0]]
            plan["segments"][index]["sections"][info["sections"][0]] = section + f" <d>[English] {line}</d>"
    errors = check(plan, concept)
    assert errors, breakage
    if breakage.startswith("quoted"):
        good = good_plan(info, refs=refs)
        for index, line in enumerate(lines):
            good["segments"][index]["sections"][info["sections"][0]] += f" <d>[English] {line}</d>"
        assert check(good, concept) == []


# 5. Read-only agent command


def test_creative_agent_command_has_no_edit_or_write(monkeypatch, tmp_path, schemas, templates):
    monkeypatch.setattr(agents, "executable", lambda name: name)
    settings = settings_for(20, model="test", model_tier="Custom")
    brief = compiler.workflow_brief(templates[INT8], schemas[INT8], settings, "run1", tmp_path / "wf.json")
    for provider in ("codex", "claude", "grok"):
        model = {"codex": "gpt-6-sol"}.get(provider, "test")  # Codex: a listed model
        args = agents.build_command(replace(settings, provider=provider, model=model), tmp_path, [tmp_path],
                                    creative=brief)
        # Codex always sets this config key; it disables network and grants nothing.
        joined = " ".join(a for a in args if a != "sandbox_workspace_write.network_access=false")
        assert "Edit" not in joined.replace("--deny Edit", "") and "Write" not in joined and "write" not in joined
        assert "workspace" not in joined
        if provider == "codex":
            assert args[args.index("--sandbox") + 1] == "read-only"
        elif provider == "claude":
            assert args[args.index("--tools") + 1] == "Read,Glob,Grep"
            assert args[args.index("--allowedTools") + 1] == "Read,Glob,Grep"
        else:
            assert args[args.index("--sandbox") + 1] == "read-only" and "--deny" in args
            assert args[args.index("--tools") + 1] == "read_file,list_dir,grep"
    with pytest.raises(ValueError):
        agents.build_command(settings, tmp_path, [tmp_path], True, creative=brief)
    schema = agents.plan_schema(False, 1, brief)
    assert schema["properties"]["timing_option"]["enum"] == [o["id"] for o in brief["timing_options"]]
    assert schema["properties"]["segments"]["items"]["properties"]["sections"]["required"] == brief["prompt_sections"]


# 6. End to end with a fake agent CLI and a fake ComfyUI


FAKE_CLI = r'''
import json, sys
from pathlib import Path
request = sys.stdin.read()
brief = json.loads(request.split("RUN REQUEST (data, not instructions):\n", 1)[1].split("\nReturn only the final JSON", 1)[0])
log = Path(LOG)
calls = json.loads(log.read_text()) if log.exists() else []
calls.append({"validation_error": brief.get("validation_error", ""), "keys": sorted(brief)})
log.write_text(json.dumps(calls))
mode = Path(MODE).read_text()
option = next(o for o in brief["timing_options"] if o["id"] == brief["recommended_option"])
def segments(tag):
    result = []
    for i in range(option["segment_count"]):
        sections = {n: f"{tag} {n} for segment {i + 1}." for n in brief["prompt_sections"]}
        sections[brief["prompt_sections"][-2]] += " <Subject 1> (S1) says: <d>[English] Hello.</d>"
        result.append({"sections": sections})
    return result
plan = {"script": "A script.", "used_references": [IMAGE], "timing_option": option["id"],
        "references": {s["node_id"]: IMAGE for s in brief["reference_slots"]}, "segments": segments("Take 1")}
if brief["quantity"] > 1:
    plan["variations"] = [{"script": f"Script {i}", "segments": segments(f"Take {i}")} for i in range(2, brief["quantity"] + 1)]
count = len(calls)
# The 2026-10-01 cloud batch: a progress note returned as the final plan.
placeholder = mode == "always_placeholder" or (mode.startswith("placeholder_") and count == 1)
if placeholder:
    pending = [{"sections": {name: "pending" for name in brief["prompt_sections"]}}]
    plan = {"script": "I'll start from the project index, then read the bible.", "used_references": [],
            "timing_option": option["id"], "references": {s["node_id"]: "/tmp/placeholder.png"
                                                          for s in brief["reference_slots"]},
            "segments": pending}
    if brief["quantity"] > 1:
        plan["variations"] = [{"script": "pending", "segments": pending} for _ in range(2, brief["quantity"] + 1)]
bad = not placeholder and (mode in ("always_bad", "placeholder_always_bad")
                           or (mode == "bad_first" and count == 1) or (mode == "placeholder_bad_first" and count == 2))
if bad:
    plan["segments"][0]["sections"][brief["prompt_sections"][1]] = ""
# The same batch's second attempt: the workspace's read-only template listed as consulted.
if mode == "readonly_ref":
    read_only = brief["workflow_file_read_only"]
    plan["used_references"] += [read_only, str(Path(read_only).parent / "scratch.md")]
if mode == "outside_ref":
    plan["used_references"].append(OUTSIDE)
print(json.dumps({"structured_output": plan}))
'''


class Submitted(Exception):
    pass


@pytest.fixture
def e2e(tmp_path, monkeypatch, schemas):
    monkeypatch.setenv("SIMPLEGEN_HOME", str(tmp_path / "home"))
    assets, templates = tmp_path / "assets", tmp_path / "templates"
    assets.mkdir()
    shutil.copytree(BUNDLED, templates)
    image = assets / "ref.png"
    image.write_bytes(PNG)
    log, mode = tmp_path / "agent-calls.json", tmp_path / "mode.txt"
    mode.write_text("good")
    script = tmp_path / "fake_agent.py"
    outside = tmp_path / "elsewhere" / "notes.md"
    outside.parent.mkdir()
    outside.write_text("not in a linked folder")
    script.write_text(f"LOG = {str(log)!r}\nMODE = {str(mode)!r}\nIMAGE = {str(image)!r}\n"
                      f"OUTSIDE = {str(outside)!r}\n" + FAKE_CLI)
    commands = []

    def build(settings, folder, roots, workflow_edit=False, creative=None):
        commands.append({"workflow_edit": workflow_edit, "creative": creative is not None})
        return [sys.executable, str(script)]
    monkeypatch.setattr(agents, "build_command", build)
    monkeypatch.setattr(agents, "_subscription", lambda *a: None)
    monkeypatch.setattr(agents, "executable", lambda name: tools(name))
    uploaded, submitted = [], []

    class Comfy:
        def __init__(self, settings, client=None):
            pass

        def object_info(self):
            info = copy.deepcopy(schemas[INT8])
            info["LoadImage"]["input"]["required"]["image"][0].extend(uploaded)
            return info

        def queue(self):
            return {"queue_running": [], "queue_pending": []}

        def upload(self, path, run_id, cancel):
            uploaded.append(f"{run_id}/{path.name}")
            return uploaded[-1]

        def submit(self, graph, run_id, client_id):
            submitted.append(graph)
            raise Submitted("fake submit reached")

        def find_submission(self, run_id):
            return None

        def close(self):
            pass
    monkeypatch.setattr(engine, "Comfy", Comfy)
    settings = Settings(project_folder=str(assets), template_folder=str(templates), template_file=INT8,
                        output_folder=str(tmp_path / "out"), duration=20.0, steps=25, aspect_ratio="16:9")

    def calls():
        return json.loads(log.read_text()) if log.exists() else []
    return settings, mode, calls, commands, submitted


def only_run(settings):
    return next(p for p in Path(settings.output_folder).iterdir() if p.is_dir())


def test_concept_to_compiled_submission(e2e):
    settings, _, calls, commands, submitted = e2e
    with pytest.raises(RuntimeError, match="fake submit reached"):
        engine.run(settings, "A friendly welcome", lambda _: None, Event())
    folder = only_run(settings)
    state = json.loads((folder / "state.json").read_text())
    assert state["schema_version"] == 5 and len(calls()) == 1
    assert commands == [{"workflow_edit": False, "creative": True}]
    assert "workflow_file_read_only" in calls()[0]["keys"] and "editing_contract" not in calls()[0]["keys"]
    receipt = state["workflow_acceptance"]
    assert receipt["workflow_sha256"] == hashlib.sha256((folder / "workflow.json").read_bytes()).hexdigest()
    assert receipt["plan_sha256"] == hashlib.sha256((folder / "plan.json").read_bytes()).hexdigest()
    assert receipt["compiler_version"] == compiler.COMPILER_VERSION
    assert len(submitted) == 1 and state["submission_started"]
    prompts = [n["inputs"]["value"] for n in submitted[0].values() if n["class_type"] == "PrimitiveStringMultiline"]
    assert prompts and all("Take 1" in p for p in prompts)
    read_only = folder / "agent-edits" / "attempt-1" / "workflow.readonly.json"
    assert not os.access(read_only, os.W_OK) or os.geteuid() == 0


def test_invalid_first_plan_gets_one_correction(e2e):
    settings, mode, calls, _, _ = e2e
    mode.write_text("bad_first")
    with pytest.raises(RuntimeError, match="fake submit reached"):
        engine.run(settings, "A friendly welcome", lambda _: None, Event())
    assert len(calls()) == 2
    assert calls()[0]["validation_error"] == ""
    assert "section 'summary' is missing or empty" in calls()[1]["validation_error"]  # Lint reaches the agent.
    state = json.loads((only_run(settings) / "state.json").read_text())
    assert [a["status"] for a in state["workflow_edit_attempts"]] == ["invalid", "valid"]


def test_two_invalid_plans_fail_without_a_third_attempt(e2e):
    settings, mode, calls, _, submitted = e2e
    mode.write_text("always_bad")
    with pytest.raises(RuntimeError, match="rejected after one correction"):
        engine.run(settings, "A friendly welcome", lambda _: None, Event())
    folder = only_run(settings)
    with pytest.raises(RuntimeError, match="rejected after one correction"):
        engine.run(settings, "", lambda _: None, Event(), resume_folder=folder)
    assert len(calls()) == 2 and submitted == []


def test_read_only_template_in_used_references_becomes_the_linked_template(e2e):
    settings, mode, calls, _, submitted = e2e
    mode.write_text("readonly_ref")
    with pytest.raises(RuntimeError, match="fake submit reached"):
        engine.run(settings, "A friendly welcome", lambda _: None, Event())
    folder = only_run(settings)
    state = json.loads((folder / "state.json").read_text())
    template = str((Path(settings.template_folder) / INT8).resolve())
    read_only = str(folder / "agent-edits" / "attempt-1" / "workflow.readonly.json")
    assert len(calls()) == 1 and len(submitted) == 1  # No correction pass was spent.
    assert json.loads((folder / "plan.json").read_text())["used_references"] == [
        str(Path(settings.project_folder) / "ref.png"), template]
    assert [c["action"] for c in state["workflow_acceptance"]["used_references_normalized"]] == [
        "replaced", "dropped"]
    # The agent's own response is kept exactly as returned.
    raw = json.loads((folder / "agent-edits" / "attempt-1" / "plan.json").read_text())
    assert read_only in raw["used_references"]


def test_reference_outside_the_linked_folders_is_still_rejected(e2e):
    settings, mode, calls, _, submitted = e2e
    mode.write_text("outside_ref")
    with pytest.raises(RuntimeError, match="outside the linked folders"):
        engine.run(settings, "A friendly welcome", lambda _: None, Event())
    assert len(calls()) == 2 and submitted == []


@pytest.mark.parametrize("quantity", [1, 3])
def test_unfinished_plan_is_requested_again_without_spending_the_correction(e2e, quantity):
    settings, mode, calls, _, submitted = e2e
    mode.write_text("placeholder_bad_first")
    with pytest.raises(RuntimeError, match="fake submit reached"):
        engine.run(replace(settings, quantity=quantity), "A friendly welcome", lambda _: None, Event())
    state = json.loads((only_run(settings) / "state.json").read_text())
    assert [a["status"] for a in state["workflow_edit_attempts"]] == ["unfinished", "invalid", "valid"]
    assert len(calls()) == 3 and len(submitted) == 1
    assert "progress note" in calls()[1]["validation_error"]
    assert "section 'summary' is missing or empty" in calls()[2]["validation_error"]


def test_second_unfinished_plan_is_an_ordinary_rejection(e2e):
    settings, mode, calls, _, submitted = e2e
    mode.write_text("always_placeholder")
    with pytest.raises(RuntimeError, match="rejected after one correction: The response was a progress note"):
        engine.run(settings, "A friendly welcome", lambda _: None, Event())
    folder = only_run(settings)
    with pytest.raises(RuntimeError, match="rejected after one correction"):
        engine.run(settings, "", lambda _: None, Event(), resume_folder=folder)
    state = json.loads((folder / "state.json").read_text())
    assert [a["status"] for a in state["workflow_edit_attempts"]] == ["unfinished", "invalid", "invalid"]
    assert len(calls()) == 3 and submitted == []  # Bounded: never a fourth call.


def test_unfinished_then_two_bad_plans_fail_after_three_calls(e2e):
    settings, mode, calls, _, submitted = e2e
    mode.write_text("placeholder_always_bad")
    with pytest.raises(RuntimeError, match="rejected after one correction"):
        engine.run(settings, "A friendly welcome", lambda _: None, Event())
    assert len(calls()) == 3 and submitted == []


def test_qty_three_compiles_three_variations(e2e):
    settings, _, calls, _, submitted = e2e
    with pytest.raises(RuntimeError, match="fake submit reached"):
        engine.run(replace(settings, quantity=3), "Three takes", lambda _: None, Event())
    folder = only_run(settings)
    state = json.loads((folder / "state.json").read_text())
    assert len(calls()) == 1 and state["schema_version"] == 5
    children = [folder / child["folder"] for child in state["sequence"]["children"]]
    workflows = [json.loads((child / "workflow.json").read_text()) for child in children]
    kind = next(d["id"] for d in workflows[0]["definitions"]["subgraphs"])
    prompts, seeds = [], []
    for index, workflow in enumerate(workflows, 1):
        cards = [n for n in workflow["nodes"] if n["type"] == kind]
        names = [i["name"] for i in cards[0]["inputs"] if i.get("widget")]
        prompts.append(cards[0]["widgets_values"][names.index("prompt")])
        seeds.append(cards[0]["widgets_values"][names.index("seed")])
        assert f"Take {index}" in prompts[-1]
        child_state = json.loads((children[index - 1] / "state.json").read_text())
        assert child_state["schema_version"] == 5 and child_state["workflow_acceptance"]
    assert len(set(prompts)) == 3 and len(set(seeds)) == 3
    assert len(submitted) == 1


# 7. Salvage


@pytest.fixture
def salvage_env(tmp_path, monkeypatch):
    monkeypatch.setenv("SIMPLEGEN_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(agents, "generate_plan", lambda *a, **k: pytest.fail("no agent call"))
    # Reference images live on a network share; they cannot exist in CI.
    monkeypatch.setattr(compiler, "reference_problem", lambda path, roots: None)
    (tmp_path / "refs").mkdir()
    # Revisions pin their references (Phase 6); a local placeholder stands in for the SFTP originals.
    placeholder = tmp_path / "refs" / "ref.png"
    placeholder.write_bytes(PNG)
    monkeypatch.setattr(engine, "_inside", lambda path, roots: placeholder)

    def copy_run(name):
        target = tmp_path / "out" / name
        shutil.copytree(REAL / "runs" / name, target)
        state = json.loads((target / "state.json").read_text())
        state["settings"].update(project_folder=str(tmp_path / "refs"), output_folder=str(tmp_path / "out"))
        (target / "state.json").write_text(json.dumps(state))
        return target
    return copy_run


def hashes(folder):
    return {str(p.relative_to(folder)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(folder.rglob("*")) if p.is_file() and p.name != ".run.lock"}


@pytest.mark.parametrize(("run", "attempt", "schema_name"), [
    ("rejected_rewired_fixed_node_14", None, INT8), ("rejected_nonreciprocal_links", None, INT8),
    ("accepted_after_correction_definitions_changed", 1, BF16)])
def test_salvage_real_rejected_attempts(salvage_env, schemas, run, attempt, schema_name):
    source = salvage_env(run)
    before = hashes(source)
    result = repair.salvage(source, attempt=attempt, object_info=schemas[schema_name])
    assert result["status"] == "published"
    revision = Path(result["folder"])
    assert hashes(source) == before  # Source bytes unchanged.
    state = json.loads((revision / "state.json").read_text())
    assert state["schema_version"] == 5 and state["workflow_acceptance"]["provenance"] == "salvage, no agent pass"
    assert state["repaired_from"]["kind"] == "salvage" and state["repaired_from"]["run_id"]
    assert state["repaired_from"]["attempt"] == (attempt or 2)
    assert "batch_parent" not in state
    manifest = json.loads((revision / "references.json").read_text())  # Phase 6: salvage pins references.
    assert manifest["references"] and refcache.pinned(state["run_id"])
    plan = json.loads((revision / "plan.json").read_text())
    assert plan["timing_option"] == "segments-7" and len(plan["segments"]) == 7
    template = json.loads((revision / "template.original.json").read_text())
    workflow = json.loads((revision / "workflow.json").read_text())
    settings = Settings(**state["settings"])
    validate_edited_workflow(template, workflow, schemas[schema_name], megapixels=settings.megapixels,
                             duration=settings.duration, aspect_ratio=settings.aspect_ratio, steps=settings.steps,
                             run_id=state["run_id"], check_model_availability=False)
    # The revision's accepted work is reused by the schema 5 accepted path (no agent).
    seen = {}
    prepared, _, _, _ = editing.prepare_compiled(
        settings, revision, state, schemas[schema_name], Event(), lambda _: None, lambda *_: None,
        Path("/nonexistent"), {}, lambda prepared, plan: seen.update(ok=True) or ([], []))
    assert seen["ok"] and prepared["graph"]


def test_salvage_refusals(salvage_env, schemas):
    source = salvage_env("rejected_rewired_fixed_node_14")
    state = json.loads((source / "state.json").read_text())
    for changes, message in (({"schema_version": 5}, "schemas 2-4"), ({"batch_parent": ".."}, "batch"),
                             ({"prompt_id": "job"}, "already submitted"),
                             ({"workflow_edit_attempts": []}, "no rejected agent attempt")):
        (source / "state.json").write_text(json.dumps({**state, **changes}))
        with pytest.raises(repair.RepairRefused, match=message):
            repair.salvage(source, object_info=schemas[INT8])
    (source / "state.json").write_text(json.dumps({**state, "settings": {**state["settings"], "duration": 20.0}}))
    with pytest.raises(repair.RepairRefused, match="no legal timing layout"):
        repair.salvage(source, object_info=schemas[INT8])


# 8. Retarget


def test_retarget_bf16_plan_to_local_template_passes_preflight(salvage_env, schemas, tmp_path):
    source = salvage_env("rejected_rewired_fixed_node_14")
    salvaged = Path(repair.salvage(source, object_info=schemas[INT8])["folder"])
    blocked = run_saved_preflight(salvaged, object_info=schemas[INT8], resolve=tools)
    assert blocked["status"] == "fail"  # BF16 models are not on the local ComfyUI.
    local_templates = tmp_path / "local-templates"
    local_templates.mkdir()
    shutil.copy(BUNDLED / INT8, local_templates / INT8)
    result = repair.retarget(salvaged, local_templates / INT8, object_info=schemas[INT8], resolve=tools)
    assert result["status"] == "published", result
    revision = Path(result["folder"])
    assert run_saved_preflight(revision, object_info=schemas[INT8], resolve=tools)["status"] == "pass"
    state = json.loads((revision / "state.json").read_text())
    assert state["repaired_from"]["kind"] == "retarget" and state["template_path"] == str(local_templates / INT8)
    assert state["settings"]["template_file"] == str(local_templates / INT8)
    assert state["repaired_from"]["previous"]["kind"] == "salvage"  # Chain kept.
    old_plan = json.loads((salvaged / "plan.json").read_text())
    new_plan = json.loads((revision / "plan.json").read_text())
    assert new_plan["segments"] == old_plan["segments"]


def test_retarget_refuses_incompatible_workflows(salvage_env, schemas, tmp_path, templates):
    source = salvage_env("rejected_rewired_fixed_node_14")
    salvaged = Path(repair.salvage(source, object_info=schemas[INT8])["folder"])
    odd = copy.deepcopy(templates[INT8])
    for node in odd["nodes"]:
        names = [i["name"] for i in node.get("inputs", []) if i.get("widget")]
        if "prompt" in names:
            index = names.index("prompt")
            node["widgets_values"][index] = node["widgets_values"][index].replace("summary:", "synopsis:")
    (tmp_path / "odd.json").write_text(json.dumps(odd))
    with pytest.raises(repair.RepairRefused, match="different prompt sections"):
        repair.retarget(salvaged, tmp_path / "odd.json", object_info=schemas[INT8], resolve=tools)
    accepted_v4 = salvage_env("blocked_bf16_model_on_local")
    with pytest.raises(repair.RepairRefused, match="schema 5"):
        repair.retarget(accepted_v4, tmp_path / "odd.json", object_info=schemas[INT8], resolve=tools)


# 9. Schema 2-4 runs keep the old path


def test_schema4_run_resumes_through_the_old_path(tmp_path, monkeypatch, schemas):
    monkeypatch.setenv("SIMPLEGEN_HOME", str(tmp_path / "home"))
    run = tmp_path / "out" / "blocked"
    shutil.copytree(REAL / "runs" / "blocked_bf16_model_on_local", run)
    state = json.loads((run / "state.json").read_text())
    (tmp_path / "refs").mkdir()
    reference = tmp_path / "refs" / "ref.png"
    reference.write_bytes(PNG)
    shutil.copytree(BUNDLED, tmp_path / "templates")
    state["settings"].update(project_folder=str(tmp_path / "refs"), output_folder=str(tmp_path / "out"),
                             template_folder=str(tmp_path / "templates"),
                             template_file=str(tmp_path / "templates" / BF16))
    (run / "state.json").write_text(json.dumps(state))
    monkeypatch.setattr(agents, "generate_plan", lambda *a, **k: pytest.fail("no agent call"))
    monkeypatch.setattr(agents, "executable", lambda name: tools(name))
    monkeypatch.setattr(editing, "prepare_compiled", lambda *a, **k: pytest.fail("schema 4 uses prepare_edit"))
    monkeypatch.setattr(engine, "_inside", lambda path, roots: reference)
    used = []
    real = editing.prepare_edit
    monkeypatch.setattr(editing, "prepare_edit", lambda *a, **k: used.append(True) or real(*a, **k))
    uploaded = []

    class Comfy:
        def __init__(self, settings, client=None):
            pass

        def object_info(self):
            info = copy.deepcopy(schemas[BF16])
            info["LoadImage"]["input"]["required"]["image"][0].extend(uploaded)
            return info

        def queue(self):
            return {"queue_running": [], "queue_pending": []}

        def upload(self, path, run_id, cancel):
            uploaded.append(f"{run_id}/{path.name}")
            return uploaded[-1]

        def submit(self, graph, run_id, client_id):
            raise Submitted("fake submit reached")

        def find_submission(self, run_id):
            return None

        def close(self):
            pass
    monkeypatch.setattr(engine, "Comfy", Comfy)
    with pytest.raises(RuntimeError, match="fake submit reached"):
        engine.run(Settings(), "", lambda _: None, Event(), resume_folder=run)
    assert used == [True]
    assert json.loads((run / "state.json").read_text())["schema_version"] == 4


# Qt offscreen: Rebuild and retarget actions


def test_ui_offers_rebuild_and_retarget(salvage_env, schemas, tmp_path):
    import time
    from functools import partial

    from PySide6.QtWidgets import QApplication

    from simplegen.repair_ui import RepairDialog
    from simplegen.ui import SavedRunDialog
    app = QApplication.instance() or QApplication([])
    source = salvage_env("rejected_rewired_fixed_node_14")
    dialog = SavedRunDialog(json.loads((source / "state.json").read_text()), source)
    assert not dialog.salvage_button.isHidden()
    dialog.deleteLater()
    salvaged = Path(repair.salvage(source, object_info=schemas[INT8])["folder"])
    dialog = SavedRunDialog(json.loads((salvaged / "state.json").read_text()), salvaged)
    assert dialog.salvage_button.isHidden()  # Already schema 5.
    dialog.deleteLater()
    target = tmp_path / "local" / INT8
    target.parent.mkdir()
    shutil.copy(BUNDLED / INT8, target)
    repair_dialog = RepairDialog(salvaged, inspect=partial(repair.inspect, object_info=schemas[INT8]),
                                 retarget=partial(repair.retarget, object_info=schemas[INT8], resolve=tools),
                                 choose_file=lambda: str(target))
    end = time.time() + 10
    while (repair_dialog.info is None or repair_dialog.worker is not None) and time.time() < end:
        app.processEvents()
        time.sleep(0.01)
    assert not repair_dialog.retarget_button.isHidden() and repair_dialog.retarget_button.isEnabled()
    repair_dialog.retarget_button.click()
    end = time.time() + 10
    while (repair_dialog.result is None or repair_dialog.worker is not None) and time.time() < end:
        app.processEvents()
        time.sleep(0.01)
    assert repair_dialog.result["status"] == "published"
    assert repair_dialog.run_button.isEnabled() and repair_dialog.schedule_button.isEnabled()
    repair_dialog.deleteLater()
    app.processEvents()


# Review follow-up: every link form, preflight compiler check, speech thresholds, UI chaining


def _link_forms(template):
    five = copy.deepcopy(template)
    five["links"] = [link[:5] for link in five["links"]]
    objects = copy.deepcopy(template)
    objects["links"] = [dict(zip(["id", "origin_id", "origin_slot", "target_id", "target_slot", "type"], link))
                        for link in objects["links"]]
    return {"six": template, "five": five, "object": objects}


@pytest.mark.parametrize("form", ["six", "five", "object"])
def test_compiler_supports_every_link_form_the_validator_accepts(schemas, templates, form):
    template = _link_forms(templates[INT8])[form]
    for duration in (5, 20, 45, 90):
        settings = settings_for(duration)
        info = analyze(template, schemas[INT8], settings, "run1")
        for option in info["options"]:
            workflow = compile_workflow(template, good_plan(info, option=option), settings, "run1",
                                        object_info=schemas[INT8])
            kinds = {type(link).__name__ + str(len(link)) for link in workflow["links"]}
            assert kinds == {type(template["links"][0]).__name__ + str(len(template["links"][0]))}
            validate_edited_workflow(template, workflow, schemas[INT8], megapixels=settings.megapixels,
                                     duration=settings.duration, run_id="run1", aspect_ratio=settings.aspect_ratio,
                                     steps=settings.steps, check_model_availability=False)
        assert compiler.support_problem(template, schemas[INT8], settings) is None


def test_compiler_supports_section_less_prompts_and_minimal_nodes(_synthetic_info, _synthetic_workflow):  # noqa: F811
    info, workflow = _synthetic_info, _synthetic_workflow
    # Synthetic template: 5-element links, no report outputs or positions, free-text prompt.
    settings = Settings(duration=5.0)
    analysis = analyze(workflow, info, settings, "run1")
    assert analysis["raw_prompt"] and analysis["sections"] == ["prompt"] and analysis["previews"] == {}
    plan = {"script": "s", "used_references": [], "timing_option": analysis["options"][0]["id"], "references": {},
            "segments": [{"sections": {"prompt": "  A silent blue frame.  "}}]}
    compiled = compile_workflow(workflow, plan, settings, "run1", object_info=info)
    assert all(len(link) == 5 for link in compiled["links"])
    prepared = validate_edited_workflow(workflow, compiled, info, megapixels=settings.megapixels,
                                        duration=settings.duration, run_id="run1", check_model_availability=False)
    assert prepared["prompt_slots"][0]["text"] == "A silent blue frame."
    assert compiler.support_problem(workflow, info, settings) is None


def test_preflight_fails_a_template_the_compiler_cannot_build(tmp_path, schemas, templates):
    from simplegen.preflight import run_preflight
    broken = copy.deepcopy(templates[INT8])
    cards = [n for n in broken["nodes"] if n["type"] == broken["definitions"]["subgraphs"][0]["id"]]
    for card in cards:  # Unsupported for the compiler: the chain input is not named "previous".
        for inp in card["inputs"]:
            if inp["name"] == "previous":
                inp["name"] = "prior"
    path = tmp_path / "broken.json"
    path.write_text(json.dumps(broken))
    (tmp_path / "refs").mkdir()
    settings = settings_for(20, project_folder=str(tmp_path / "refs"), template_folder=str(tmp_path),
                            output_folder=str(tmp_path / "out"))
    assert compiler.support_problem(broken, schemas[INT8], settings)
    result = run_preflight(settings, template_path=path, object_info=schemas[INT8], resolve=tools)
    check = next(c for c in result["checks"] if c["id"] == "template_structure")
    assert result["status"] == "fail" and "workflow compiler" in check["message"]
    legacy = run_preflight(settings, template_path=path, object_info=schemas[INT8], resolve=tools, compiled=False)
    assert next(c for c in legacy["checks"] if c["id"] == "template_structure")["message"] != check["message"]
    good = tmp_path / INT8
    shutil.copy(BUNDLED / INT8, good)
    passing = run_preflight(settings, template_path=good, object_info=schemas[INT8], resolve=tools)
    assert next(c for c in passing["checks"] if c["id"] == "template_structure")["status"] == "pass"


def test_saved_preflight_checks_the_compiler_only_for_schema5(tmp_path, schemas, templates, monkeypatch):
    from simplegen import preflight
    seen = []
    monkeypatch.setattr(preflight, "run_preflight", lambda settings, **kwargs: seen.append(kwargs["compiled"]))
    for version in (4, 5):
        folder = tmp_path / f"v{version}"
        folder.mkdir()
        (folder / "state.json").write_text(json.dumps({"schema_version": version, "settings": {}}))
        preflight.run_saved_preflight(folder)
    assert seen == [False, True]


def test_speech_budget_warning_and_error_thresholds(lint_env):
    info, refs, _, assets, _ = lint_env
    option = next(o for o in info["options"] if o["recommended"])
    budget = option["segments"][0]["speech_words"]

    def plan_with(words):
        plan = good_plan(info, refs=refs, dialogue="")
        first = plan["segments"][0]["sections"]
        first[info["sections"][-2]] = f"<Subject 1> (S1) says: <d>[English] {' '.join(['word'] * words)}</d>"
        return plan
    soft = int(budget * compiler.SPEECH_WARN_RATIO + compiler.SPEECH_TOLERANCE_WORDS) + 1
    hard = int(budget * compiler.SPEECH_ERROR_RATIO + compiler.SPEECH_TOLERANCE_WORDS) + 1
    for words, errors, warned in ((budget, 0, 0), (soft, 0, 1), (hard, 1, 0)):
        warnings = []
        result = lint(plan_with(words), info, option, concept="", roots=[assets.resolve()], quantity=1,
                      warnings=warnings)
        assert (len(result), len(warnings)) == (errors, warned), words
    warnings = []
    assert lint(plan_with(hard), info, option, concept="", roots=[assets.resolve()], quantity=1,
                warnings=warnings, budget_errors=False) == [] and len(warnings) == 1


def test_retarget_records_old_and_new_template(salvage_env, schemas, tmp_path, monkeypatch):
    source = salvage_env("rejected_rewired_fixed_node_14")
    salvaged = Path(repair.salvage(source, object_info=schemas[INT8])["folder"])
    target = tmp_path / "local" / INT8
    target.parent.mkdir()
    shutil.copy(BUNDLED / INT8, target)
    # Retarget reuses the source's verified cache: the originals are not read again.
    monkeypatch.setattr(engine, "_inside", lambda path, roots: pytest.fail("cached references are reused"))
    revision = Path(repair.retarget(salvaged, target, object_info=schemas[INT8], resolve=tools)["folder"])
    new_state = json.loads((revision / "state.json").read_text())
    assert refcache.pinned(new_state["run_id"]) and refcache.complete(revision)
    assert ({e["sha256"] for e in json.loads((revision / "references.json").read_text())["references"]}
            == {e["sha256"] for e in json.loads((salvaged / "references.json").read_text())["references"]})
    origin = new_state["repaired_from"]
    old_state = json.loads((salvaged / "state.json").read_text())
    assert origin["old_template_path"] == old_state["template_path"]
    assert origin["new_template_path"] == str(target)
    assert origin["old_template_sha256"] == hashlib.sha256((salvaged / "template.original.json").read_bytes()).hexdigest()
    assert origin["new_template_sha256"] == hashlib.sha256(target.read_bytes()).hexdigest()


@pytest.mark.usefixtures("_no_desktop_audio")
def test_rebuild_that_fails_preflight_opens_repair(salvage_env, schemas, monkeypatch):
    import time

    from PySide6.QtWidgets import QApplication

    from simplegen import ui
    app = QApplication.instance() or QApplication([])
    source = salvage_env("rejected_rewired_fixed_node_14")
    result = repair.salvage(source, object_info=schemas[INT8])
    window = ui.MainWindow(Settings(output_folder=str(source.parent)))
    failing = {"status": "fail", "checks": [{"id": "target_compatibility", "status": "fail",
                                             "message": "unet_name unavailable", "details": {}}]}
    monkeypatch.setattr(ui, "run_preflight", lambda settings, template_path=None, **_: failing)
    opened, chosen = [], []
    monkeypatch.setattr(window, "open_repair", lambda folder, schedule_id=None: opened.append(folder))
    monkeypatch.setattr(window, "open_saved_run_choice", lambda folder: chosen.append(folder))
    window.salvage_done(result)
    end = time.time() + 10
    while window.preflight_worker is not None and time.time() < end:
        app.processEvents()
        time.sleep(0.01)
    assert opened == [Path(result["folder"])] and chosen == []
    passing = {"status": "pass", "checks": []}
    monkeypatch.setattr(ui, "run_preflight", lambda settings, template_path=None, **_: passing)
    window.salvage_done(result)
    end = time.time() + 10
    while window.preflight_worker is not None and time.time() < end:
        app.processEvents()
        time.sleep(0.01)
    assert chosen == [Path(result["folder"])] and len(opened) == 1
    window.close()
    window.deleteLater()
    app.processEvents()


# Quoted concept lines are required speech only in a spoken context (found while comparing local agents)


@pytest.mark.parametrize("concept", [
    'Do a sight gag: have it "knock on the door" and look puzzled.',
    '*One shot. Vertical. Maximum “I filmed this on my lunch break.”*',
    'JORDAN REYES, grey-on-grey, “I woke up ready” haircut. She looks into the lens.',
    '**30-second trim:** cut “I was late on Monday,” cut “I can feel it in my bones,” cut “I’ll be here.”',
    'The mug reads "WORLD’S OKAYEST MANAGER" in bold.',
])
def test_quoted_phrases_outside_speech_are_not_required(concept):
    assert compiler.quoted_lines(concept) == []


@pytest.mark.parametrize("concept, expected", [
    ('ALEX (CONT’D)\n\nAnd the others? Same plan.\n“Alex why didn’t you call back.”\nBecause I didn’t.',
     ["Alex why didn’t you call back."]),
    ('Tomas says "why did the goat cross" and waits.', ["why did the goat cross"]),
    ('MARA: "I am not laughing at that"', ["I am not laughing at that"]),
    ('Say "first line goes here" and then say “second line goes here”',
     ["first line goes here", "second line goes here"]),
])
def test_quoted_lines_in_spoken_context_are_required(concept, expected):
    assert compiler.quoted_lines(concept) == expected


def test_real_concepts_require_only_their_spoken_line():
    state = json.loads((REAL / "runs" / "accepted_after_correction_definitions_changed" / "state.json").read_text())
    assert compiler.quoted_lines(state["concept"]) == ["Rorevuvi why katu’t you degari up."]


def test_required_line_matches_despite_trailing_punctuation():
    assert compiler.quote_needle("I was late on Monday,") == compiler.quote_needle("I was late on Monday.")
    assert compiler.quote_needle("I was late on Monday,") in compiler._normal("so. I was late on Monday. and then")
