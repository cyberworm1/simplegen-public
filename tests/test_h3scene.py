"""Offline integration, golden graph/message and overflow revision tests."""
import copy
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from threading import Event

import pytest

from simplegen import agents, editing, engine, preflight, repair
from simplegen import h3scene_family as scene
from simplegen.h3scene import run
from simplegen.h3scene.__main__ import template
from simplegen.h3scene_review import OverflowReview, decide
from simplegen.settings import Settings, atomic_json

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).parent / "fixtures/real/h3scene"


def old_saved_run(context, monkeypatch, *, accepted=False):
    """Load an old-format fixture, binding its receipts to temporary run files."""
    settings, _, schema = context
    source = Path(settings.template_folder) / "H3_Lab_Chain_LOCAL_INT8.json"
    source.write_bytes((FIXTURES / "legacy/template.json").read_bytes())
    settings = replace(settings, template_file=source.name)
    folder, state = engine.create_run(settings, "PERSON: Ready.")

    def fake_agent(settings, workspace, *args):
        atomic_json(workspace / "plan.json", candidate() if accepted else candidate((15, 15, 15, 5)))

    monkeypatch.setattr(agents, "generate_plan", fake_agent)
    if accepted:
        prepare(settings, folder, state, schema, lambda p, v: ([], v["used_references"]))
    else:
        with pytest.raises(OverflowReview):
            prepare(settings, folder, state, schema, lambda p, v: ([], []))
    legacy = json.loads((FIXTURES / "legacy/state.json").read_text())
    if accepted:
        receipt = state.pop("workflow_acceptance")
        receipt.pop("scene_chain")
        receipt.pop("h3scene_format")
        state["workflow_acceptance"] = {**receipt, **legacy["workflow_acceptance"]}
    else:
        review = state.pop("h3scene_review")
        state["h3lab_review"] = {**legacy["h3lab_review"], **review}
    state["summary"] = legacy["summary"]
    # Deliberately bypass the current writer to represent bytes saved by the old app.
    (folder / "state.json").write_text(json.dumps(state, indent=2) + "\n")
    monkeypatch.setattr(agents, "generate_plan", lambda *a: pytest.fail("old saved plan called agent"))
    return settings, folder, state, schema


def test_old_template_preserves_graph_profile_and_preflight(context):
    settings, native, schema = context
    old = json.loads((FIXTURES / "legacy/template.json").read_text())
    assert scene.is_scene_chain(old) and scene.profile(old) == scene.profile(native)
    assert old["nodes"] == native["nodes"] and old["links"] == native["links"]
    source = Path(settings.template_folder) / "H3_Lab_Chain_LOCAL_INT8.json"
    source.write_bytes((FIXTURES / "legacy/template.json").read_bytes())
    settings = replace(settings, template_file=source.name)
    checked = preflight.run_preflight(settings, object_info=schema, resolve=lambda n: n)
    assert checked["status"] == "pass"
    assert scene.compile_graph(old, candidate(), settings, "fixture") == scene.compile_graph(
        native, candidate(), settings, "fixture")


@pytest.mark.parametrize("refmod", [False, True])
def test_templates_saved_with_earlier_profile_labels_still_load(refmod):
    from simplegen.h3scene.__main__ import template_bytes
    native = json.loads(template_bytes(refmod=refmod))
    saved = copy.deepcopy(native)
    profile = saved["extra"]["h3scene"]["profile"]
    profile["prompt"] = "production_v3.md plus the RefMod section" if refmod else "production_v3.md"
    profile["plan_contract"] = "DECISIONS 13 plus optional warnings and overflow" + (" plus refmods" if refmod else "")
    assert scene.profile(saved) == scene.profile(native)
    profile["fps"] = 30
    with pytest.raises(ValueError, match="approved fixed values"):
        scene.profile(saved)


@pytest.mark.parametrize("action", ["approve", "edit", "deny"])
def test_old_review_opens_and_creates_new_named_revision(context, monkeypatch, action):
    from PySide6.QtWidgets import QApplication

    from simplegen.ui import SavedRunDialog

    app = QApplication.instance() or QApplication([])
    settings, folder, state, schema = old_saved_run(context, monkeypatch)
    original = (folder / "state.json").read_bytes()
    dialog = SavedRunDialog(state, folder)
    assert dialog.resume_button.text() == "Approve" and dialog.scene_pending
    assert "H3 Scene Chain" in dialog.explanation.text() and "50" in dialog.explanation.text()
    dialog.deleteLater()
    app.processEvents()
    revision, child = decide(folder, action)
    assert child["h3scene_decision"]["action"] == action
    assert "h3lab" not in (revision / "state.json").read_text()
    assert (folder / "state.json").read_bytes() == original
    assert (revision / "template.original.json").read_bytes() == (folder / "template.original.json").read_bytes()
    if action == "approve":
        prepared, plan, _, _ = prepare(settings, revision, child, schema, lambda p, v: ([], []))
        assert prepared["final_node"] and sum(s["seconds"] for s in plan["segments"]) == 50
        assert child["workflow_acceptance"]["scene_chain"] == scene.KIND
    elif action == "deny":
        with pytest.raises(ValueError, match="denied"):
            prepare(settings, revision, child, schema, lambda *a: pytest.fail("denied upload"))
    else:
        assert "h3scene_review" not in child and not (revision / "agent-edits").exists()


def test_old_approved_decision_resumes_without_agent(context, monkeypatch):
    settings, folder, state, schema = old_saved_run(context, monkeypatch)
    legacy = json.loads((FIXTURES / "legacy/state.json").read_text())
    state["h3lab_decision"] = {**legacy["h3lab_decision"],
                              "plan_sha256": editing.digest(folder / "plan.json")}
    (folder / "state.json").write_text(json.dumps(state))
    prepared, plan, _, _ = prepare(settings, folder, state, schema, lambda p, v: ([], []))
    assert prepared["final_node"] and sum(s["seconds"] for s in plan["segments"]) == 50
    saved = json.loads((folder / "state.json").read_text())
    assert saved["h3scene_decision"]["action"] == "approve" and "h3lab_decision" not in saved


def test_old_pending_review_still_blocks_compilation(context, monkeypatch):
    settings, folder, state, schema = old_saved_run(context, monkeypatch)
    monkeypatch.setattr(scene, "compile_graph", lambda *a, **k: pytest.fail("pending review compiled"))
    with pytest.raises(OverflowReview):
        prepare(settings, folder, state, schema, lambda *a: pytest.fail("pending review uploaded"))
    saved = json.loads((folder / "state.json").read_text())
    assert "h3scene_review" in saved and "h3lab_review" not in saved


def test_old_denied_decision_blocks_before_service_setup(context, monkeypatch):
    from PySide6.QtWidgets import QApplication

    from simplegen.ui import SavedRunDialog

    app = QApplication.instance() or QApplication([])
    settings, folder, state, _schema = old_saved_run(context, monkeypatch)
    state["h3lab_decision"] = {"action": "deny"}
    (folder / "state.json").write_text(json.dumps(state))
    dialog = SavedRunDialog(state, folder)
    assert dialog.resume_button.isHidden() and not dialog.scene_pending
    dialog.deleteLater()
    app.processEvents()
    monkeypatch.setattr(engine, "Comfy", lambda *a: pytest.fail("denied run contacted ComfyUI"))
    with pytest.raises(ValueError, match="denied"):
        engine.run(settings, "ignored", lambda _: None, Event(), resume_folder=folder)


def test_old_accepted_run_resumes_through_engine(context, monkeypatch):
    settings, folder, _state, schema = old_saved_run(context, monkeypatch, accepted=True)
    graph_bytes = (folder / "workflow.json").read_bytes()
    plan_bytes = (folder / "plan.json").read_bytes()
    submitted = []

    class FakeComfy:
        def __init__(self, settings):
            pass
        def object_info(self):
            return copy.deepcopy(schema)
        def queue(self):
            return {"queue_running": [], "queue_pending": []}
        def upload(self, path, run_id, cancel):
            return "uploaded/ref.png"
        def submit(self, graph, *args):
            submitted.append(graph)
            raise RuntimeError("old accepted submission reached")
        def close(self):
            pass

    monkeypatch.setattr(engine, "Comfy", FakeComfy)
    with pytest.raises(RuntimeError, match="old accepted submission reached"):
        engine.run(settings, "ignored", lambda _: None, Event(), resume_folder=folder)
    assert len(submitted) == 1
    saved = json.loads((folder / "state.json").read_text())
    assert saved["workflow_acceptance"]["scene_chain"] == scene.KIND
    assert "lab_chain" not in saved["workflow_acceptance"] and "h3lab_format" not in saved["workflow_acceptance"]
    assert "lab" not in saved["summary"].lower()
    assert (folder / "workflow.json").read_bytes() == graph_bytes
    assert (folder / "plan.json").read_bytes() == plan_bytes


@pytest.mark.parametrize("provider", ["codex", "claude", "grok"])
def test_old_creative_brief_retains_tools_and_message(context, monkeypatch, provider):
    from simplegen.h3scene_compat import normalize_creative

    settings, native, _ = context
    settings = replace(settings, provider=provider, model_tier="Custom", model="", reasoning="auto")
    current = scene.brief(native, settings, "fixture", Path("unused"), concept="PERSON: Ready.")
    old = dict(current)
    old["lab_chain"] = old.pop("scene_chain")
    old["lab_messages"] = old.pop("scene_messages")
    old["skill"] = "h3-lab-production"
    assert normalize_creative(old) == current
    monkeypatch.setattr(agents, "executable", lambda n: n)
    assert agents.plan_schema(creative=old) == agents.plan_schema(creative=current)
    assert agents.build_command(settings, Path("/tmp/unused"), [], creative=old) == agents.build_command(
        settings, Path("/tmp/unused"), [], creative=current)


@pytest.fixture
def context(tmp_path):
    assets = tmp_path / "assets"
    assets.mkdir()
    (assets / "INDEX.md").write_text("Index")
    # Valid PNG, probed locally by the vendored checker.
    import struct
    import zlib
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
    (assets / "ref.png").write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 32, 32, 8, 2, 0, 0, 0))
                                   + chunk(b"IDAT", zlib.compress((b"\0" + b"\xff\0\0" * 32) * 32)) + chunk(b"IEND", b""))
    path = template(tmp_path / "templates")
    settings = Settings(project_folder=str(assets), template_folder=str(path.parent), template_file=path.name,
                        output_folder=str(tmp_path / "output"), duration=30, megapixels=.52,
                        aspect_ratio="16:9", steps=30)
    schema = json.loads((ROOT / "tests/fixtures/real/schemas/local_object_info.json").read_text())
    return settings, json.loads(path.read_text()), schema


def load_record(folder):
    record = json.loads((folder / "run.json").read_text())
    uploads = folder / "uploads.json"
    if uploads.is_file():
        record["uploads"] = json.loads(uploads.read_text())
    return record


def candidate(seconds=(15, 15), **fields):
    sections = {name: "Fixture text." for name in run.SECTIONS}
    sections["subject_definitions"] = "<Subject 1> from <Picture 1>."
    sections["detailed_description"] = " ".join(["A person walks through the room."] * 12) + " <d>[English] Ready.</d>"
    segments = [{"join": "first" if i == 0 else "continue", "seconds": s, "sections": copy.deepcopy(sections)}
                for i, s in enumerate(seconds)]
    for segment in segments[1:]:
        segment["sections"]["detailed_description"] = " ".join(["Continuing the shot."] * 20)
    return {"concept_id": "fixture", "title": "Fixture", "script": "Owner", "input_mode": "script",
            "pictures": [{"path": "ref.png", "subject": "Person"}], "segments": segments, **fields}


def normalized_golden(graph, plan, record, settings, *, uploaded=False, normalize_seeds=False):
    """Resolve each filename independently to its plan picture; preserve node assignment."""
    filenames = {}
    for index, picture in enumerate(plan["pictures"]):
        path = picture["path"] if isinstance(picture, dict) else picture
        if uploaded and "settings" in record:
            # The name ComfyUI received when the golden run uploaded this picture.
            filename = record["uploads"][path]
        elif uploaded:
            filename = "wl_" + Path(path).name
        elif "settings" in record:
            root = Path(settings.project_folder).resolve()
            filename = str(run.picture_path(path, root))
        else:
            filename = path
        assert filename not in filenames, "Golden pictures must have distinct filenames"
        filenames[filename] = index
    graph = copy.deepcopy(graph)
    for node in graph.values():
        if node["class_type"] == "LoadImage":
            node["inputs"]["image"] = f"PICTURE-{filenames[node['inputs']['image']]}"
        elif node["class_type"] == "SaveVideo":
            node["inputs"]["filename_prefix"] = "PREFIX"
        elif normalize_seeds and node["class_type"] == "RandomNoise":
            node["inputs"]["noise_seed"] = "SEED"
    return graph


@pytest.mark.parametrize("name", ["degu_nanomek_zadusu_binepu-efc99950f0", "gozodur-0p5mp", "viv_rorevuvi_dun-74cc8e36de"])
def test_golden_graphs(name, context, monkeypatch):
    settings, native, _ = context
    folder = FIXTURES / name
    plan = json.loads((folder / "plan.json").read_text())
    record = load_record(folder)
    expected = json.loads((folder / "workflow.api.json").read_text())
    if "settings" in record:
        monkeypatch.setattr(scene, "render_settings", lambda *args: record["settings"])
        actual = scene.compile_graph(native, plan, settings, record["run_id"])
    else:
        assert [plan["width"], plan["height"]] == record["size"]
        actual = scene.compile_spec(plan)
    scene.verify(actual, native)
    assert normalized_golden(actual, plan, record, settings) == normalized_golden(
        expected, plan, record, settings, uploaded=True)


@pytest.mark.parametrize(("name", "mp", "aspect", "duration", "size"), [
    ("degu_nanomek_zadusu_binepu-efc99950f0", .52, "16:9", 60, (960, 544)),
    ("viv_rorevuvi_dun-74cc8e36de", .36, "9:16", 30, (448, 800)),
])
def test_golden_graphs_real_settings(name, mp, aspect, duration, size, context):
    settings, native, _ = context
    settings = replace(settings, megapixels=mp, aspect_ratio=aspect, steps=30, duration=duration)
    folder = FIXTURES / name
    plan = json.loads((folder / "plan.json").read_text())
    record = load_record(folder)
    expected = json.loads((folder / "workflow.api.json").read_text())
    render = scene.render_settings(native, settings, record["run_id"])
    assert (render["width"], render["height"]) == size
    assert settings.duration == plan["requested_seconds"]
    actual = scene.compile_graph(native, plan, settings, record["run_id"])
    scene.verify(actual, native)
    assert [n["inputs"]["noise_seed"] for n in actual.values() if n["class_type"] == "RandomNoise"] == [
        render["seed_base"] + i for i in range(len(plan["segments"]))]
    # Run IDs use simpleGen seeds; only that input differs from the scene's seed base.
    assert normalized_golden(actual, plan, record, settings, normalize_seeds=True) == normalized_golden(
        expected, plan, record, settings, uploaded=True, normalize_seeds=True)


def test_golden_normalization_detects_swapped_pictures(context):
    settings, _, _ = context
    folder = FIXTURES / "degu_nanomek_zadusu_binepu-efc99950f0"
    plan = json.loads((folder / "plan.json").read_text())
    record = load_record(folder)
    graph = json.loads((folder / "workflow.api.json").read_text())
    expected = normalized_golden(graph, plan, record, settings, uploaded=True)
    images = [n["inputs"] for n in graph.values() if n["class_type"] == "LoadImage"]
    images[0]["image"], images[1]["image"] = images[1]["image"], images[0]["image"]
    assert normalized_golden(graph, plan, record, settings, uploaded=True) != expected


def test_scene_messages_hold_skill_owner_text_and_cards_only(context):
    settings, native, _ = context
    assets = Path(settings.project_folder)
    (assets / "INDEX.md").write_text("Index text\n")
    (assets / "characters/a").mkdir(parents=True)
    (assets / "characters/a/card.md").write_text("A card\n")
    (assets / "places/p").mkdir(parents=True)
    (assets / "places/p/card.md").write_text("P card\n")
    (assets / "bible.md").write_text("FORBIDDEN")
    owner = "MARA: Ready.\n"
    actual = scene.brief(native, settings, "fixture", Path("unused"), concept=owner)["scene_messages"]
    master = (ROOT / "src/simplegen/skills/h3-scene-production/SKILL.md").read_text()
    assert actual == [
        {"role": "system", "content": master},
        {"role": "user", "content": "# Expectations\nConcept id: fixture\nRequested seconds: 30\n"
                                    "\n# Owner text (verbatim)\nMARA: Ready.\n\n\n# Library\n"
                                    "=== INDEX.md ===\nIndex text\n\n\n=== characters/a/card.md ===\nA card\n"
                                    "\n\n=== places/p/card.md ===\nP card\n"}]
    assert "FORBIDDEN" not in str(actual)


@pytest.mark.parametrize(("change", "message"), [({"quantity": 2}, "Qty 1"), ({"cloud": True}, "MultiGPU")])
def test_preflight_scope(context, change, message):
    settings, _, schema = context
    result = preflight.run_preflight(replace(settings, **change), object_info=schema, resolve=lambda n: n)
    check = next(c for c in result["checks"] if c["id"] == "h3scene_scope")
    assert check["status"] == "fail" and message in check["message"]


def test_missing_index_and_repair(context):
    settings, native, schema = context
    Path(settings.project_folder, "INDEX.md").unlink()
    result = preflight.run_preflight(settings, object_info=schema, resolve=lambda n: n)
    assert any(c["status"] == "fail" and "INDEX.md" in c["message"] for c in result["checks"])
    folder, state = engine.create_run(settings, "Owner")
    atomic_json(folder / "template.original.json", native)
    assert "Repair" in repair.refusal(folder, state) and "Edit as new run" in repair.refusal(folder, state)
    with pytest.raises(repair.RepairRefused, match="Repair.*Edit as new run"):
        repair.inspect(folder, object_info=schema)


def test_cached_references_skip_missing_project_checks(context, tmp_path):
    settings, _, schema = context
    settings = replace(settings, project_folder=str(tmp_path / "missing-project"))
    result = preflight.run_preflight(settings, object_info=schema, resolve=lambda n: n,
                                     cached_references=True, agent_needed=False)
    for check_id in ("linked_folders", "h3scene_scope", "template_structure"):
        assert next(c for c in result["checks"] if c["id"] == check_id)["status"] == "pass"


@pytest.mark.parametrize("snapshot", ["{broken", '{"extra": null}', "[]"])
def test_repair_refusal_tolerates_corrupt_snapshot(context, snapshot):
    settings, _, _ = context
    folder, state = engine.create_run(settings, "Owner")
    (folder / "template.original.json").write_text(snapshot)
    assert "no accepted workflow" in repair.refusal(folder, state)
    state["workflow_acceptance"] = {"valid": True}
    assert repair.refusal(folder, state) is None
    state["workflow_acceptance"]["scene_chain"] = scene.KIND
    assert "Repair" in repair.refusal(folder, state)


@pytest.mark.parametrize(("seconds", "mp", "aspect", "verified", "ram"), [
    (60, .52, "16:9", True, False), (60.01, .52, "16:9", False, False),
    (30, .72, "16:9", True, False), (30.01, .72, "16:9", False, False),
    (45, .72, "16:9", False, True), (44.99, .72, "16:9", False, False),
    (30, 1, "16:9", False, False),
])
def test_envelope_boundaries(context, seconds, mp, aspect, verified, ram):
    settings, native, schema = context
    settings = replace(settings, duration=seconds, megapixels=mp, aspect_ratio=aspect)
    result = scene.envelope(native, settings)
    assert result["verified"] == verified
    assert ("Host-RAM" in result["message"]) == ram
    assert verified or "nearest tested point" in result["message"]
    checks = preflight.run_preflight(settings, object_info=schema, resolve=lambda n: n)["checks"]
    assert next(c for c in checks if c["id"] == "h3scene_envelope")["status"] == "pass"


def prepare(settings, folder, state, info, references):
    def persist(status=None):
        if status:
            state["status"] = status
        atomic_json(folder / "state.json", state)
    return editing.prepare_compiled(settings, folder, state, info, Event(), lambda _: None, persist,
                                    Path(settings.template_folder) / settings.template_file, {}, references)


@pytest.mark.parametrize("action", ["approve", "edit", "deny"])
def test_overflow_revisions_stop_before_compile_upload(context, monkeypatch, action):
    settings, _, schema = context
    folder, state = engine.create_run(settings, "PERSON: Ready.")
    plan = candidate((15, 15, 15, 5))
    def fake_agent(settings, workspace, *args):
        atomic_json(workspace / "plan.json", plan)
    monkeypatch.setattr(agents, "generate_plan", fake_agent)
    original_compile = scene.compile_graph
    monkeypatch.setattr(scene, "compile_graph", lambda *a, **kw: pytest.fail("review plan compiled"))
    with pytest.raises(OverflowReview):
        prepare(settings, folder, state, schema, lambda *a: pytest.fail("review plan reached uploads"))
    original_bytes = (folder / "state.json").read_bytes()
    assert not (folder / "workflow.json").exists()
    revision, child = decide(folder, action)
    assert revision != folder and child["forked_from"]["run_id"] == state["run_id"]
    assert child["h3scene_decision"]["action"] == action
    assert (folder / "state.json").read_bytes() == original_bytes
    if action == "approve":
        monkeypatch.setattr(scene, "compile_graph", original_compile)
        monkeypatch.setattr(agents, "generate_plan", lambda *a: pytest.fail("approved plan called agent"))
        prepared, accepted_plan, uploads, _ = prepare(settings, revision, child, schema, lambda p, v: ([], v["used_references"]))
        assert prepared["final_node"] and accepted_plan["overflow"]["needs_review"] and not uploads
        assert child["workflow_acceptance"]["scene_chain"] == scene.KIND
        assert "plan total 50s" in prepared["summary"]
    elif action == "deny":
        with pytest.raises(ValueError, match="denied"):
            prepare(settings, revision, child, schema, lambda *a: pytest.fail("denied uploads"))
    else:
        assert not (revision / "agent-edits").exists()
        assert "h3scene_review" not in child


def test_retry_aggregates_errors_then_stops(context, monkeypatch):
    settings, _, schema = context
    folder, state = engine.create_run(settings, "PERSON: Ready.")
    requests = []
    def fake_agent(settings, workspace, request, *args):
        requests.append(request)
        plan = candidate()
        plan["pictures"][0]["path"] = "missing.png"
        plan["segments"][0]["join"] = "cut"
        atomic_json(workspace / "plan.json", plan)
    monkeypatch.setattr(agents, "generate_plan", fake_agent)
    with pytest.raises(ValueError, match="after one correction"):
        prepare(settings, folder, state, schema, lambda *a: pytest.fail("invalid plan uploaded"))
    assert len(requests) == 2
    assert "picture not found" in requests[1]["validation_error"]
    assert "join=first" in requests[1]["validation_error"]


@pytest.mark.parametrize("provider", ["codex", "claude", "grok"])
def test_scene_commands_have_no_discovery_tools(context, monkeypatch, provider):
    settings, _, _ = context
    settings = replace(settings, provider=provider, model_tier="Custom", model="", reasoning="auto")
    monkeypatch.setattr(agents, "executable", lambda n: "/fake/agent")
    args = agents.build_command(settings, Path("/tmp/unused"), [], creative={"scene_chain": True})
    assert "--add-dir" not in args
    if provider == "codex":
        assert "features.shell_tool=false" in args and "include_environment_context=false" in args
    else:
        assert args[args.index("--tools") + 1] == ""
    if provider == "grok":
        assert "--disallowed-tools" in args and "read_file" in args[args.index("--disallowed-tools") + 1]
    assert settings.provider == provider


def test_path_escape_and_flagged_overflow(context):
    settings, _, _ = context
    root = Path(settings.project_folder)
    plan = candidate(overflow={"needs_review": True, "reason": "Pacing"})
    duration = scene.check(plan, settings, "fixture", concept="PERSON: Ready.", roots=[root])
    assert duration["needs_review"] and plan["overflow"]["reason"] == "Pacing"
    for path in ("../assets/ref.png", "/tmp/missing.png"):
        plan = candidate()
        plan["pictures"][0]["path"] = path
        with pytest.raises(ValueError):
            scene.check(plan, settings, "fixture", concept="PERSON: Ready.", roots=[root])


def test_tampered_graph_rejected(context):
    settings, native, _ = context
    graph = scene.compile_graph(native, candidate(), settings, "fixture")
    scene.verify(graph, native)
    graph["1"]["inputs"]["device"] = "cuda:1"
    with pytest.raises(ValueError, match="model loader"):
        scene.verify(graph, native)


def test_schedule_uses_identical_prepare_path(context, tmp_path, monkeypatch):
    from datetime import UTC, datetime

    from simplegen.scheduling import Scheduler
    settings, _, schema = context
    passed = preflight.run_preflight(settings, object_info=schema, resolve=lambda n: n)
    clock = [datetime(2030, 1, 1, tzinfo=UTC).timestamp()]
    scheduler = Scheduler(tmp_path / "schedules", clock=lambda: clock[0])
    try:
        monkeypatch.setattr(agents, "generate_plan", lambda *a: pytest.fail("agent before dispatch"))
        record = scheduler.add(settings, "PERSON: Ready.", datetime.fromtimestamp(clock[0] + 60, UTC), preflight=passed)
        clock[0] += 60
        folder = scheduler.claim(record["id"], passed)
        assert folder is not None
        state = json.loads((folder / "state.json").read_text())
        assert state["schedule"]["released"]
        def fake_agent(settings, workspace, *args):
            atomic_json(workspace / "plan.json", candidate())
        monkeypatch.setattr(agents, "generate_plan", fake_agent)
        prepared, _, _, _ = prepare(settings, folder, state, schema, lambda p, v: ([], v["used_references"]))
        assert state["workflow_acceptance"]["scene_chain"] == scene.KIND and prepared["final_node"]
    finally:
        scheduler.close()


@pytest.mark.parametrize("workflow_folder", ["", "missing", "unrelated"])
def test_builtin_preflight_schedule_prepare_and_resume(context, tmp_path, monkeypatch, workflow_folder):
    from datetime import UTC, datetime

    from simplegen.builtin_workflows import ensure_scene_template
    from simplegen.scheduling import Scheduler

    settings, native, schema = context
    folder_text = str(tmp_path / workflow_folder) if workflow_folder else ""
    if workflow_folder == "unrelated":
        Path(folder_text).mkdir()
        Path(folder_text, "other.json").write_text("{}")
    source = ensure_scene_template()
    settings = replace(settings, template_file=str(source), template_folder=folder_text)
    settings.validate()
    assert engine._template(settings) == source
    passed = preflight.run_preflight(settings, object_info=schema, resolve=lambda n: n)
    assert passed["status"] == "pass"
    assert "Workflow" not in next(c for c in passed["checks"] if c["id"] == "linked_folders")["details"]["folders"]
    clock = [datetime(2030, 1, 1, tzinfo=UTC).timestamp()]
    scheduler = Scheduler(tmp_path / "schedules", clock=lambda: clock[0])
    try:
        monkeypatch.setattr(agents, "generate_plan", lambda *a: pytest.fail("agent before dispatch"))
        record = scheduler.add(settings, "PERSON: Ready.", datetime.fromtimestamp(clock[0] + 60, UTC), preflight=passed)
        folder = Path(record["folder"])
        assert json.loads((folder / "template.original.json").read_text()) == native
        source.unlink()  # The run owns its snapshot after acceptance of the schedule.
        checked = preflight.run_saved_preflight(folder, object_info=schema, resolve=lambda n: n)
        assert checked["status"] == "pass"
        clock[0] += 60
        assert scheduler.claim(record["id"], checked) == folder
        state = json.loads((folder / "state.json").read_text())

        def fake_agent(settings, workspace, *args):
            atomic_json(workspace / "plan.json", candidate())

        monkeypatch.setattr(agents, "generate_plan", fake_agent)
        prepared, plan, _, _ = prepare(settings, folder, state, schema, lambda p, v: ([], v["used_references"]))
        assert state["workflow_acceptance"]["scene_chain"] == scene.KIND
        assert prepared["final_node"] and plan["script"] == "PERSON: Ready."
        monkeypatch.setattr(agents, "generate_plan", lambda *a: pytest.fail("agent on accepted resume"))
        resumed, resumed_plan, _, _ = prepare(settings, folder, state, schema, lambda p, v: ([], v["used_references"]))
        assert resumed == prepared and resumed_plan == plan
    finally:
        scheduler.close()


def test_builtin_cloud_preflight_block_and_unverified_schema(context, monkeypatch):
    from simplegen.builtin_workflows import ensure_scene_template

    settings, _, schema = context
    settings = replace(settings, template_file=str(ensure_scene_template()), template_folder="")
    cloud = preflight.run_preflight(replace(settings, cloud=True), object_info=schema, resolve=lambda n: n)
    assert cloud["status"] == "fail"
    scope = next(c for c in cloud["checks"] if c["id"] == "h3scene_scope")
    assert scope["status"] == "fail" and "MultiGPU" in scope["message"]
    monkeypatch.setattr(preflight, "target_schema", lambda *a, **k: (None, {"error": "offline"}))
    unknown = preflight.run_preflight(settings, resolve=lambda n: n)
    assert unknown["status"] == "unverified"
    assert next(c for c in unknown["checks"] if c["id"] == "template_structure")["status"] == "unverified"


@pytest.mark.parametrize(("change", "reason"), [({"cloud": True}, "MultiGPU"), ({"quantity": 2}, "Qty 1")])
def test_engine_refuses_unsupported_before_service_or_variations(context, monkeypatch, change, reason):
    settings, _, _ = context
    monkeypatch.setattr(engine, "Comfy", lambda *a: pytest.fail("unsupported contacted ComfyUI"))
    from simplegen import variations
    monkeypatch.setattr(variations, "run_sequence", lambda *a: pytest.fail("scene variations expanded"))
    with pytest.raises(ValueError, match=reason):
        engine.run(replace(settings, **change), "PERSON: Ready.", lambda _: None, Event())


def test_saved_review_buttons(context, monkeypatch):
    from PySide6.QtWidgets import QApplication, QPushButton

    from simplegen.ui import SavedRunDialog
    app = QApplication.instance() or QApplication([])
    settings, native, _ = context
    folder, state = engine.create_run(settings, "PERSON: Ready.")
    atomic_json(folder / "template.original.json", native)
    state["h3scene_review"] = {"reason": "Plan requires 50 seconds."}
    dialog = SavedRunDialog(state, folder)
    assert dialog.resume_button.text() == "Approve"
    assert dialog.new_button.text() == "Edit as new run"
    assert "50 seconds" in dialog.explanation.text()
    assert "Deny" in [b.text() for b in dialog.findChildren(QPushButton)]
    assert not dialog.repair_button.isVisible()
    dialog.choose("scene_approve")
    assert dialog.choice == "scene_approve"
    dialog.deleteLater()
    app.processEvents()


@pytest.mark.parametrize("provider", ["codex", "claude", "grok"])
def test_agent_exact_input_neutral_cwd_and_selected_provider(context, monkeypatch, tmp_path, provider):
    import sys
    settings, native, _ = context
    settings = replace(settings, provider=provider, model_tier="Custom", model="", reasoning="auto")
    brief = scene.brief(native, settings, "fixture", Path("unused"), concept="PERSON: Ready.")
    captured = []
    fake = tmp_path / "fake_agent.py"
    # A fake executable writes the request/argv/cwd to a temporary test receipt.
    receipt = tmp_path / "call.json"
    fake.write_text("import json,sys,os\nfrom pathlib import Path\n" +
                    f"Path({str(receipt)!r}).write_text(json.dumps({{'stdin':sys.stdin.read(),'argv':sys.argv,'cwd':os.getcwd()}}))\n" +
                    f"print(json.dumps({{'structured_output':{{'plans':[{candidate()!r}]}}}}))\n")
    original = agents.build_command
    monkeypatch.setattr(agents, "executable", lambda n: "/fake/" + n)
    def command(*args, **kwargs):
        argv = original(*args, **kwargs)
        captured.append(argv)
        return [sys.executable, str(fake), *argv[1:]]
    monkeypatch.setattr(agents, "build_command", command)
    monkeypatch.setattr(agents, "_subscription", lambda *a: None)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    plan = agents.generate_plan(settings, workspace, brief, Event(), lambda _: None)
    assert plan["segments"] and captured
    call = json.loads(receipt.read_text())
    assert call["stdin"] == brief["scene_messages"][1]["content"]
    assert Path(call["cwd"]) != workspace and not Path(call["cwd"]).exists()
    assert json.loads((workspace / "agent-messages.json").read_text()) == brief["scene_messages"]
    assert (workspace / "scene-system.txt").read_text() == brief["scene_messages"][0]["content"]
    assert "project_index" not in call["stdin"]


@pytest.mark.parametrize("provider", ["codex", "claude", "grok"])
def test_agent_start_failure_cleans_neutral_cwd(context, monkeypatch, tmp_path, provider):
    settings, native, _ = context
    settings = replace(settings, provider=provider)
    brief = scene.brief(native, settings, "fixture", Path("unused"), concept="PERSON: Ready.")
    monkeypatch.setattr(agents, "build_command", lambda *a, **k: ["fake-agent", "--cwd", "unused"])
    monkeypatch.setattr(agents, "_subscription", lambda *a: None)
    temporary = agents.tempfile.TemporaryDirectory
    retained = []
    def track_temporary(*args, **kwargs):
        directory = temporary(*args, **kwargs)
        retained.append(directory)  # Prevent implicit destruction from satisfying the assertion.
        return directory
    monkeypatch.setattr(agents.tempfile, "TemporaryDirectory", track_temporary)
    directories = []
    def fail_start(*args, **kwargs):
        directory = Path(kwargs["cwd"])
        assert directory.is_dir() and directory != tmp_path
        if provider == "grok":
            assert args[0][args[0].index("--cwd") + 1] == str(directory)
        directories.append(directory)
        raise OSError("fake startup failure")
    monkeypatch.setattr(agents.subprocess, "Popen", fail_start)
    with pytest.raises(OSError, match="fake startup failure"):
        agents.generate_plan(settings, tmp_path, brief, Event(), lambda _: None)
    assert len(retained) == len(directories) == 1 and not directories[0].exists()


@pytest.mark.parametrize(("requested", "total", "mp", "warning"), [
    (60, 61, .52, "Unverified local tested envelope"),
    (30, 31, .72, "Unverified local tested envelope"),
    (44, 45, .72, "Host-RAM warning"),
    (45, 46, .72, None),
    (30, 30, .52, None),
])
def test_plan_total_envelope_warning_reaches_report_and_log(context, monkeypatch, requested, total, mp, warning):
    settings, _, schema = context
    settings = replace(settings, duration=requested, megapixels=mp)
    events, submissions = [], []
    class FakeComfy:
        def __init__(self, settings):
            pass
        def object_info(self):
            return copy.deepcopy(schema)
        def queue(self):
            return {"queue_running": [], "queue_pending": []}
        def upload(self, path, run_id, cancel):
            return "uploaded/ref.png"
        def submit(self, graph, *args):
            submissions.append(graph)
            return "fake-job"
        def monitor(self, prompt_id, client_id, graph, cancel, emit):
            return {"status": {"completed": True, "status_str": "success"},
                    "outputs": {key: {"videos": [{"filename": f"{key}.mp4", "type": "output", "subfolder": ""}]}
                                for key, node in graph.items() if node["class_type"] == "SaveVideo"}}
        def download(self, descriptor, target, cancel, callback):
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"video")
            return {"path": str(target), "size_bytes": 5, "sha256": hashlib.sha256(b"video").hexdigest()}
        def close(self):
            pass
    monkeypatch.setattr(engine, "Comfy", FakeComfy)
    monkeypatch.setattr(engine, "probe_video", lambda p: {"width": 960, "height": 544, "duration": total, "size_bytes": 5})
    monkeypatch.setattr(agents, "executable", lambda n: n)
    count = (total + 14) // 15
    def fake_agent(settings, workspace, *args):
        atomic_json(workspace / "plan.json", candidate((total / count,) * count))
    monkeypatch.setattr(agents, "generate_plan", fake_agent)
    result = engine.run(settings, "PERSON: Ready.", events.append, Event())
    folder = Path(result["folder"])
    state = json.loads((folder / "state.json").read_text())
    assert len(submissions) == 1 and state["status"] == "complete"
    warnings = [w for w in state["workflow_acceptance"]["lint_warnings"] if w.startswith("H3 Scene Chain plan total")]
    log_warnings = [e["message"] for e in events if e["message"].startswith("Warning: H3 Scene Chain plan total")]
    report = (folder / "report.txt").read_text()
    if warning:
        assert len(warnings) == 1 and warning in warnings[0]
        assert log_warnings == [f"Warning: {warnings[0]}"]
        assert f"Warning: {warnings[0]}" in report
    else:
        assert not warnings and not log_warnings and "Warning: H3 Scene Chain plan total" not in report


def test_approval_uses_reviewed_plan_not_modified_attempt(context, monkeypatch):
    settings, _, schema = context
    settings = replace(settings, megapixels=.72)
    folder, state = engine.create_run(settings, "PERSON: Ready.")
    def fake_agent(settings, workspace, *args):
        atomic_json(workspace / "plan.json", candidate((15, 15, 15, 5)))
    monkeypatch.setattr(agents, "generate_plan", fake_agent)
    with pytest.raises(OverflowReview):
        prepare(settings, folder, state, schema, lambda *a: pytest.fail("review uploaded"))
    revision, child = decide(folder, "approve")
    attempt = revision / "agent-edits" / "attempt-1" / "plan.json"
    altered = candidate((5, 5))
    atomic_json(attempt, altered)
    monkeypatch.setattr(agents, "generate_plan", lambda *a: pytest.fail("approval called agent"))
    prepared, accepted, _, _ = prepare(settings, revision, child, schema, lambda p, v: ([], []))
    assert sum(s["seconds"] for s in accepted["segments"]) == 50
    assert "plan total 50s" in prepared["summary"]
    assert any("outside trained range" in w for w in child["workflow_acceptance"]["lint_warnings"])
    assert any("plan total 50s" in w and "Unverified" in w and "Host-RAM" in w
               for w in child["workflow_acceptance"]["lint_warnings"])


@pytest.mark.parametrize(("seconds", "review"), [((15, 15), False), ((15, 15, 15), False), ((15, 15, 15, 5), True)])
def test_computed_overflow_boundary(context, seconds, review):
    settings, _, _ = context
    value = candidate(seconds, overflow={"extra_seconds": 0, "needs_review": False})
    checked = scene.check(value, settings, "fixture", concept="PERSON: Ready.", roots=[Path(settings.project_folder)])
    assert checked["needs_review"] == review and value["overflow"]["needs_review"] == review
    assert value["overflow"]["extra_seconds"] == max(0, sum(seconds) - 30)


@pytest.mark.parametrize("overflow", [False, True])
def test_engine_upload_submission_and_review_gate(context, monkeypatch, overflow):
    settings, _, schema = context
    uploaded, submitted = [], []
    class FakeComfy:
        def __init__(self, settings):
            pass
        def object_info(self):
            return copy.deepcopy(schema)
        def queue(self):
            return {"queue_running": [], "queue_pending": []}
        def upload(self, path, run_id, cancel):
            uploaded.append(path)
            return "uploaded/ref.png"
        def submit(self, graph, *args):
            submitted.append(graph)
            raise RuntimeError("fake submission reached")
        def close(self):
            pass
    monkeypatch.setattr(engine, "Comfy", FakeComfy)
    monkeypatch.setattr(agents, "executable", lambda n: n)
    def fake_agent(settings, workspace, *args):
        atomic_json(workspace / "plan.json", candidate(overflow={"needs_review": overflow, "reason": "Pacing"}))
    monkeypatch.setattr(agents, "generate_plan", fake_agent)
    with pytest.raises(RuntimeError, match="Pacing" if overflow else "fake submission reached"):
        engine.run(settings, "PERSON: Ready.", lambda _: None, Event())
    folder = next(Path(settings.output_folder).iterdir())
    state = json.loads((folder / "state.json").read_text())
    if overflow:
        assert state["status"] == "needs_review"
        assert not state.get("submission_started") and not state.get("workflow_acceptance")
        assert not (folder / "workflow.json").exists()
        assert not uploaded and not submitted
    else:
        assert state["workflow_acceptance"]["scene_chain"] == scene.KIND
        assert len(uploaded) == 1 and len(submitted) == 1
        assert all(n["inputs"]["image"] == "uploaded/ref.png" for n in submitted[0].values() if n["class_type"] == "LoadImage")
        assert len([n for n in submitted[0].values() if n["class_type"] == "SaveVideo"]) == 3
        assert "plan total 30s" in state["summary"]


@pytest.mark.parametrize("missing", ["MiniMaxH3VideoExtender", "MiniMaxH3VideoMerge"])
def test_preflight_checks_all_join_nodes_before_agent(context, missing):
    settings, native, schema = context
    schema = {k: v for k, v in schema.items() if k != missing}
    problem = scene.support_problem(native, schema, settings, "local")
    assert missing in problem
    result = preflight.run_preflight(settings, object_info=schema, resolve=lambda n: n)
    assert next(c for c in result["checks"] if c["id"] == "template_structure")["status"] == "fail"


def test_strict_output_with_null_overflow_and_empty_warnings_passes(context):
    """Codex/Claude strict output sends every field; null overflow means the plan fits."""
    settings, _, _ = context
    item = scene.plan_schema()["properties"]["plans"]["items"]
    plan = candidate(overflow=None, warnings=[])
    assert set(plan) == set(item["required"])
    duration = scene.check(plan, settings, "fixture", concept="PERSON: Ready.", roots=[Path(settings.project_folder)])
    assert duration["needs_review"] is False and plan["overflow"]["needs_review"] is False


def test_strict_output_empty_overflow_strings_get_the_computed_reason(context):
    settings, _, _ = context
    plan = candidate((15, 15, 15, 5), warnings=[],
                     overflow={"needs_review": True, "extra_seconds": 20, "reason": "", "options": ""})
    duration = scene.check(plan, settings, "fixture", concept="PERSON: Ready.", roots=[Path(settings.project_folder)])
    assert duration["needs_review"] is True
    assert plan["overflow"]["reason"].startswith("Plan requires")
    assert plan["overflow"]["options"] == "approve / edit / deny"
