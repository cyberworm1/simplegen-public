"""Single-shot H3 Ref2VA voice templates (h3voice): preflight, plan checks, compile, run.

Real template/schema fixtures, a fake agent CLI and a fake ComfyUI. No real services.
"""

import copy
import hashlib
import json
import math
import os
import shutil
import struct
import sys
import wave
from dataclasses import replace
from pathlib import Path
from threading import Event

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

from simplegen import agents, batches, engine, global_session, preflight, refcache, singleshot
from simplegen.dimensions import resolve_dimensions
from simplegen.settings import Settings
from simplegen.workflows import _flatten, validate_graph

HERE = Path(__file__).parent
REAL = HERE / "fixtures" / "real"
BUNDLED = Path(__file__).parents[1] / "src" / "simplegen" / "data" / "workflows"
CLOUD = "H3_Ref2VA_Voice_SingleShot.json"
LOCAL = "H3_Ref2VA_Voice_SingleShot_LOCAL_INT8.json"
PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32
VENDORED = HERE.parent / "src" / "simplegen" / "h3voice"
CANONICAL = Path(os.environ.get("H3VOICE_SRC", "/nonexistent"))  # the external h3voice source, when synced locally
needs_ffprobe = pytest.mark.skipif(not shutil.which("ffprobe"), reason="install ffmpeg (ffprobe) for audio checks")


def tools(name):
    return f"/usr/bin/{name}"


@pytest.fixture(scope="module")
def schemas():
    cloud = json.loads((REAL / "schemas" / "cloud_global_object_info.json").read_text())
    return {CLOUD: cloud.get("object_info", cloud),
            LOCAL: json.loads((REAL / "schemas" / "local_object_info.json").read_text())}


@pytest.fixture(scope="module")
def templates():
    return {name: json.loads((BUNDLED / name).read_text()) for name in (CLOUD, LOCAL)}


def write_wav(path: Path, seconds: float = 3.0, rate: int = 24000) -> Path:
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(rate)
        out.writeframes(b"".join(struct.pack("<h", int(8000 * math.sin(i / 20))) for i in range(int(seconds * rate))))
    return path


@pytest.fixture
def assets(tmp_path):
    root = tmp_path / "assets"
    root.mkdir()
    (root / "anna.png").write_bytes(PNG)
    write_wav(root / "anna_voice.wav")
    return root


def plan_for(assets: Path, text="We set off at noon.") -> dict:
    return {
        "script": "Anna tells the crew the plan.",
        "used_references": [str(assets / "anna.png"), str(assets / "anna_voice.wav")],
        "shot": {
            "pictures": [{"id": "anna_face", "file": str(assets / "anna.png"), "note": ""}],
            "voices": [{"id": "anna_voice", "file": str(assets / "anna_voice.wav"), "subject": "anna",
                        "start": 0.0, "duration": 3.0, "language": "English", "delivery": "measured"}],
            "subjects": [{"id": "anna", "who": "the woman", "pictures": ["anna_face"],
                          "keep": "same face, hair, and wardrobe"}],
            "shots": [{"description": "Medium close-up of [[anna]] at the window.",
                       "lines": [{"speaker": "anna", "text": text, "verb": "says"}], "after": ""}],
            "style": "realistic cinematic", "soundscape": "Quiet room tone.", "music": "none",
        },
    }


def settings_for(assets: Path, tmp_path: Path, template=LOCAL, **changes) -> Settings:
    folder = tmp_path / "templates"
    if not folder.exists():
        shutil.copytree(BUNDLED, folder)
    base = Settings(project_folder=str(assets), template_folder=str(folder), template_file=template,
                    output_folder=str(tmp_path / "out"), duration=6.0, steps=20, megapixels=0.5,
                    aspect_ratio="16:9")
    return replace(base, **changes)


# Vendoring


def test_vendored_files_match_manifest():
    manifest = json.loads((VENDORED / "VENDORED.json").read_text())
    for name, sha in manifest["files"].items():
        assert hashlib.sha256((VENDORED / name).read_bytes()).hexdigest() == sha, f"{name} edited in place; run tools/sync_h3voice.py"


@pytest.mark.skipif(not CANONICAL.is_dir(), reason="canonical h3voice source not on this machine")
def test_vendored_files_match_canonical_source():
    manifest = json.loads((VENDORED / "VENDORED.json").read_text())
    stale = [n for n in manifest["files"] if (CANONICAL / n).read_bytes() != (VENDORED / n).read_bytes()]
    assert not stale, f"run tools/sync_h3voice.py: {stale}"


# Template recognition and preflight


def test_templates_are_single_shot_with_their_own_models(templates):
    assert singleshot.profile(templates[CLOUD])["models"]["unet"]["inputs"]["unet_name"] == "minimax_h3_ref2va_bf16.safetensors"
    assert "int8" in singleshot.profile(templates[LOCAL])["models"]["unet"]["inputs"]["unet_name"]
    chain = json.loads((BUNDLED / "H3_Persistent_References_Clonable_Chain.json").read_text())
    assert not singleshot.is_single_shot(chain)


@pytest.mark.parametrize(("template", "schema", "expected"), [
    (LOCAL, LOCAL, "pass"), (CLOUD, CLOUD, "pass"),
    (CLOUD, LOCAL, "fail"),  # BF16 models are not on the workstation
    (LOCAL, CLOUD, "fail"),  # INT8 + MultiGPU loaders pick models the pod lacks
])
def test_preflight_matrix(tmp_path, assets, schemas, template, schema, expected):
    settings = settings_for(assets, tmp_path, template)
    result = preflight.run_preflight(settings, object_info=schemas[schema], resolve=tools)
    checks = {c["id"]: c for c in result["checks"]}
    if (template, schema) == (LOCAL, CLOUD):  # the pod has one GPU: cuda:1 does not exist there
        assert checks["template_structure"]["status"] == "fail"
        assert "'cuda:1'" in checks["template_structure"]["message"]
    else:
        assert checks["template_structure"]["status"] == "pass", checks["template_structure"]["message"]
    assert checks["target_compatibility"]["status"] == expected, checks["target_compatibility"]["message"]
    if expected == "fail":
        assert checks["target_compatibility"]["details"]["missing_models"]


@pytest.mark.parametrize(("pack_class", "missing"), [("MultiGPU", "UNETLoaderMultiGPU"), ("VRAM_Debug", "VRAM_Debug")])
def test_preflight_names_missing_node_pack(tmp_path, assets, schemas, pack_class, missing):
    info = {k: v for k, v in schemas[LOCAL].items() if pack_class not in k}
    result = preflight.run_preflight(settings_for(assets, tmp_path), object_info=info, resolve=tools)
    checks = {c["id"]: c for c in result["checks"]}
    assert result["status"] == "fail"
    for name in ("template_structure", "target_compatibility"):
        assert checks[name]["status"] == "fail" and missing in checks[name]["message"], checks[name]["message"]


def test_local_template_device_map(templates, schemas):
    graph, _ = _flatten(copy.deepcopy(templates[LOCAL]), schemas[LOCAL])
    devices = {n["class_type"]: n["inputs"].get("device") for n in graph.values() if "Loader" in n["class_type"]}
    assert devices == {"UNETLoaderMultiGPU": "cuda:0", "CLIPLoaderMultiGPU": "cuda:1", "VAELoaderMultiGPU": "cuda:0"}
    assert [n["class_type"] for n in graph.values() if n["inputs"].get("device") == "cuda:1"] == ["CLIPLoaderMultiGPU"]
    vram = next(n for n in graph.values() if n["class_type"] == "VRAM_Debug")["inputs"]
    assert (vram["empty_cache"], vram["gc_collect"], vram["unload_all_models"]) == (True, True, True)


def test_cloud_template_has_no_multigpu_nodes(templates, schemas):
    workflow = templates[CLOUD]
    assert not [n["type"] for n in workflow["nodes"] if "MultiGPU" in n["type"]]
    assert singleshot.multigpu_nodes(workflow) == []
    graph, _ = _flatten(copy.deepcopy(workflow), schemas[CLOUD])
    assert not [n["class_type"] for n in graph.values() if "MultiGPU" in n["class_type"]]
    assert not any(n["inputs"].get("device") == "cuda:1" for n in graph.values())


def test_cloud_target_rejects_multigpu_template(tmp_path, assets, schemas):
    settings = settings_for(assets, tmp_path, LOCAL, cloud=True)
    result = preflight.run_preflight(settings, object_info=schemas[CLOUD], resolve=tools)
    check = next(c for c in result["checks"] if c["id"] == "template_structure")
    assert check["status"] == "fail" and "MultiGPU loaders" in check["message"]
    ok = preflight.run_preflight(settings_for(assets, tmp_path, CLOUD, cloud=True), object_info=schemas[CLOUD],
                                 resolve=tools)
    assert next(c for c in ok["checks"] if c["id"] == "template_structure")["status"] == "pass"


@pytest.mark.parametrize(("changes", "message"), [
    ({"duration": 45.0}, "set Duration to 15 or less"),
    ({"quantity": 2}, "set Qty to 1"),
])
def test_preflight_rejects_unsupported_settings(tmp_path, assets, schemas, changes, message):
    settings = settings_for(assets, tmp_path, **changes)
    result = preflight.run_preflight(settings, object_info=schemas[LOCAL], resolve=tools)
    check = next(c for c in result["checks"] if c["id"] == "template_structure")
    assert check["status"] == "fail" and message in check["message"]


# Plan checks and compile


@needs_ffprobe
@pytest.mark.parametrize("name", [LOCAL, CLOUD])
def test_compile_validates_against_target(tmp_path, assets, schemas, templates, name):
    settings = settings_for(assets, tmp_path, name)
    spec, text = singleshot.check(plan_for(assets), settings, "run-1", concept="", roots=[assets])
    graph = singleshot.compile_graph(templates[name], spec, text, settings, "run-1", schemas[name])
    validate_graph(graph, schemas[name])
    r2v = graph["r2v"]["inputs"]
    assert (r2v["width"], r2v["height"]) == resolve_dimensions(0.5, 16, 9, "16:9") == (928, 544)
    assert r2v["length"] % 17 == 5
    assert r2v["ref_audios.ref_audio_0"] == ["trim_anna_voice", 0]
    assert "<Audio 1> is the voice-timbre reference for <Subject 1> (S1)" in r2v["prompt"]
    assert "without copying the original signal" in r2v["prompt"]
    assert graph["save"]["inputs"]["filename_prefix"] == "simpleGen/run-1/final"
    assert graph["sigmas"]["inputs"]["steps"] == 20
    singleshot.verify(graph, templates[name])
    again = singleshot.compile_graph(templates[name], spec, text, settings, "run-1", schemas[name])
    assert json.dumps(again, sort_keys=True) == json.dumps(graph, sort_keys=True)  # deterministic
    prepared = singleshot.prepared(graph)
    assert {s["input"] for s in prepared["reference_slots"]} == {"image", "audio"}
    assert all(Path(s["value"]).is_absolute() for s in prepared["reference_slots"])


@needs_ffprobe
def test_plan_problems_reach_the_correction_pass(tmp_path, assets):
    settings = settings_for(assets, tmp_path)
    bad = plan_for(assets, text=" ".join(["word"] * 40) + ".")
    bad["shot"]["voices"][0]["file"] = "/elsewhere/voice.wav"
    with pytest.raises(ValueError) as caught:
        singleshot.check(bad, settings, "r", concept='She says "we set off at noon tonight" quietly.', roots=[assets])
    message = str(caught.value)
    assert "is outside the linked folders" in message
    assert "budget" in message
    assert "must be spoken verbatim" in message
    with pytest.raises(ValueError, match="not a subject id"):
        wrong = plan_for(assets)
        wrong["shot"]["shots"][0]["lines"][0]["speaker"] = "ghost"
        singleshot.check(wrong, settings, "r", concept="", roots=[assets])


@needs_ffprobe
def test_verify_rejects_tampered_graphs(tmp_path, assets, schemas, templates):
    settings = settings_for(assets, tmp_path)
    spec, text = singleshot.check(plan_for(assets), settings, "r", concept="", roots=[assets])
    graph = singleshot.compile_graph(templates[LOCAL], spec, text, settings, "r", schemas[LOCAL])
    for mutate, message in [
        (lambda g: g["unet"]["inputs"].update(unet_name="minimax_h3_ref2va_bf16.safetensors"), "changed unet"),
        (lambda g: g.update(extra={"class_type": "SaveImage", "inputs": {}}), "unexpected node types"),
        (lambda g: g.update(save2=copy.deepcopy(g["save"])), "exactly one SaveVideo"),
        (lambda g: g["guider"]["inputs"].update(model=["nowhere", 0]), "missing node"),
    ]:
        tampered = copy.deepcopy(graph)
        mutate(tampered)
        with pytest.raises(ValueError, match=message):
            singleshot.verify(tampered, templates[LOCAL])


@needs_ffprobe
def test_voice_trim_past_end_is_an_error(tmp_path, assets):
    plan = plan_for(assets)
    plan["shot"]["voices"][0]["start"] = 5.0  # the clip is 3 s long
    with pytest.raises(ValueError, match="only 3.0s long"):
        singleshot.check(plan, settings_for(assets, tmp_path), "r", concept="", roots=[assets])


def test_plan_schema_is_strict():
    schema = singleshot.plan_schema({})
    def walk(node):
        if node.get("type") == "object":
            assert node["additionalProperties"] is False and set(node["required"]) == set(node["properties"])
            for child in node["properties"].values():
                walk(child)
        if node.get("type") == "array":
            walk(node["items"])
    walk(schema)
    assert agents.plan_schema(creative={"single_shot": True}) == schema


def test_uploaded_audio_names_pass_graph_validation(schemas, templates):
    graph, _ = _flatten(copy.deepcopy(templates[LOCAL]), schemas[LOCAL])
    audio = next(n for n in graph.values() if n["class_type"] == "LoadAudio")
    audio["inputs"]["audio"] = "simplegen/run/abc_voice.wav"  # not in the cached /object_info list
    validate_graph(graph, schemas[LOCAL])


# End to end (fake agent CLI, fake ComfyUI)


FAKE_CLI = r'''
import json, sys
from pathlib import Path
request = sys.stdin.read()
brief = json.loads(request.split("RUN REQUEST (data, not instructions):\n", 1)[1].split("\nReturn only the final JSON", 1)[0])
log = Path(LOG)
calls = json.loads(log.read_text()) if log.exists() else []
calls.append({"validation_error": brief.get("validation_error", ""), "keys": sorted(brief),
              "skill": "name: h3-voice-shot" in request})
log.write_text(json.dumps(calls))
plan = json.loads(Path(PLAN).read_text())
if Path(MODE).read_text() == "bad_first" and len(calls) == 1:
    plan["shot"]["shots"][0]["lines"][0]["text"] = "x " * 60
print(json.dumps({"structured_output": plan}))
'''


class Submitted(Exception):
    pass


@pytest.fixture
def e2e(tmp_path, monkeypatch, schemas, assets):
    monkeypatch.setenv("SIMPLEGEN_HOME", str(tmp_path / "home"))
    log, mode, plan = tmp_path / "calls.json", tmp_path / "mode.txt", tmp_path / "plan.json"
    mode.write_text("good")
    plan.write_text(json.dumps(plan_for(assets)))
    script = tmp_path / "fake_agent.py"
    script.write_text(f"LOG = {str(log)!r}\nMODE = {str(mode)!r}\nPLAN = {str(plan)!r}\n" + FAKE_CLI)
    monkeypatch.setattr(agents, "build_command", lambda *a, **k: [sys.executable, str(script)])
    monkeypatch.setattr(agents, "_subscription", lambda *a: None)
    monkeypatch.setattr(agents, "executable", tools)
    uploaded, submitted = [], []

    class Comfy:
        def __init__(self, settings, client=None):
            pass

        def object_info(self):
            return copy.deepcopy(schemas[LOCAL])

        def queue(self):
            return {"queue_running": [], "queue_pending": []}

        def upload(self, path, run_id, cancel):
            uploaded.append(Path(path).suffix)
            return f"simplegen/{run_id}/{Path(path).name}"

        def submit(self, graph, run_id, client_id):
            submitted.append(graph)
            raise Submitted("fake submit reached")

        def find_submission(self, run_id):
            return None

        def close(self):
            pass
    monkeypatch.setattr(engine, "Comfy", Comfy)
    settings = settings_for(assets, tmp_path)

    def calls():
        return json.loads(log.read_text()) if log.exists() else []
    return settings, mode, calls, uploaded, submitted


def settings_path(settings):
    return Path(settings.project_folder)


def only_run(settings):
    return next(p for p in Path(settings.output_folder).iterdir() if p.is_dir())


@needs_ffprobe
def test_concept_to_submission_uploads_voice(e2e):
    settings, _, calls, uploaded, submitted = e2e
    with pytest.raises(RuntimeError, match="fake submit reached"):
        engine.run(settings, 'Anna says "we set off at noon"', lambda _: None, Event())
    folder = only_run(settings)
    state = json.loads((folder / "state.json").read_text())
    assert len(calls()) == 1 and calls()[0]["skill"] and "single_shot" in calls()[0]["keys"]
    receipt = state["workflow_acceptance"]
    assert receipt["single_shot"] == singleshot.KIND and receipt["profile"] == "local"
    assert receipt["workflow_sha256"] == hashlib.sha256((folder / "workflow.json").read_bytes()).hexdigest()
    assert sorted(uploaded) == [".png", ".wav"]
    graph = submitted[0]
    assert graph["voice_anna_voice"]["inputs"]["audio"].startswith(f"simplegen/{state['run_id']}/")  # cached copy, uploaded
    assert graph["voice_anna_voice"]["inputs"]["audio"].endswith(".wav")
    assert not Path(graph["voice_anna_voice"]["inputs"]["audio"]).is_absolute()
    assert graph["noise"]["inputs"]["noise_seed"] == singleshot.seed_for(state["run_id"])
    manifest = refcache.load(folder)
    cached = {Path(e["original"]).suffix: e for e in manifest["references"]}
    assert set(cached) == {".png", ".wav"} and refcache.pinned(state["run_id"])
    assert cached[".wav"]["sha256"] == hashlib.sha256((settings_path(settings) / "anna_voice.wav").read_bytes()).hexdigest()
    assert cached[".wav"]["blob"].endswith(".wav")


@needs_ffprobe
def test_first_plan_rejected_gets_one_correction(e2e):
    settings, mode, calls, _, submitted = e2e
    mode.write_text("bad_first")
    with pytest.raises(RuntimeError, match="fake submit reached"):
        engine.run(settings, "Anna speaks", lambda _: None, Event())
    assert len(calls()) == 2 and "budget" in calls()[1]["validation_error"]
    state = json.loads((only_run(settings) / "state.json").read_text())
    assert [a["status"] for a in state["workflow_edit_attempts"]] == ["invalid", "valid"]
    assert len(submitted) == 1


@needs_ffprobe
def test_resume_reuses_accepted_work(e2e):
    settings, _, calls, _, submitted = e2e
    with pytest.raises(RuntimeError, match="fake submit reached"):
        engine.run(settings, "Anna speaks", lambda _: None, Event())
    folder = only_run(settings)
    state = json.loads((folder / "state.json").read_text())
    state.pop("submission_started", None)
    (folder / "state.json").write_text(json.dumps(state))
    with pytest.raises(RuntimeError, match="fake submit reached"):
        engine.run(settings, "", lambda _: None, Event(), resume_folder=folder)
    assert len(calls()) == 1 and submitted[0] == submitted[1]


@needs_ffprobe
def test_repair_refuses_single_shot_runs(e2e):
    from simplegen import repair
    settings, _, _, _, _ = e2e
    with pytest.raises(RuntimeError, match="fake submit reached"):
        engine.run(settings, "Anna speaks", lambda _: None, Event())
    folder = only_run(settings)
    state = json.loads((folder / "state.json").read_text())
    state.pop("submission_started", None)
    state.pop("prompt_id", None)
    assert "single-shot voice runs" in repair.refusal(folder, state)


@needs_ffprobe
def test_batch_member_prepares_single_shot(e2e, schemas, monkeypatch, tmp_path):
    settings, _, calls, _, _ = e2e
    profile = tmp_path / "global-profile.json"
    from importlib.resources import files
    profile.write_text(json.dumps({**json.loads(files("simplegen").joinpath("data/global-runpod-profile.json").read_text()),
                     "template_id": "tmpl-test", "volume_id": "vol-test"}))
    settings = replace(settings, template_file=CLOUD, managed_runpod=True, global_desktop=True,
                       runpod_config_file=str(profile))

    class Session:
        retired = False

        def __init__(self, settings, folder, state, persist, *args):
            self.settings = settings

        def schema(self):
            return copy.deepcopy(schemas[CLOUD])

        def close(self):
            pass
    monkeypatch.setattr(global_session, "session_for", Session)
    store = batches.BatchStore(settings.output_folder)
    batch = store.create()
    folder, _ = engine.create_run(settings, "Anna speaks")
    store.add(batch["id"], folder)
    ready = store.prepare(batch["id"], lambda _: None, Event())
    assert ready["status"] == "prepared", ready
    state = json.loads((folder / "state.json").read_text())
    assert state["workflow_acceptance"]["profile"] == "runpod" and len(calls()) == 1
    graph = json.loads((folder / "workflow.json").read_text())
    assert graph["unet"]["inputs"]["unet_name"] == "minimax_h3_ref2va_bf16.safetensors"
