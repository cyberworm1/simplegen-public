"""Run lifecycle tests with synthetic workflows, references and provider output."""

import copy
import hashlib
import json
import shutil
from dataclasses import replace
from pathlib import Path
from threading import Event

import pytest

from simplegen import engine
from simplegen.settings import Settings, atomic_json


@pytest.fixture
def scenario(tmp_path, monkeypatch):
    # Retain service/recovery coverage for pre-editing schema 1 projects.
    monkeypatch.setattr(engine, "STATE_VERSION", 1)
    assets = tmp_path / "assets"
    templates = tmp_path / "templates"
    assets.mkdir()
    templates.mkdir()
    reference = assets / "reference.png"
    reference.write_bytes(b"synthetic reference")
    (templates / "test.json").write_text("{}")
    settings = Settings(
        project_folder=str(assets),
        template_folder=str(templates),
        output_folder=str(tmp_path / "outputs"),
        template_file="test.json",
    )
    graph = {
        "p": {"class_type": "Text", "inputs": {"text": "old"}},
        "image": {"class_type": "LoadImage", "inputs": {"image": "old.png"}},
        "final": {"class_type": "SaveVideo", "inputs": {}},
        "shot": {"class_type": "SaveVideo", "inputs": {}},
    }
    prepared = {
        "graph": graph,
        "prompt_slots": [{"node_id": "p", "input": "text", "text": "old"}],
        "reference_slots": [{"node_id": "image", "input": "image", "value": "old.png"}],
        "final_node": "final",
        "summary": "Synthetic test workflow",
    }
    plan = {
        "script": "A quiet landscape.",
        "prompts": [{"node_id": "p", "input": "text", "text": "A meadow."}],
        "references": [{"node_id": "image", "input": "image", "path": str(reference)}],
        "used_references": [str(reference)],
    }
    history = {
        "status": {"completed": True, "status_str": "success"},
        "outputs": {
            "final": {"images": [{"filename": "final.mp4", "type": "output", "subfolder": "x"}]},
            "shot": {"videos": [{"filename": "shot.mp4", "type": "output", "subfolder": "x"}]},
        },
    }
    calls = {"agent": 0, "submit": 0, "download": 0, "upload": 0}

    class FakeComfy:
        fail_download = False
        uncertain = False
        recovered = "job-123"

        def __init__(self, config):
            calls["endpoint"] = config.comfy_url

        def object_info(self):
            return {}

        def upload(self, path, run_id, cancel):
            calls["upload"] += 1
            return f"{run_id}/{path.name}"

        def submit(self, workflow, run_id, client_id):
            calls["submit"] += 1
            assert workflow["p"]["inputs"]["text"] == "A meadow."
            if self.uncertain:
                raise RuntimeError("connection lost after submission")
            return "job-123"

        def find_submission(self, run_id):
            return self.recovered

        def monitor(self, prompt_id, client_id, graph, cancel, emit):
            assert prompt_id == "job-123"
            return copy.deepcopy(history)

        def download(self, descriptor, target, cancel, callback):
            calls["download"] += 1
            if self.fail_download and descriptor["node_id"] == "shot":
                raise RuntimeError("interrupted download")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"video-bytes")
            return {
                "path": str(target),
                "size_bytes": 11,
                "sha256": hashlib.sha256(b"video-bytes").hexdigest(),
            }

        def close(self):
            pass

    def agent(config, folder, brief, cancel, emit):
        calls["agent"] += 1
        assert "text" not in brief["prompt_slots"][0]
        atomic_json(folder / "plan.json", plan)
        return copy.deepcopy(plan)

    monkeypatch.setattr(engine, "Comfy", FakeComfy)
    monkeypatch.setattr(engine, "prepare_workflow", lambda *a, **k: copy.deepcopy(prepared))
    monkeypatch.setattr(engine, "validate_graph", lambda *a: None)
    monkeypatch.setattr(engine.agents, "generate_plan", agent)
    monkeypatch.setattr(engine.agents, "executable", lambda _: "ffprobe")
    monkeypatch.setattr(
        engine,
        "probe_video",
        lambda p: {"size_bytes": p.stat().st_size, "width": 640, "height": 360, "duration": 15.0},
    )
    return settings, FakeComfy, calls, prepared, plan, history


def saved_folder(settings):
    return next(Path(settings.output_folder).iterdir())


def test_fork_creates_fresh_run_from_bound_saved_snapshot(tmp_path):
    assets = tmp_path / "assets"
    templates = tmp_path / "templates"
    assets.mkdir()
    templates.mkdir()
    source_file = templates / "workflow.json"
    source_file.write_text('{"original":true}')
    settings = Settings(project_folder=str(assets), template_folder=str(templates),
                        template_file=str(source_file), output_folder=str(tmp_path / "outputs"))
    source, original = engine.create_run(settings, "First concept")
    snapshot = b'{"original":true}\n'
    (source / "template.original.json").write_bytes(snapshot)
    sha = hashlib.sha256(snapshot).hexdigest()
    original.update(template_path=str(source_file), template_sha256=sha,
                    prompt_id="old-job", submission_started=True, workflow_edit_attempts=[{"number": 1}])
    atomic_json(source / "state.json", original)
    seed = {"folder": str(source), "run_id": original["run_id"], "template_sha256": sha,
            "template_file": settings.template_file, "template_folder": settings.template_folder}
    source_file.unlink()
    fork, state = engine.create_run(settings, "Revised concept", seed)
    assert fork != source and state["run_id"] != original["run_id"]
    assert (fork / "template.original.json").read_bytes() == snapshot
    assert state["forked_from"]["run_id"] == original["run_id"]
    assert not any(key in state for key in ("prompt_id", "submission_started", "workflow_edit_attempts", "managed_session"))
    (source / "template.original.json").write_text("{}")
    with pytest.raises(ValueError, match="snapshot changed"):
        engine.create_run(settings, "Another concept", seed)


def test_fork_uses_newly_selected_workflow_instead_of_old_snapshot(tmp_path):
    assets = tmp_path / "assets"
    templates = tmp_path / "templates"
    assets.mkdir()
    templates.mkdir()
    old = templates / "old.json"
    new = templates / "new.json"
    old.write_text('{"workflow":"old"}')
    new.write_text('{"workflow":"new"}')
    settings = Settings(project_folder=str(assets), template_folder=str(templates),
                        template_file=str(old), output_folder=str(tmp_path / "outputs"))
    source, state = engine.create_run(settings, "First")
    payload = b'{"workflow":"saved"}'
    (source / "template.original.json").write_bytes(payload)
    sha = hashlib.sha256(payload).hexdigest()
    state["template_sha256"] = sha
    atomic_json(source / "state.json", state)
    seed = {"folder": str(source), "run_id": state["run_id"], "template_sha256": sha,
            "template_file": settings.template_file, "template_folder": settings.template_folder}
    fork, fork_state = engine.create_run(replace(settings, template_file=str(new)), "Second", seed)
    assert not (fork / "template.original.json").exists()
    assert fork_state["forked_from"]["run_id"] == state["run_id"]


def test_complete_package_references_are_paths_only(scenario):
    settings, _, calls, _, _, _ = scenario
    events = []
    result = engine.run(settings, "A meadow", events.append, Event())
    folder = Path(result["folder"])
    assert Path(result["final_path"]).is_file()
    assert len(list((folder / "shots").rglob("*.mp4"))) == 1
    assert not list(folder.rglob("*.png"))
    prepared = json.loads((folder / "prepared.json").read_text())
    assert prepared["prompt_slots"][0]["text"] == ""
    assert prepared["graph"]["p"]["inputs"]["text"] == ""
    assert "reference.png" in (folder / "used_references.txt").read_text()
    assert calls["submit"] == calls["agent"] == 1
    assert calls["download"] == 2
    assert "SUCCESS" in (folder / "report.txt").read_text()
    state = json.loads((folder / "state.json").read_text())
    assert state["status"] == "complete"
    assert not Path(state["result"]["final_path"]).is_absolute()
    assert events[-1]["progress"] == 100


def test_resume_download_never_resubmits_or_rewrites_script(scenario):
    settings, fake, calls, *_ = scenario
    fake.fail_download = True
    with pytest.raises(RuntimeError, match="interrupted download"):
        engine.run(settings, "A meadow", lambda _: None, Event())
    folder = saved_folder(settings)
    script = (folder / "script.txt").read_bytes()
    fake.fail_download = False
    engine.run(
        Settings(comfy_url="https://wrong.invalid", cloud=True),
        "ignored",
        lambda _: None,
        Event(),
        folder,
    )
    assert calls["submit"] == calls["agent"] == 1
    assert calls["download"] == 3  # successful final is not fetched twice
    assert calls["endpoint"] == settings.comfy_url
    assert (folder / "script.txt").read_bytes() == script


def test_moved_completed_package_uses_its_own_files(scenario, tmp_path):
    settings, _, calls, *_ = scenario
    result = engine.run(settings, "A meadow", lambda _: None, Event())
    moved = tmp_path / "moved"
    shutil.copytree(result["folder"], moved)
    result = engine.run(settings, "ignored", lambda _: None, Event(), moved)
    assert Path(result["final_path"]).is_relative_to(moved)
    assert calls["download"] == 2
    assert calls["submit"] == 1


def test_corrupted_local_download_is_retrieved_again(scenario):
    settings, _, calls, *_ = scenario
    result = engine.run(settings, "A meadow", lambda _: None, Event())
    Path(result["final_path"]).write_bytes(b"wrong-bytes")
    engine.run(settings, "ignored", lambda _: None, Event(), Path(result["folder"]))
    assert calls["download"] == 3
    assert calls["submit"] == 1


def test_uncertain_submission_reconciles_without_second_post(scenario):
    settings, fake, calls, *_ = scenario
    fake.uncertain = True
    with pytest.raises(RuntimeError, match="connection lost"):
        engine.run(settings, "A meadow", lambda _: None, Event())
    folder = saved_folder(settings)
    result = engine.run(settings, "ignored", lambda _: None, Event(), folder)
    assert Path(result["final_path"]).is_file()
    assert calls["submit"] == 1


def test_uncertain_submission_missing_job_does_not_retry(scenario):
    settings, fake, calls, *_ = scenario
    fake.uncertain = True
    fake.recovered = None
    with pytest.raises(RuntimeError):
        engine.run(settings, "A meadow", lambda _: None, Event())
    with pytest.raises(RuntimeError, match="uncertain"):
        engine.run(settings, "ignored", lambda _: None, Event(), saved_folder(settings))
    assert calls["submit"] == 1


def test_preflight_before_agent_or_upload(scenario, monkeypatch):
    settings, _, calls, *_ = scenario

    def fail(*args):
        raise ValueError("missing model")

    monkeypatch.setattr(engine, "validate_graph", fail)
    with pytest.raises(RuntimeError, match="missing model"):
        engine.run(settings, "A meadow", lambda _: None, Event())
    assert calls["agent"] == calls["upload"] == calls["submit"] == 0
    assert "missing model" in (saved_folder(settings) / "report.txt").read_text()


def test_missing_shot_is_not_success(scenario):
    settings, _, _, _, _, history = scenario
    del history["outputs"]["shot"]
    with pytest.raises(RuntimeError, match="all saved shots"):
        engine.run(settings, "A meadow", lambda _: None, Event())


def test_cancel_before_external_work(scenario):
    settings, _, calls, *_ = scenario
    cancel = Event()
    cancel.set()
    with pytest.raises(RuntimeError, match="Cancelled"):
        engine.run(settings, "A meadow", lambda _: None, cancel)
    assert calls["submit"] == calls["agent"] == 0


def test_duplicate_resume_lock_is_rejected(scenario):
    settings, _, _, *_ = scenario
    result = engine.run(settings, "A meadow", lambda _: None, Event())
    folder = Path(result["folder"])
    with engine.run_lock(folder), pytest.raises(RuntimeError, match="already running"):
        engine.run(settings, "ignored", lambda _: None, Event(), folder)


def test_invalid_agent_reference_stays_inside_linked_root(scenario, tmp_path):
    settings, _, _, prepared, plan, _ = scenario
    secret = tmp_path / "secret.txt"
    secret.write_text("private")
    plan["references"][0]["path"] = str(secret)
    with pytest.raises(ValueError, match="outside"):
        engine.apply_plan(prepared, plan, [Path(settings.project_folder)])


def test_agent_must_fill_every_prompt(scenario):
    settings, _, _, prepared, plan, _ = scenario
    plan["prompts"] = []
    with pytest.raises(ValueError, match="every required prompt"):
        engine.apply_plan(prepared, plan, [Path(settings.project_folder)])


def test_prompt_brief_excludes_sample_prose():
    slots = [
        {
            "node_id": "1",
            "input": "text",
            "text": "overall_soundscape: copied sample scene",
            "overlap_seconds": 1,
        }
    ]
    brief = engine._prompt_brief(slots)
    assert "copied sample" not in json.dumps(brief)
    assert brief[0]["sections"] == ["overall_soundscape"]


def test_cloud_status_reconnect_without_provisioning(tmp_path):
    settings = Settings(managed_runpod=True)
    state = {"settings": settings.public_dict(), "managed_session": {}, "status": "created"}
    atomic_json(tmp_path / "state.json", state)

    assert engine.cloud_action(tmp_path, "status") == {"phase": "not_started"}
    saved = json.loads((tmp_path / "state.json").read_text())
    assert saved["cloud_status"]["phase"] == "not_started"


def test_completed_run_with_4k60_leaves_a_durable_upscale_request(scenario):
    from simplegen import upscale
    settings, *_ = scenario
    result = engine.run(replace(settings, upscale=True, upscale_url="http://upscaler:8190"), "A meadow",
                        lambda _event: None, Event())
    [record] = upscale.records()
    assert record["status"] == "pending" and record["origin"] == "auto"
    assert (Path(record["run_folder"]) / record["source"]).resolve() == Path(result["final_path"]).resolve()
    assert json.loads((Path(result["folder"]) / "state.json").read_text())["status"] == "complete"


def test_completed_run_without_4k60_requests_nothing(scenario):
    from simplegen import upscale
    settings, *_ = scenario
    engine.run(settings, "A meadow", lambda _event: None, Event())
    assert upscale.records() == []
