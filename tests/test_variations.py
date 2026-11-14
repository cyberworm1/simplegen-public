"""Sequential execution and recovery with synthetic workflows and offline transports."""

import copy
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from threading import Event

import pytest
from test_workflows import info, schema, workflow  # noqa: F401 -- synthetic fixtures

from simplegen import agents, engine, managed, resources
from simplegen.lifecycle import Detached
from simplegen.settings import Settings, atomic_json
from simplegen.variations import variant_workflow


@pytest.fixture
def sequence(tmp_path, monkeypatch, info, workflow):  # noqa: F811
    monkeypatch.setenv("SIMPLEGEN_HOME", str(tmp_path / "home"))
    assets, templates = tmp_path / "assets", tmp_path / "templates"
    assets.mkdir()
    templates.mkdir()
    image = assets / "reference.png"
    image.write_bytes(b"synthetic reference")
    info["LoadImage"] = schema({"image": [["reference.png"], {"image_upload": True}]}, ["IMAGE"])
    workflow["nodes"].append({"id": 10, "type": "LoadImage", "widgets_values": ["reference.png"]})
    atomic_json(templates / "template.json", workflow)
    settings = Settings(project_folder=str(assets), template_folder=str(templates),
                        template_file="template.json", output_folder=str(tmp_path / "runs"),
                        quantity=3, duration=4)
    calls, jobs, trace, acknowledged = [], {}, [], []
    fault = {"mode": "", "fired": False}
    detach = Event()

    def agent(config, folder, brief, cancel, emit):
        calls.append(brief)
        assert brief["quantity"] == 3
        native = json.loads(Path(brief["workflow_file"]).read_text())
        contract = brief["editing_contract"]
        project = next(n for n in native["nodes"] if n["id"] == contract["project_node_id"])
        for mapping in contract["project_widgets"]:
            if mapping["input"] in contract["resolution"]:
                project["widgets_values"][mapping["widget_index"]] = contract["resolution"][mapping["input"]]
        segment = next(n for n in native["nodes"] if n["type"] == contract["segment_type"])
        for mapping in contract["segment_widgets"]:
            if mapping["input"] == "prompt":
                segment["widgets_values"][mapping["widget_index"]] = "First take"
            if mapping["input"] == "duration_override":
                segment["widgets_values"][mapping["widget_index"]] = contract["timing"]["recommended_segments"][0]["duration_override"]
        next(n for n in native["nodes"] if n["type"] == "LoadImage")["widgets_values"] = [str(image)]
        plan = {"script": "First script", "used_references": [str(image)], "variations": [
            {"script": f"Script {i}", "prompts": [{"node_id": str(segment["id"]), "text": f"Take {i}"}]}
            for i in (2, 3)]}
        if fault["mode"] == "invalid_prompts":
            plan["variations"][0]["prompts"] = []
        atomic_json(Path(brief["workflow_file"]), native)
        atomic_json(folder / "plan.json", plan)
        return plan

    class Comfy:
        def __init__(self, config):
            pass

        def object_info(self):
            return copy.deepcopy(info)

        def upload(self, source, run_id, cancel):
            trace.append(("upload", str(source)))
            return "uploaded.png"

        def submit(self, graph, run_id, client_id):
            prompt_id = f"job-{len(jobs) + 1}"
            jobs[prompt_id] = {"run_id": run_id, "graph": copy.deepcopy(graph)}
            trace.append(("submit", prompt_id))
            if len(jobs) == 2 and fault["mode"] == "uncertain" and not fault["fired"]:
                fault["fired"] = True
                raise RuntimeError("response lost after submission")
            return prompt_id

        def find_submission(self, run_id):
            return next((key for key, value in jobs.items() if value["run_id"] == run_id), None)

        def monitor(self, prompt_id, client_id, graph, cancel, emit, **kwargs):
            trace.append(("monitor", prompt_id))
            if prompt_id == "job-2" and fault["mode"] in ("monitor", "detach") and not fault["fired"]:
                fault["fired"] = True
                if fault["mode"] == "detach":
                    detach.set()
                    raise Detached("desktop disconnected")
                raise RuntimeError("proxy unavailable")
            outputs = {}
            for node_id, value in graph.items():
                if value["class_type"] == "SaveVideo":
                    prefix = Path(value["inputs"]["filename_prefix"])
                    outputs[node_id] = {"videos": [{"filename": prefix.name + "_00001.mp4",
                                                   "subfolder": str(prefix.parent), "type": "output"}]}
            return {"status": {"completed": True}, "outputs": outputs}

        def download(self, descriptor, target, cancel, on_progress):
            trace.append(("download", descriptor["subfolder"]))
            target.parent.mkdir(parents=True, exist_ok=True)
            payload = b"synthetic-video"
            target.write_bytes(payload)
            return {"path": str(target), "size_bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}

        def close(self):
            pass

    class Manager:
        retired = False

        def __init__(self, config, folder, state, persist, emit, cancel, detach):
            self.settings, self.folder, self.state, self.persist = config, folder, state, persist

        def schema(self):
            return copy.deepcopy(info)

        def config(self):
            return object()

        def connect(self, choice=None):
            if not self.state.get("managed_session", {}).get("sid"):
                assert choice == {"gpu_id": "offline"}
                self.state["managed_session"] = {"sid": "one-session"}
                self.persist()
                trace.append(("provision", "one-session"))
            return self.settings

        def before_submit(self, job_id):
            trace.append(("register", job_id))
            return True

        def bind(self, prompt_id, job_id):
            assert jobs[prompt_id]["run_id"] == job_id

        def complete(self, receipts):
            assert len(jobs) == 3
            assert len(receipts) == 6
            for child in self.state["sequence"]["children"]:
                assert json.loads((self.folder / child["folder"] / "state.json").read_text())["status"] == "complete"
            assert all(hashlib.sha256(Path(r["path"]).read_bytes()).hexdigest() == r["sha256"] for r in receipts)
            acknowledged.append(receipts)

        def close(self):
            pass

    monkeypatch.setattr(agents, "generate_plan", agent)
    # This fake agent edits workflow JSON: the legacy contract that schema 2-4 runs keep.
    monkeypatch.setattr(engine, "STATE_VERSION", 4)
    monkeypatch.setattr(agents, "executable", lambda _: "ffprobe")
    monkeypatch.setattr(engine, "Comfy", Comfy)
    monkeypatch.setattr(engine, "probe_video", lambda p: {"width": 160, "height": 96, "duration": 4, "size_bytes": p.stat().st_size})
    monkeypatch.setattr(managed, "ManagedSession", Manager)
    monkeypatch.setattr(resources, "gpu_options", lambda *a: [{"gpu_id": "offline"}])
    from simplegen import credentials
    monkeypatch.setattr(credentials, "load_credentials", lambda _: {})
    events = []

    def run(resume=None, cloud=False):
        return engine.run(replace(settings, managed_runpod=cloud), "Three takes", events.append,
                          Event(), resume_folder=resume, detach=detach, choose_gpu=lambda options: options[0])

    return run, calls, jobs, trace, events, fault, detach, acknowledged, tmp_path, workflow


@pytest.mark.parametrize("cloud", [False, True])
def test_one_agent_preparation_and_serial_verified_packages(sequence, cloud):
    run, calls, jobs, trace, _events, _fault, _detach, acknowledged, root, original = sequence
    result = run(cloud=cloud)
    assert len(calls) == 1 and len(jobs) == 3
    assert [op for op, _ in trace].count("upload") == 1
    # Download final + shot before each next submission.
    execution = [op for op, _ in trace if op in ("submit", "download")]
    assert execution == ["submit", "download", "download"] * 3
    assert len(result["variations"]) == 3
    folder = Path(result["folder"])
    assert json.loads((root / "templates/template.json").read_text()) == original
    prefixes = [j["graph"]["3"]["inputs"]["filename_prefix"] for j in jobs.values()]
    assert len(set(prefixes)) == 3
    prompts = [j["graph"]["2:3"]["inputs"]["value"] for j in jobs.values()]
    assert prompts == ["First take", "Take 2", "Take 3"]
    assert all((folder / entry["final_path"]).exists() for entry in result["variations"])
    if cloud:
        assert len(acknowledged) == 1
        assert [op for op, _ in trace].count("provision") == 1
        assert [op for op, _ in trace].count("register") == 3
    run(resume=folder)
    assert len(calls) == 1 and len(jobs) == 3


@pytest.mark.parametrize("mode", ["monitor", "uncertain", "detach"])
def test_resume_retains_finished_variations_and_prompt_identity(sequence, mode):
    run, calls, jobs, trace, events, fault, detach, _acknowledged, *_ = sequence
    fault["mode"] = mode
    with pytest.raises(RuntimeError):
        run()
    folder = Path(events[-1]["folder"])
    assert len(jobs) == 2
    assert len(json.loads((folder / "state.json").read_text())["result"]["variations"]) == 1
    detach.clear()
    # Opening a child resumes the parent and the remaining set.
    result = run(resume=folder / "variation-002")
    assert len(calls) == 1 and len(jobs) == 3 and len(result["variations"]) == 3
    assert [key for op, key in trace if op == "submit"] == ["job-1", "job-2", "job-3"]


def test_cloud_failure_never_acknowledges_partial_set(sequence):
    run, _calls, _jobs, trace, events, fault, _detach, acknowledged, *_ = sequence
    fault["mode"] = "monitor"
    with pytest.raises(RuntimeError):
        run(cloud=True)
    assert acknowledged == []
    run(resume=Path(events[-1]["folder"]))
    assert len(acknowledged) == 1
    assert [op for op, _ in trace].count("provision") == 1


def test_incomplete_prompt_sets_fail_before_any_submission(sequence):
    run, calls, jobs, trace, _events, fault, *_ = sequence
    fault["mode"] = "invalid_prompts"
    with pytest.raises(RuntimeError, match="every native segment"):
        run()
    assert len(calls) == 2 and not jobs and not trace


def test_copy_changes_only_prompt_and_seed():
    node = {"id": 2, "type": "segment", "inputs": [{"name": key, "widget": {"name": key}}
            for key in ("prompt", "seed", "steps_override")], "widgets_values": ["base", 42, 20]}
    original = {"nodes": [node], "links": [], "definitions": {"unchanged": True}}
    result = variant_workflow(original, [{"node_id": "2", "text": "new"}], "segment", "run-2")
    assert original["nodes"][0]["widgets_values"] == ["base", 42, 20]
    assert result["nodes"][0]["widgets_values"][0] == "new"
    assert result["nodes"][0]["widgets_values"][1] != 42
    assert 0 <= result["nodes"][0]["widgets_values"][1] <= 2**53 - 1
    result["nodes"][0]["widgets_values"] = ["base", 42, 20]
    assert result == original


def test_retired_manifest_filters_jobs_and_maps_saved_video_nodes():
    graph = {"3": {"class_type": "SaveVideo", "inputs": {"filename_prefix": "simpleGen/one/final"}}}
    history = engine._history_from_manifest({"manifest": [
        {"prompt_id": "job1", "filename": "final_00001.mp4", "subfolder": "simpleGen/one"},
        {"prompt_id": "job2", "filename": "final_00001.mp4", "subfolder": "simpleGen/two"},
    ]}, graph, "job1")
    assert engine.output_files(history) == [{"node_id": "3", "filename": "final_00001.mp4",
                                            "subfolder": "simpleGen/one", "type": "output"}]


def test_quantity_response_schema_includes_all_variation_prompts():
    schema = agents.plan_schema(True, 4)
    variants = schema["properties"]["variations"]
    assert variants["minItems"] == variants["maxItems"] == 3
    assert set(variants["items"]["required"]) == {"script", "prompts"}
    assert "variations" not in agents.plan_schema(True)["properties"]


def test_scheduled_sequence_waits_for_queue_before_each_submission(sequence, monkeypatch):
    from datetime import UTC, datetime, timedelta
    run, _calls, _jobs, trace, *_ = sequence
    original_create = engine.create_run

    def create(*args):
        folder, state = original_create(*args)
        state["schedule"] = {"released": True, "due_at": (datetime.now(UTC) - timedelta(seconds=1)).isoformat()}
        atomic_json(folder / "state.json", state)
        return folder, state

    observations = [True, False, False, False]

    def queue(_self):
        busy = observations.pop(0)
        trace.append(("busy" if busy else "idle", "queue"))
        return {"queue_running": ["other-job"] if busy else [], "queue_pending": []}

    monkeypatch.setattr(engine, "create_run", create)
    monkeypatch.setattr(engine.Comfy, "queue", queue, raising=False)
    monkeypatch.setattr(engine.Interrupt, "wait", lambda *a: None)
    run()
    for index, (operation, _value) in enumerate(trace):
        if operation == "submit":
            assert trace[index - 1][0] == "idle"
    assert not observations
