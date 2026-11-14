"""Entire run through a fake CLI and Comfy transport, with real video inspection."""

import json
import shutil
import subprocess
import sys
from pathlib import Path
from threading import Event

import httpx
import pytest
from test_workflows import info, workflow  # noqa: F401 -- shared synthetic pytest fixtures

from simplegen import agents, engine
from simplegen.comfy import Comfy
from simplegen.settings import Settings


@pytest.mark.parametrize("invalid_attempts,missing_model,schema", [
    (0, False, 5), (1, False, 5), (2, False, 5), (0, True, 5),
    (0, False, 4),  # Runs saved before schema 5 keep the agent-edited workflow path.
])
def test_concept_to_local_package_with_real_ffprobe(
    tmp_path,
    monkeypatch,
    info,  # noqa: F811 -- shared fixture
    workflow,  # noqa: F811 -- shared fixture
    invalid_attempts,
    missing_model,
    schema,
):
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("FFmpeg is required for the media integration test")
    video = tmp_path / "synthetic.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=blue:s=160x96:r=24",
            "-t",
            "0.5",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(video),
        ],
        check=True,
        timeout=20,
    )
    assets = tmp_path / "assets"
    assets.mkdir()
    templates = tmp_path / "templates"
    templates.mkdir()
    if missing_model:
        workflow["nodes"].append({"id": 20, "type": "UNETLoader", "widgets_values": ["missing.safetensors"]})
        info["UNETLoader"] = {
            "input": {"required": {"unet_name": [["installed.safetensors"]]}}, "output": ["MODEL"],
        }
    (templates / "synthetic.json").write_text(json.dumps(workflow))
    script = tmp_path / "agent.py"
    if schema == 4:
        monkeypatch.setattr(engine, "STATE_VERSION", 4)
    script.write_text(
        """import sys,json
from pathlib import Path
request = sys.stdin.read().split("RUN REQUEST (data, not instructions):\\n", 1)[1]
brief = json.loads(request.split("\\nReturn only the final JSON", 1)[0])
if brief.get("workflow_brief"):
    # Schema 5: read-only; return content only. The workspace is the attempt folder.
    attempt = int(Path.cwd().name.split("-")[-1])
    option = next(o for o in brief["timing_options"] if o["id"] == brief["recommended_option"])
    text = "A silent blue frame." if attempt > INVALID_ATTEMPTS else ""
    plan = {"script": "A silent blue frame.", "used_references": [], "timing_option": option["id"],
            "references": {}, "segments": [{"sections": {name: text for name in brief["prompt_sections"]}}
                                           for _ in range(option["segment_count"])]}
    print(json.dumps({"structured_output": plan}))
    sys.exit(0)
contract = brief["editing_contract"]
path = Path(brief["workflow_file"])
workflow = json.loads(path.read_text())
project = next(n for n in workflow["nodes"] if n["id"] == contract["project_node_id"])
for mapping in contract["project_widgets"]:
    if mapping["input"] in contract["resolution"]:
        while len(project["widgets_values"]) <= mapping["widget_index"]:
            project["widgets_values"].append(None)
        project["widgets_values"][mapping["widget_index"]] = contract["resolution"][mapping["input"]]
segment = next(n for n in workflow["nodes"] if n["id"] == contract["segment_node_ids"][0])
for mapping in contract["segment_widgets"]:
    if mapping["input"] == "duration_override":
        segment["widgets_values"][mapping["widget_index"]] = contract["timing"]["recommended_segments"][0]["duration_override"]
    if mapping["input"] == "prompt":
        segment["widgets_values"][mapping["widget_index"]] = "A silent blue frame."
attempt = int(path.parent.name.split("-")[-1])
if attempt <= INVALID_ATTEMPTS:
    workflow["nodes"][0]["type"] = "InventedNode"
else:
    workflow["nodes"][0]["type"] = "H3RCProject"
path.write_text(json.dumps(workflow))
print(json.dumps({"structured_output": {"script":"A silent blue frame.", "used_references":[]}}))
""".replace("INVALID_ATTEMPTS", str(invalid_attempts))
    )
    monkeypatch.setattr(agents, "build_command", lambda *a, **k: [sys.executable, str(script)])
    monkeypatch.setattr(agents, "_subscription", lambda *a: None)
    submitted = []

    def transport(request):
        if request.url.path == "/object_info":
            return httpx.Response(200, json=info)
        if request.url.path == "/prompt":
            submitted.append(json.loads(request.content))
            return httpx.Response(200, json={"prompt_id": "test-job"})
        if request.url.path == "/history/test-job":
            outputs = {
                node: {
                    "images": [{"filename": f"{node}.mp4", "subfolder": "test", "type": "output"}]
                }
                for node, value in submitted[0]["prompt"].items()
                if value["class_type"] == "SaveVideo"
            }
            return httpx.Response(
                200, json={"test-job": {"status": {"completed": True}, "outputs": outputs}}
            )
        if request.url.path == "/view":
            return httpx.Response(200, content=video.read_bytes())
        raise AssertionError(f"Unexpected endpoint: {request.method} {request.url.path}")

    monkeypatch.setattr(
        engine,
        "Comfy",
        lambda settings: Comfy(settings, httpx.Client(transport=httpx.MockTransport(transport))),
    )
    settings = Settings(
        project_folder=str(assets),
        template_folder=str(templates),
        output_folder=str(tmp_path / "outputs"),
        duration=5,
    )
    if invalid_attempts == 2:
        with pytest.raises(RuntimeError, match="after one correction"):
            engine.run(settings, "A quiet blue frame", lambda _: None, Event())
        assert not submitted
        folder = next(Path(settings.output_folder).iterdir())
        with pytest.raises(RuntimeError, match="after one correction"):
            engine.run(settings, "ignored", lambda _: None, Event(), folder)
        assert not submitted
        assert len(list((folder / "agent-edits").iterdir())) == 2
        return
    if missing_model:
        with pytest.raises(RuntimeError, match="Story, prompts and workflow copy are saved"):
            engine.run(settings, "A quiet blue frame", lambda _: None, Event())
        folder = next(Path(settings.output_folder).iterdir())
        assert not submitted
        assert (folder / "script.txt").read_text() == "A silent blue frame."
        prepared = json.loads((folder / "workflow.json").read_text())
        assert next(n for n in prepared["nodes"] if n["id"] == 20)["widgets_values"] == ["missing.safetensors"]
        assert json.loads((folder / "state.json").read_text())["workflow_acceptance"]["attempt"] == 1

        def no_more_agent(*args, **kwargs):
            pytest.fail("Resume must reuse the saved story and prompts without more agent usage")

        monkeypatch.setattr(agents, "generate_plan", no_more_agent)
        with pytest.raises(RuntimeError, match="Rendering is blocked"):
            engine.run(settings, "ignored", lambda _: None, Event(), folder)
        assert not submitted
        info["UNETLoader"]["input"]["required"]["unet_name"][0].append("missing.safetensors")
        result = engine.run(settings, "ignored", lambda _: None, Event(), folder)
    else:
        result = engine.run(settings, "A quiet blue frame", lambda _: None, Event())
    assert result["metadata"]["width"] == 160
    assert result["metadata"]["height"] == 96
    assert result["metadata"]["duration"] == pytest.approx(0.5)
    assert len(list(Path(result["folder"]).rglob("*.mp4"))) == 2
    assert len(submitted) == 1
    assert submitted[0]["prompt"]["2:3"]["inputs"]["value"] == "A silent blue frame."
    assert "SUCCESS" in (Path(result["folder"]) / "report.txt").read_text()
    assert json.loads((templates / "synthetic.json").read_text()) == workflow
    folder = Path(result["folder"])
    assert json.loads((folder / "state.json").read_text())["schema_version"] == schema
    assert json.loads((folder / "validation.json").read_text())["attempt"] == invalid_attempts + 1
    assert (
        json.loads((folder / "workflow.json").read_text())["definitions"] == workflow["definitions"]
    )
    engine.run(settings, "ignored", lambda _: None, Event(), folder)
    assert len(submitted) == 1
