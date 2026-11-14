"""Phase 2: repair model selections and reschedule without another agent pass."""

import hashlib
import json
import os
import shutil
import threading
import time
from datetime import UTC, datetime, timedelta
from functools import partial
from pathlib import Path
from threading import Event

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import httpx
import pytest
from conftest import PASSED

from simplegen import agents, engine, preflight, refcache, repair
from simplegen.preflight import run_preflight, run_saved_preflight, schema_hash
from simplegen.repair import RepairRefused
from simplegen.scheduling import Scheduler
from simplegen.settings import Settings

REAL = Path(__file__).parent / "fixtures" / "real"
BUNDLED = Path(__file__).parents[1] / "src" / "simplegen" / "data" / "workflows"
INT8_UNET = "minimax_h3_ref2va_pruned_int8_convrot.safetensors"
NVFP4_CLIP = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
BF16_UNET = "minimax_h3_ref2va_bf16.safetensors"


def tools(name):
    return f"/usr/bin/{name}"


@pytest.fixture(scope="module")
def schema():
    return json.loads((REAL / "schemas" / "local_object_info.json").read_text())


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("SIMPLEGEN_HOME", str(tmp_path / "home"))
    (tmp_path / "references").mkdir()
    shutil.copytree(BUNDLED, tmp_path / "templates")
    # The fixture's references live on a network share; revisions pin a local placeholder.
    placeholder = tmp_path / "references" / "ref.png"
    placeholder.write_bytes(b"png")
    monkeypatch.setattr(engine, "_inside", lambda path, roots: placeholder)
    return tmp_path


def blocked_run(home, **state_changes) -> Path:
    """Copy the real blocked BF16 run with folders pointing inside the test home."""
    run = home / "out" / "blocked"
    shutil.copytree(REAL / "runs" / "blocked_bf16_model_on_local", run)
    state = json.loads((run / "state.json").read_text())
    state["settings"].update(project_folder=str(home / "references"), output_folder=str(home / "out"),
                             template_folder=str(home / "templates"),
                             template_file=str(home / "templates" / "H3_Persistent_References_Clonable_Chain.json"))
    state.update(state_changes)
    (run / "state.json").write_text(json.dumps(state))
    return run


def hashes(folder: Path) -> dict:
    return {str(p.relative_to(folder)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(folder.rglob("*")) if p.is_file() and p.name != ".run.lock"}


def fix(info) -> dict:
    wanted = {"unet_name": INT8_UNET, "clip_name": NVFP4_CLIP}
    return {s["key"]: wanted[s["input"]] for s in info["selections"] if not s["available"]}


def published(home) -> list[Path]:
    return [p for p in (home / "out").iterdir() if p.is_dir() and p.name != "blocked" and not p.name.startswith(".")]


# 1. Regression


def test_repair_real_blocked_run(home, schema):
    run = blocked_run(home)
    before = hashes(run)
    info = repair.inspect(run, object_info=schema)
    missing = {(s["class_type"], s["input"], s["current"]) for s in info["selections"] if not s["available"]}
    assert missing == {("UNETLoader", "unet_name", BF16_UNET),
                       ("CLIPLoader", "clip_name", "qwen3vl_32b_h3_ultra_uncensored_heretic_bf16.safetensors")}
    assert all(INT8_UNET in s["choices"] for s in info["selections"] if s["input"] == "unet_name")
    result = repair.apply(run, fix(info), object_info=schema, resolve=tools)
    assert result["status"] == "published"
    revision = Path(result["folder"])
    assert run_saved_preflight(revision, object_info=schema, resolve=tools)["status"] == "pass"
    assert hashes(run) == before  # Source is byte-identical.
    assert {p.name for p in run.iterdir()} - {".run.lock"} == {k.split("/")[0] for k in before}
    state = json.loads((revision / "state.json").read_text())
    origin = state["repaired_from"]
    assert origin["run_id"] == "1123b6ab077e4829b736e75b99b19059" and origin["folder"] == str(run.resolve())
    assert {(c["input"], c["old"], c["new"]) for c in origin["changes"]} == {
        ("unet_name", BF16_UNET, INT8_UNET),
        ("clip_name", "qwen3vl_32b_h3_ultra_uncensored_heretic_bf16.safetensors", NVFP4_CLIP)}
    assert origin["old_workflow_sha256"] == json.loads((run / "state.json").read_text())[
        "workflow_acceptance"]["workflow_sha256"]
    assert origin["new_workflow_sha256"] == hashlib.sha256((revision / "workflow.json").read_bytes()).hexdigest()
    assert origin["new_template_sha256"] == state["template_sha256"]
    assert state["workflow_acceptance"]["provenance"] == "repair, no agent pass"
    assert state["repair_preflight"]["status"] == "pass"
    assert state["settings"] == json.loads((run / "state.json").read_text())["settings"]
    for name in ("plan.json", "script.txt", "used_references.txt"):
        assert (revision / name).read_bytes() == (run / name).read_bytes()
    assert not (revision / "agent-edits").exists()
    # Phase 6: the revision pins its references (captured now; the source predates the cache).
    manifest = json.loads((revision / "references.json").read_text())
    assert manifest["references"] and refcache.pinned(state["run_id"]) and refcache.complete(revision)
    assert not (run / "references.json").exists()
    # Only the model selectors changed, in both snapshot and workflow.
    for name in ("workflow.json", "template.original.json"):
        old, new = (run / name).read_text(), (revision / name).read_text()
        old_json, new_json = json.loads(old), json.loads(new)
        assert new_json != old_json
        assert json.dumps(new_json).replace(INT8_UNET, BF16_UNET).replace(
            NVFP4_CLIP, "qwen3vl_32b_h3_ultra_uncensored_heretic_bf16.safetensors") == json.dumps(old_json)


# 2. Running a repaired project never calls the agent


class Submitted(Exception):
    pass


def fake_comfy(schema, calls):
    class FakeComfy:
        def __init__(self, settings, client=None):
            pass

        def object_info(self):
            info = json.loads(json.dumps(schema))
            for spec in info.values():
                for section in spec.get("input", {}).values():
                    image = section.get("image") if isinstance(section, dict) else None
                    if image and isinstance(image[0], list):
                        image[0].extend(calls["uploaded"])
            return info

        def queue(self):
            return {"queue_running": [], "queue_pending": []}

        def upload(self, path, run_id, cancel):
            name = f"{run_id}/{path.name}"
            calls["uploaded"].append(name)
            return name

        def submit(self, workflow, run_id, client_id):
            calls["submitted"] = workflow
            raise Submitted("fake submit reached")

        def find_submission(self, run_id):
            return None

        def close(self):
            pass
    return FakeComfy


def no_agent(monkeypatch, home):
    reference = home / "references" / "ref.png"
    reference.write_bytes(b"png")
    monkeypatch.setattr(agents, "generate_plan", lambda *a, **k: pytest.fail("agent CLI must not run"))
    monkeypatch.setattr(agents, "executable",
                        lambda name: tools(name) if name == "ffprobe" else pytest.fail("agent CLI must not run"))
    # The fixture's references live on a network share.
    monkeypatch.setattr(engine, "_inside", lambda path, roots: reference)


def test_running_repaired_project_reuses_accepted_work(home, schema, monkeypatch):
    run = blocked_run(home)
    result = repair.apply(run, fix(repair.inspect(run, object_info=schema)), object_info=schema, resolve=tools)
    no_agent(monkeypatch, home)
    calls = {"uploaded": []}
    monkeypatch.setattr(engine, "Comfy", fake_comfy(schema, calls))
    with pytest.raises(RuntimeError, match="fake submit reached"):
        engine.run(Settings(), "", lambda event: None, Event(), resume_folder=Path(result["folder"]))
    graph = calls["submitted"]
    assert graph["1"]["inputs"]["unet_name"] == INT8_UNET
    assert graph["2"]["inputs"]["clip_name"] == NVFP4_CLIP
    state = json.loads((Path(result["folder"]) / "state.json").read_text())
    assert state["submission_started"] and not state.get("workflow_edit_attempts")


# 3. Refusals


@pytest.mark.parametrize(("changes", "message"), [
    ({"prompt_id": "job-1"}, "already submitted"),
    ({"submission_started": True}, "already submitted"),
    ({"batch_parent": ".."}, "batch"),
    ({"sequence_parent": ".."}, "not yet supported"),
    ({"sequence": {"children": []}}, "not yet supported"),
    ({"workflow_acceptance": None}, "no accepted workflow"),
])
def test_ineligible_runs_are_refused(home, schema, changes, message):
    run = blocked_run(home, **changes)
    before = hashes(run)
    with pytest.raises(RepairRefused, match=message):
        repair.inspect(run, object_info=schema)
    with pytest.raises(RepairRefused, match=message):
        repair.apply(run, {"|1|unet_name": INT8_UNET}, object_info=schema, resolve=tools)
    assert hashes(run) == before and published(home) == []


def test_qty_greater_than_one_and_locked_runs_are_refused(home, schema):
    run = blocked_run(home)
    state = json.loads((run / "state.json").read_text())
    state["settings"]["quantity"] = 2
    (run / "state.json").write_text(json.dumps(state))
    with pytest.raises(RepairRefused, match="Qty > 1"):
        repair.apply(run, {"|1|unet_name": INT8_UNET}, object_info=schema, resolve=tools)
    state["settings"]["quantity"] = 1
    (run / "state.json").write_text(json.dumps(state))
    with engine.run_lock(run), pytest.raises(RuntimeError, match="already running"):
        repair.apply(run, {"|1|unet_name": INT8_UNET}, object_info=schema, resolve=tools)
    assert published(home) == []


def test_changed_accepted_work_is_refused(home, schema):
    run = blocked_run(home)
    (run / "plan.json").write_text("{}")
    with pytest.raises(RepairRefused, match="changed after validation"):
        repair.inspect(run, object_info=schema)


# 4. Only the target's choices; nothing automatic


def test_replacement_must_be_a_target_choice(home, schema):
    run = blocked_run(home)
    info = repair.inspect(run, object_info=schema)
    unet = next(s for s in info["selections"] if s["input"] == "unet_name")
    with pytest.raises(RepairRefused, match="not one of the target's available models"):
        repair.apply(run, {unet["key"]: "invented.safetensors"}, object_info=schema, resolve=tools)
    with pytest.raises(RepairRefused, match="Unknown model selection"):
        repair.apply(run, {"|99|unet_name": INT8_UNET}, object_info=schema, resolve=tools)
    with pytest.raises(RepairRefused, match="at least one"):
        repair.apply(run, {}, object_info=schema, resolve=tools)
    # Fixing only the UNET leaves the CLIP unavailable: nothing is substituted, so preflight fails.
    result = repair.apply(run, {unet["key"]: INT8_UNET}, object_info=schema, resolve=tools)
    assert result["status"] == "fail" and "clip_name" in preflight.summary(result["preflight"])
    assert published(home) == []


def test_unreadable_target_schema_is_not_compatible(home):
    run = blocked_run(home)

    def refuse(request):
        raise httpx.ConnectError("refused", request=request)
    client = httpx.Client(transport=httpx.MockTransport(refuse))
    info = repair.inspect(run, comfy_client=client)
    assert info["selections"] == [] and info["error"]
    with pytest.raises(RepairRefused, match="cannot be verified"):
        repair.apply(run, {"|1|unet_name": INT8_UNET}, comfy_client=client, resolve=tools)


# 5. Crash safety


def test_crash_at_each_write_step_never_publishes_partial(home, schema, monkeypatch):
    run = blocked_run(home)
    replacements = fix(repair.inspect(run, object_info=schema))
    original = repair._write
    steps = [0]

    def counting(path, data):
        steps[0] += 1
        original(path, data)
    monkeypatch.setattr(repair, "_write", counting)
    first = repair.apply(run, replacements, object_info=schema, resolve=tools)
    total = steps[0]
    assert total >= 7
    shutil.rmtree(first["folder"])
    for crash_at in range(1, total + 1):
        count = [0]

        def crashing(path, data, crash_at=crash_at, count=count):
            count[0] += 1
            if count[0] == crash_at:
                raise OSError("power lost")
            original(path, data)
        monkeypatch.setattr(repair, "_write", crashing)
        with pytest.raises(OSError, match="power lost"):
            repair.apply(run, replacements, object_info=schema, resolve=tools)
        assert published(home) == []

    def crashing_rename(*args):
        raise OSError("power lost")
    monkeypatch.setattr(repair, "_write", original)
    monkeypatch.setattr(repair.os, "rename", crashing_rename)
    with pytest.raises(OSError):
        repair.apply(run, replacements, object_info=schema, resolve=tools)
    assert published(home) == []
    monkeypatch.undo()
    monkeypatch.setenv("SIMPLEGEN_HOME", str(home / "home"))
    monkeypatch.setattr(engine, "_inside", lambda path, roots: home / "references" / "ref.png")
    assert list((home / "out").glob(repair.STAGING_PREFIX + "*"))  # Leftovers are hidden and unpublished.
    result = repair.apply(run, replacements, object_info=schema, resolve=tools)
    assert published(home) == [Path(result["folder"])]
    assert not list((home / "out").glob(repair.STAGING_PREFIX + "*"))  # Cleaned by the next repair.
    assert run_saved_preflight(Path(result["folder"]), object_info=schema, resolve=tools)["status"] == "pass"


# 6. Repair of a repair


def test_repair_of_a_repair_keeps_the_chain(home, schema):
    run = blocked_run(home)
    first = repair.apply(run, fix(repair.inspect(run, object_info=schema)), object_info=schema, resolve=tools)
    info = repair.inspect(first["folder"], object_info=schema)
    clip = next(s for s in info["selections"] if s["input"] == "clip_name")
    assert clip["available"] and clip["current"] == NVFP4_CLIP
    other = next(c for c in clip["choices"] if c != NVFP4_CLIP)
    second = repair.apply(first["folder"], {clip["key"]: other}, object_info=schema, resolve=tools)
    assert second["status"] == "published"
    chain = json.loads((Path(second["folder"]) / "state.json").read_text())["repaired_from"]
    first_state = json.loads((Path(first["folder"]) / "state.json").read_text())
    assert chain["run_id"] == first["run_id"] and chain["changes"][0]["new"] == other
    assert chain["previous"] == first_state["repaired_from"]
    assert chain["previous"]["run_id"] == "1123b6ab077e4829b736e75b99b19059"


# 7. fail blocks; unverified needs a recorded override


def test_fail_blocks_and_unverified_needs_recorded_override(home, schema, monkeypatch):
    run = blocked_run(home)
    replacements = fix(repair.inspect(run, object_info=schema))

    def missing(name):
        raise ValueError(f"{name} is not installed")
    result = repair.apply(run, replacements, object_info=schema, resolve=missing)
    assert result["status"] == "fail" and published(home) == []

    state = json.loads((run / "state.json").read_text())
    state["settings"]["project_folder"] = str(home / "unmounted")
    (run / "state.json").write_text(json.dumps(state))
    # Phase 6: a revision with a complete reference cache doesn't need the mount.
    cached = repair.apply(run, replacements, object_info=schema, resolve=tools)
    assert cached["status"] == "published" and cached["preflight"]["status"] == "pass"
    shutil.rmtree(cached["folder"])
    # Without a complete cache the unmounted folder is unverified and needs a recorded override.
    from simplegen import refcache
    monkeypatch.setattr(refcache, "complete", lambda *a, **k: False)
    result = repair.apply(run, replacements, object_info=schema, resolve=tools)
    assert result["status"] == "unverified" and published(home) == []
    result = repair.apply(run, replacements, object_info=schema, resolve=tools, override_unverified=True)
    assert result["status"] == "published"
    saved = json.loads((Path(result["folder"]) / "state.json").read_text())["repair_preflight"]
    assert saved["status"] == "unverified" and saved["override_unverified"] is True


# 8. Scheduled attention → repair → schedule revision → dispatch without agent


def test_scheduled_attention_repair_and_reschedule(home, schema, monkeypatch):
    clock = [datetime.now(UTC).timestamp()]
    scheduler = Scheduler(home / "schedules", clock=lambda: clock[0])
    run = blocked_run(home)
    source_state = json.loads((run / "state.json").read_text())
    record = {"schema_version": 1, "id": source_state["run_id"], "folder": str(run), "concept": "Lighthouse",
              "due_at": source_state["schedule"]["due_at"], "selected_local": source_state["schedule"]["due_at"],
              "status": "attention", "started": True, "next_retry_at": source_state["schedule"]["due_at"],
              "retry_count": 0, "note": "Rendering is blocked"}
    scheduler._save(record)
    result = repair.apply(run, fix(repair.inspect(run, object_info=schema)), object_info=schema, resolve=tools)
    revision = Path(result["folder"])
    with pytest.raises(ValueError, match="not repaired from"):
        scheduler.mark_repaired(record["id"], run, source_state["run_id"])
    scheduler.mark_repaired(record["id"], revision, result["run_id"])
    with pytest.raises(ValueError, match="already repaired"):
        scheduler.mark_repaired(record["id"], revision, result["run_id"])
    with pytest.raises(ValueError, match="repaired revision"):
        scheduler.change(record["id"], "now")
    due = datetime.fromtimestamp(clock[0] + 3600, UTC)
    with pytest.raises(ValueError, match="Preflight failed"):
        scheduler.add_prepared(revision, due, preflight=dict(PASSED, status="fail"))
    new = scheduler.add_prepared(revision, due, preflight=result["preflight"], source_record=record["id"])
    with pytest.raises(ValueError, match="already scheduled"):
        scheduler.add_prepared(revision, due, preflight=result["preflight"])
    original = scheduler.get(record["id"])
    assert original["status"] == "repaired" and original["repaired_to"] == {
        "run_id": result["run_id"], "folder": str(revision), "at": original["repaired_to"]["at"],
        "record_id": new["id"]}
    assert new["prepared"] and new["id"] == result["run_id"]
    assert len(list(home.joinpath("out").iterdir())) == 3  # source, revision, repair lock; no create_run

    clock[0] += 7200
    due_now = scheduler.tick()
    assert [r["id"] for r in due_now] == [new["id"]]  # The repaired original never dispatches.
    assert scheduler.get(record["id"])["status"] == "repaired"
    recheck = run_saved_preflight(revision, object_info=schema, resolve=tools)
    assert scheduler.claim(new["id"], recheck) == revision

    no_agent(monkeypatch, home)
    calls = {"uploaded": []}
    monkeypatch.setattr(engine, "Comfy", fake_comfy(schema, calls))
    monkeypatch.setattr(engine, "datetime", type("Later", (datetime,), {
        "now": classmethod(lambda cls, tz=None: datetime.fromtimestamp(clock[0], tz))}))
    with pytest.raises(RuntimeError, match="fake submit reached"):
        engine.run(Settings(), "", lambda event: None, Event(), resume_folder=revision)
    assert calls["submitted"]["1"]["inputs"]["unet_name"] == INT8_UNET
    scheduler.close()


def test_add_prepared_rejects_unprepared_projects(home, schema):
    scheduler = Scheduler(home / "schedules")
    run = blocked_run(home, prompt_id="job-1")
    with pytest.raises(ValueError, match="already submitted"):
        scheduler.add_prepared(run, datetime.now(UTC) + timedelta(hours=1), preflight=PASSED)
    assert scheduler.records() == []
    scheduler.close()


# 10. Manual cloud endpoints use the live schema


def test_manual_cloud_endpoint_uses_live_schema(home, schema):
    (home / "cloud.json").write_text(json.dumps({"UNETLoader": {"input": {"required": {}}}}))
    served = []

    def respond(request):
        served.append(str(request.url))
        return httpx.Response(200, json=schema)
    client = httpx.Client(transport=httpx.MockTransport(respond))
    settings = Settings(project_folder=str(home / "references"), template_folder=str(home / "templates"),
                        template_file="H3_Persistent_References_Clonable_Chain_LOCAL_INT8_NVFP4.json",
                        output_folder=str(home / "out"), steps=25, aspect_ratio="16:9",
                        runpod_schema_file=str(home / "cloud.json"), cloud=True, managed_runpod=False,
                        comfy_url="https://my-endpoint.example")
    result = run_preflight(settings, comfy_client=client, resolve=tools)
    assert result["target"] == "cloud" and result["status"] == "pass", preflight.summary(result)
    assert served == ["https://my-endpoint.example/object_info"]
    assert result["schema_source"] == "https://my-endpoint.example/object_info"
    assert result["schema_sha256"] == schema_hash(schema)


# 11. Hung filesystem probes


def test_hung_path_starts_only_one_probe_thread(home, monkeypatch):
    release = threading.Event()
    started = []
    original = preflight.os.scandir

    def hung(path):
        started.append(path)
        release.wait(10)
        return original(path)
    monkeypatch.setattr(preflight.os, "scandir", hung)
    folder = str(home / "references")
    threads = threading.active_count()
    status, message = preflight._folder("Project", folder, 0.05)
    assert status == "unverified" and "did not respond" in message
    for _ in range(5):
        status, message = preflight._folder("Project", folder, 0.05)
        assert status == "unverified" and "still not responding" in message
    assert len(started) == 1 and threading.active_count() <= threads + 1
    release.set()
    deadline = time.time() + 5
    while time.time() < deadline and preflight._OUTSTANDING.get(str(Path(folder))) is not None \
            and not preflight._OUTSTANDING[str(Path(folder))].done():
        time.sleep(0.01)
    assert preflight._folder("Project", folder, 1)[0] == "pass"


# Qt offscreen: 9 (Resume gate) and 12 (Repair dialog)


@pytest.fixture
def app():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def pump(app, done, timeout=10):
    end = time.time() + timeout
    while not done() and time.time() < end:
        app.processEvents()
        time.sleep(0.01)


@pytest.fixture
def window(home, app, _no_desktop_audio):
    from simplegen import ui
    settings = Settings(project_folder=str(home / "references"), template_folder=str(home / "templates"),
                        template_file="H3_Persistent_References_Clonable_Chain.json", output_folder=str(home / "out"))
    instance = ui.MainWindow(settings)
    yield instance
    instance.worker = None
    instance.close()
    instance.deleteLater()
    app.processEvents()


def test_resume_of_unsubmitted_run_is_gated_and_offers_repair(window, home, app, monkeypatch):
    from simplegen import ui
    run = blocked_run(home)
    failing = dict(PASSED, status="fail", checks=[{"id": "target_compatibility", "status": "fail",
                                                  "message": "unet_name unavailable", "details": {}}])
    templates = []
    monkeypatch.setattr(ui, "run_preflight",
                        lambda settings, template_path=None, **_: templates.append(template_path) or failing)
    offered, started = [], []
    monkeypatch.setattr(window, "offer_repair", lambda folder, result: offered.append((folder, result)))
    monkeypatch.setattr(window, "start_generation", lambda folder=None, **k: started.append(folder))
    window.resume_saved(run, json.loads((run / "state.json").read_text()))
    pump(app, lambda: window.preflight_worker is None)
    assert offered == [(run, failing)] and started == []
    assert templates == [run / "template.original.json"]
    assert "Preflight fail" in window.status.text()

    submitted = blocked_run(home / "second", prompt_id="job-1")
    window.resume_saved(submitted, json.loads((submitted / "state.json").read_text()))
    assert started == [submitted] and len(offered) == 1 and window.preflight_worker is None


def test_saved_run_dialog_offers_repair_only_when_eligible(home, app):
    from simplegen.ui import SavedRunDialog
    run = blocked_run(home)
    dialog = SavedRunDialog(json.loads((run / "state.json").read_text()), run)
    assert not dialog.repair_button.isHidden()
    dialog.deleteLater()
    state = json.loads((run / "state.json").read_text())
    state["prompt_id"] = "job-1"
    dialog = SavedRunDialog(state, run)
    assert dialog.repair_button.isHidden()
    dialog.deleteLater()


def test_repair_dialog_lists_loaders_and_only_offers_target_choices(home, app, schema):
    from simplegen.repair_ui import RepairDialog
    run = blocked_run(home)
    dialog = RepairDialog(run, inspect=partial(repair.inspect, object_info=schema),
                          apply=partial(repair.apply, object_info=schema, resolve=tools))
    pump(app, lambda: dialog.info is not None and dialog.worker is None)
    info = dialog.info
    assert len(dialog.combos) == len(info["selections"]) == 4
    for selection in info["selections"]:
        combo = dialog.combos[selection["key"]]
        offered = [combo.itemData(i) for i in range(combo.count())]
        assert offered == [None, *selection["choices"]]
        assert combo.currentData() == (selection["current"] if selection["available"] else None)
    assert not dialog.save_button.isEnabled()
    assert not dialog.run_button.isEnabled() and not dialog.schedule_button.isEnabled()
    for identity, model in fix(info).items():
        combo = dialog.combos[identity]
        combo.setCurrentIndex(combo.findData(model))
    assert dialog.save_button.isEnabled()
    dialog.save_button.click()
    pump(app, lambda: dialog.result is not None and dialog.worker is None)
    assert dialog.result["status"] == "published"
    assert dialog.run_button.isEnabled() and dialog.schedule_button.isEnabled()
    assert not dialog.save_button.isEnabled()
    dialog.run_button.click()
    assert dialog.choice == "run"
    dialog.deleteLater()


def test_schedule_dialog_shows_repaired_link_and_repair_action(home, app):
    from simplegen.schedule_ui import ScheduleDialog
    scheduler = Scheduler(home / "schedules")
    base = {"schema_version": 1, "concept": "Lighthouse", "due_at": "2026-09-28T13:09:19+00:00",
            "selected_local": "2026-09-28T13:09:19+00:00", "started": True, "retry_count": 0}
    scheduler._save({**base, "id": "a", "folder": str(home / "a"), "status": "attention",
                     "note": "Rendering is blocked"})
    scheduler._save({**base, "id": "b", "folder": str(home / "b"), "status": "repaired",
                     "repaired_to": {"run_id": "new", "folder": str(home / "20260928-repair-x")}})
    repaired = []
    dialog = ScheduleDialog(scheduler, lambda: None, repair=repaired.append)
    rows = {dialog.rows.topLevelItem(i).data(0, 256)["id"]: dialog.rows.topLevelItem(i)
            for i in range(dialog.rows.topLevelItemCount())}
    assert rows["b"].text(2) == "Repaired → 20260928-repair-x"
    dialog.rows.setCurrentItem(rows["b"])
    assert not dialog.repair_button.isEnabled() and not dialog.now.isEnabled()
    dialog.rows.setCurrentItem(rows["a"])
    assert dialog.repair_button.isEnabled()
    dialog.repair_button.click()
    assert [r["id"] for r in repaired] == ["a"]
    dialog.timer.stop()
    dialog.deleteLater()
    scheduler.close()


# Review follow-up: Remove releases a scheduled prepared revision


def scheduled_revision(home, schema, clock):
    scheduler = Scheduler(home / "schedules", clock=lambda: clock[0])
    run = blocked_run(home)
    result = repair.apply(run, fix(repair.inspect(run, object_info=schema)), object_info=schema, resolve=tools)
    revision = Path(result["folder"])
    record = scheduler.add_prepared(revision, datetime.fromtimestamp(clock[0] + 3600, UTC),
                                    preflight=result["preflight"])
    return scheduler, revision, record, result


def test_removed_prepared_schedule_is_released_and_can_be_scheduled_again(home, schema):
    clock = [datetime.now(UTC).timestamp()]
    scheduler, revision, record, result = scheduled_revision(home, schema, clock)
    scheduler.change(record["id"], "remove")
    state = json.loads((revision / "state.json").read_text())
    assert "schedule" not in state and state["status"] == "validated"
    assert state["schedule_history"][0]["due_at"] == record["due_at"]
    assert state["schedule_history"][0]["record_id"] == record["id"]
    assert state["workflow_acceptance"]["provenance"] == "repair, no agent pass"
    # A removed record never dispatches.
    clock[0] += 7200
    assert scheduler.tick() == []
    assert scheduler.claim(record["id"], PASSED) is None
    assert scheduler.get(record["id"])["status"] == "removed"
    # Scheduling again works and keeps the removed record in history.
    again = scheduler.add_prepared(revision, datetime.fromtimestamp(clock[0] + 3600, UTC),
                                   preflight=result["preflight"])
    assert again["status"] == "scheduled" and scheduler.get(record["id"])["status"] == "scheduled"
    archived = [json.loads(p.read_text()) for p in (home / "schedules" / "history").glob("*.json")]
    assert [a["status"] for a in archived] == ["removed"] and archived[0]["due_at"] == record["due_at"]
    with pytest.raises(ValueError, match="already scheduled"):
        scheduler.add_prepared(revision, datetime.fromtimestamp(clock[0] + 3600, UTC),
                               preflight=result["preflight"])
    clock[0] += 7200
    assert [r["id"] for r in scheduler.tick()] == [record["id"]]
    scheduler.close()


def test_removed_prepared_schedule_resumes_to_submission_without_agent(home, schema, monkeypatch):
    clock = [datetime.now(UTC).timestamp()]
    scheduler, revision, record, _ = scheduled_revision(home, schema, clock)
    with pytest.raises(ValueError, match="scheduled"):
        engine.run(Settings(), "", lambda event: None, Event(), resume_folder=revision)
    scheduler.change(record["id"], "remove")
    no_agent(monkeypatch, home)
    calls = {"uploaded": []}
    monkeypatch.setattr(engine, "Comfy", fake_comfy(schema, calls))
    with pytest.raises(RuntimeError, match="fake submit reached"):
        engine.run(Settings(), "", lambda event: None, Event(), resume_folder=revision)
    assert calls["submitted"]["1"]["inputs"]["unet_name"] == INT8_UNET
    scheduler.close()


def test_removing_a_started_schedule_keeps_its_protection(home, schema):
    clock = [datetime.now(UTC).timestamp()]
    scheduler, revision, record, _ = scheduled_revision(home, schema, clock)
    clock[0] += 7200
    scheduler.tick()
    assert scheduler.claim(record["id"], PASSED) == revision
    before = json.loads((revision / "state.json").read_text())
    scheduler.finish(record["id"], False)
    scheduler.change(record["id"], "remove")
    after = json.loads((revision / "state.json").read_text())
    assert after == before and "schedule_history" not in after
    scheduler.close()


def test_scheduled_repair_links_existing_unlinked_revision(window, home, schema, monkeypatch):
    from simplegen import repair_ui
    run = blocked_run(home)
    state = json.loads((run / "state.json").read_text())
    record = {"schema_version": 1, "id": state["run_id"], "folder": str(run), "concept": "Lighthouse",
              "due_at": state["schedule"]["due_at"], "selected_local": state["schedule"]["due_at"],
              "status": "attention", "started": True, "retry_count": 0, "note": "Rendering is blocked"}
    window.scheduler._save(record)
    first = repair.apply(run, fix(repair.inspect(run, object_info=schema)), object_info=schema, resolve=tools)
    monkeypatch.setattr(repair_ui, "RepairDialog", lambda *a, **k: pytest.fail("repaired again"))
    offered = []
    monkeypatch.setattr(window, "choose_existing_revision", lambda revision: offered.append(revision) or "link")
    window.open_repair(run, record["id"])
    assert offered == [Path(first["folder"])]
    linked = window.scheduler.get(record["id"])
    assert linked["status"] == "repaired" and linked["repaired_to"]["folder"] == first["folder"]
    assert published(home) == [Path(first["folder"])]
    assert repair.unlinked_revision(run, {first["folder"]}) is None
