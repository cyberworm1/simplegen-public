"""Phase 1 preflight gate: real fixtures, no services, no agent CLI."""

import json
import os
import shutil
import time
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import Event

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import httpx
import pytest
from conftest import PASSED
from test_batches import _approved, _member

from simplegen import agents, batches, preflight
from simplegen.preflight import run_preflight, run_saved_preflight
from simplegen.scheduling import Scheduler
from simplegen.settings import Settings, atomic_json, with_generation_location

REAL = Path(__file__).parent / "fixtures" / "real"
BUNDLED = Path(__file__).parents[1] / "src" / "simplegen" / "data" / "workflows"
BF16 = "H3_Persistent_References_Clonable_Chain.json"
INT8 = "H3_Persistent_References_Clonable_Chain_LOCAL_INT8_NVFP4.json"
REAL_MEMBER_PREFLIGHT = batches.member_preflight


def tools(name):
    return f"/usr/bin/{name}"


@pytest.fixture(scope="module")
def local_schema():
    return json.loads((REAL / "schemas" / "local_object_info.json").read_text())


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("SIMPLEGEN_HOME", str(tmp_path / "home"))
    templates = tmp_path / "templates"
    shutil.copytree(BUNDLED, templates)
    (tmp_path / "references").mkdir()
    shutil.copy(REAL / "schemas" / "cloud_global_object_info.json", tmp_path / "cloud.json")
    return tmp_path


def make(home, template, **changes):
    return replace(Settings(project_folder=str(home / "references"), template_folder=str(home / "templates"),
                            template_file=template, output_folder=str(home / "out"), steps=25,
                            aspect_ratio="16:9", runpod_schema_file=str(home / "cloud.json")), **changes)


def cloud(settings):
    """Managed RunPod: checked against the cached Global Volume schema."""
    return replace(settings, cloud=True, managed_runpod=True, comfy_url="https://example-8189.proxy.runpod.net")


def check(result, identity):
    return next(c for c in result["checks"] if c["id"] == identity)


def test_bf16_template_fails_on_local_and_names_available_models(home, local_schema):
    result = run_preflight(make(home, BF16), object_info=local_schema, resolve=tools)
    assert result["status"] == "fail"
    target = check(result, "target_compatibility")
    assert target["status"] == "fail"
    assert "UNETLoader" in target["message"] and "unet_name" in target["message"]
    assert "minimax_h3_ref2va_bf16.safetensors" in target["message"]
    unet = next(m for m in target["details"]["missing_models"] if m["input"] == "unet_name")
    assert unet["class_type"] == "UNETLoader" and unet["model"] == "minimax_h3_ref2va_bf16.safetensors"
    assert "minimax_h3_ref2va_pruned_int8_convrot.safetensors" in unet["available"]
    assert "minimax_h3_ref2va_pruned_int8_convrot.safetensors" in target["message"]
    json.dumps(result)  # JSON-serializable receipt.


def test_local_int8_template_passes_local_and_fails_cloud(home, local_schema):
    assert run_preflight(make(home, INT8), object_info=local_schema, resolve=tools)["status"] == "pass"
    result = run_preflight(cloud(make(home, INT8)), resolve=tools)
    assert result["status"] == "fail"
    assert "cannot map 2 frontend widgets" in check(result, "target_compatibility")["message"]


def test_bf16_template_passes_cloud_with_schema_age(home):
    old = time.time() - 7200
    os.utime(home / "cloud.json", (old, old))
    result = run_preflight(cloud(make(home, BF16)), resolve=tools)
    assert result["status"] == "pass", result
    assert 7000 <= result["schema_age_seconds"] <= 7400
    assert "cached schema is 2.0 h old" in check(result, "target_compatibility")["message"]
    assert result["schema_sha256"]


def test_unreachable_comfy_and_missing_cache_are_unverified(home):
    def refuse(request):
        raise httpx.ConnectError("refused", request=request)
    client = httpx.Client(transport=httpx.MockTransport(refuse))
    result = run_preflight(make(home, BF16), comfy_client=client, resolve=tools)
    assert result["status"] == "unverified"
    assert check(result, "target_compatibility")["status"] == "unverified"
    assert check(result, "template_structure")["status"] == "unverified"
    (home / "cloud.json").unlink()
    assert run_preflight(cloud(make(home, BF16)), resolve=tools)["status"] == "unverified"


def test_missing_linked_folder_is_unverified_unset_folder_fails(home, local_schema):
    # A missing folder may be an unmounted SFTP share: unverified (retryable), never pass.
    settings = make(home, BF16, project_folder=str(home / "unmounted"))
    result = run_preflight(settings, object_info=local_schema, template_path=home / "templates" / INT8,
                           resolve=tools)
    assert check(result, "linked_folders")["status"] == "unverified"
    assert result["status"] == "unverified"
    shutil.rmtree(home / "templates")
    result = run_preflight(make(home, INT8), object_info=local_schema, resolve=tools)
    assert result["status"] == "unverified"
    assert check(result, "template_structure")["status"] == "unverified"
    result = run_preflight(make(home, INT8, project_folder=""), object_info=local_schema, resolve=tools)
    assert check(result, "linked_folders")["status"] == "fail"


def test_slow_mount_times_out_as_unverified(home, local_schema, monkeypatch):
    original = os.scandir
    monkeypatch.setattr(preflight.os, "scandir", lambda path: (time.sleep(1), original(path))[1])
    result = run_preflight(make(home, INT8), object_info=local_schema, resolve=tools, fs_timeout=0.1)
    assert check(result, "linked_folders")["status"] == "unverified"
    assert "did not respond" in check(result, "linked_folders")["message"]


def test_tools_and_disk_space(home, local_schema):
    def missing(name):
        raise ValueError(f"Install {name}")
    result = run_preflight(make(home, INT8, preflight_min_free_gb=1e9), object_info=local_schema, resolve=missing)
    assert check(result, "tools")["status"] == "fail"
    assert check(result, "tools")["details"]["missing"] == ["codex", "ffprobe"]
    assert check(result, "disk_space")["status"] == "fail"


def local_schedule_env(home, monkeypatch):
    monkeypatch.setattr(agents, "executable", lambda name: pytest.fail("agent CLI must not run"))
    clock = [datetime(2030, 1, 1, tzinfo=UTC).timestamp()]
    return Scheduler(home / "schedules", clock=lambda: clock[0]), clock


def test_scheduler_add_requires_passing_or_overridden_preflight(home, local_schema, monkeypatch):
    scheduler, clock = local_schedule_env(home, monkeypatch)
    settings = make(home, BF16)
    due = datetime.fromtimestamp(clock[0] + 3600, UTC)
    failing = run_preflight(settings, object_info=local_schema, resolve=tools)
    with pytest.raises(ValueError, match="unet_name"):
        scheduler.add(settings, "Nope", due, preflight=failing)
    unverified = dict(PASSED, status="unverified",
                      checks=[{"id": "target_compatibility", "status": "unverified",
                               "message": "ComfyUI unreachable", "details": {}}])
    with pytest.raises(ValueError, match="could not verify"):
        scheduler.add(settings, "Nope", due, preflight=unverified)
    with pytest.raises(ValueError, match="requires a preflight"):
        scheduler.add(settings, "Nope", due, preflight={})
    assert scheduler.records() == []
    assert not (home / "out").exists() or not any((home / "out").iterdir())
    record = scheduler.add(settings, "Allowed", due, preflight=unverified, override_unverified=True)
    assert record["preflight"]["override_unverified"] is True
    state = json.loads((Path(record["folder"]) / "state.json").read_text())
    assert state["preflight"]["override_unverified"] is True
    assert state["preflight"]["status"] == "unverified" and state["preflight"]["checked_at"]


def test_dispatch_recheck_moves_to_attention_without_agent(home, local_schema, monkeypatch):
    scheduler, clock = local_schedule_env(home, monkeypatch)
    settings = make(home, INT8)
    passing = run_preflight(settings, object_info=local_schema, resolve=tools)
    assert passing["status"] == "pass"
    record = scheduler.add(settings, "Later", datetime.fromtimestamp(clock[0] + 60, UTC), preflight=passing)
    state = json.loads((Path(record["folder"]) / "state.json").read_text())
    assert state["preflight"]["schema_sha256"] == passing["schema_sha256"]
    # The model disappears from the local machine after scheduling.
    removed = json.loads(json.dumps(local_schema))
    choices = removed["UNETLoader"]["input"]["required"]["unet_name"][0]
    choices.remove("minimax_h3_ref2va_pruned_int8_convrot.safetensors")
    clock[0] += 120
    assert [r["id"] for r in scheduler.tick()] == [record["id"]]
    recheck = run_saved_preflight(Path(record["folder"]), object_info=removed, resolve=tools)
    assert recheck["status"] == "fail"
    assert scheduler.claim(record["id"], recheck) is None
    saved = scheduler.get(record["id"])
    assert saved["status"] == "attention" and not saved["started"]
    assert "minimax_h3_ref2va_pruned_int8_convrot.safetensors" in saved["note"]
    assert saved["preflight"]["status"] == "fail"


def test_dispatch_unverified_uses_blocked_retry(home, local_schema, monkeypatch):
    scheduler, clock = local_schedule_env(home, monkeypatch)
    record = scheduler.add(make(home, INT8), "Later", datetime.fromtimestamp(clock[0] + 60, UTC), preflight=PASSED)
    clock[0] += 120
    scheduler.tick()
    unverified = dict(PASSED, status="unverified",
                      checks=[{"id": "linked_folders", "status": "unverified", "message": "Mount gone", "details": {}}])
    assert scheduler.claim(record["id"], unverified) is None
    saved = scheduler.get(record["id"])
    assert saved["status"] == "waiting" and not saved["started"] and saved["retry_count"] == 1
    assert datetime.fromisoformat(saved["next_retry_at"]).timestamp() == clock[0] + 1800
    assert scheduler.claim(record["id"], PASSED) is None  # Not before its retry instant.
    clock[0] += 1800
    assert scheduler.claim(record["id"], PASSED) == Path(record["folder"])


def test_regression_blocked_bf16_run_snapshot(home, local_schema):
    run = home / "run"
    shutil.copytree(REAL / "runs" / "blocked_bf16_model_on_local", run)
    state = json.loads((run / "state.json").read_text())
    settings = replace(Settings(**{k: v for k, v in state["settings"].items() if k in Settings.__dataclass_fields__}),
                       project_folder=str(home / "references"), output_folder=str(home / "out"))
    result = run_preflight(settings, template_path=run / "template.original.json", object_info=local_schema,
                           resolve=tools)
    target = check(result, "target_compatibility")
    assert result["status"] == "fail"
    assert "UNETLoader (node 1) unet_name: model 'minimax_h3_ref2va_bf16.safetensors'" in target["message"]


def test_workflow_is_per_destination_and_old_settings_load(home, tmp_path):
    old = {"project_folder": str(home / "references"), "template_folder": str(home / "templates"),
           "template_file": INT8, "output_folder": str(home / "out"), "cloud": False,
           "connection_profiles": {"cloud": {"comfy_url": "https://x-8189.proxy.runpod.net"}}}
    path = tmp_path / "settings.json"
    path.write_text(json.dumps(old))
    loaded = Settings.load(path)
    assert loaded.template_file == INT8 and loaded.preflight_min_free_gb == 5.0
    # Old cloud profile lacks workflow keys: Cloud falls back to its default workflow.
    to_cloud = with_generation_location(loaded, True)
    assert to_cloud.template_file == loaded.runpod_default_workflow == BF16
    chosen = replace(to_cloud, template_file="cloud-choice.json")
    back = with_generation_location(chosen, False)
    assert back.template_file == INT8
    assert with_generation_location(back, True).template_file == "cloud-choice.json"
    # With no cloud default, the other destination starts unset.
    assert with_generation_location(replace(loaded, runpod_default_workflow=""), True).template_file == ""
    back.save(path)
    assert Settings.load(path).template_file == INT8


def armed_member_with_reference(home):
    store = batches.BatchStore(home)
    state = store.create()
    target = _member(store, state, home, 1)
    reference = home / "mount" / "ref.png"
    reference.parent.mkdir()
    reference.write_bytes(b"png")
    (target / "used_references.txt").write_text(str(reference))
    state["members"][0]["prepared_fingerprint"] = batches._fingerprint(target)
    state.update(status="armed", gpu_choices=_approved(), next_check_at=datetime.now(UTC).isoformat())
    store._save(Path(state["folder"]), state)
    return store, state, reference


def test_batch_unavailable_reference_waits_changed_reference_fails(home, monkeypatch):
    monkeypatch.setattr(batches, "_catalog", lambda *_: pytest.fail("no provider before references"))
    store, state, reference = armed_member_with_reference(home)
    shutil.rmtree(reference.parent)  # SFTP mount disappeared; no pod exists yet.
    waiting = store.run_batch(state["id"], lambda _: None, Event())
    assert waiting["status"] == "waiting"
    assert "Prepared reference is missing" in waiting["waiting_reason"]
    assert datetime.fromisoformat(waiting["next_check_at"]) > datetime.now(UTC) + timedelta(minutes=4)

    store, state, reference = armed_member_with_reference(home / "second")
    reference.write_bytes(b"changed content")
    os.utime(reference, (1, 1))
    with pytest.raises(ValueError, match="changed"):
        store.run_batch(state["id"], lambda _: None, Event())
    assert store.get(state["id"])["status"] == "failed"


def test_batch_add_and_prepare_gate_on_cloud_preflight(home, monkeypatch):
    run = home / "member"
    run.mkdir()
    settings = cloud(make(home, INT8, managed_runpod=True, global_desktop=True))
    atomic_json(run / "state.json", {"schema_version": 4, "run_id": "member1", "concept": "c",
                                     "status": "created", "settings": settings.public_dict(), "downloads": {}})
    monkeypatch.setattr(batches, "member_preflight", REAL_MEMBER_PREFLIGHT)
    monkeypatch.setattr(agents, "executable", tools)
    store = batches.BatchStore(home)
    batch = store.create()
    with pytest.raises(ValueError, match="cannot map 2 frontend widgets"):
        store.add(batch["id"], run)
    assert store.get(batch["id"])["members"] == []
    atomic_json(run / "state.json", {**json.loads((run / "state.json").read_text()),
                                     "settings": replace(settings, template_file=BF16).public_dict()})
    added = store.add(batch["id"], run)
    assert added["members"][0]["preflight"]["status"] == "pass"
    # The cached schema loses the BF16 model before preparation: no agent work runs.
    schema = json.loads((home / "cloud.json").read_text())
    schema["UNETLoader"]["input"]["required"]["unet_name"][0].remove("minimax_h3_ref2va_bf16.safetensors")
    (home / "cloud.json").write_text(json.dumps(schema))
    monkeypatch.setattr(batches, "_run_locked", lambda *a, **k: pytest.fail("agent work before preflight"))
    with pytest.raises(ValueError, match="unet_name"):
        store.prepare(batch["id"], lambda _: None, Event())
    assert store.get(batch["id"])["status"] == "preparation_failed"


# Qt offscreen: Generate gate

@pytest.fixture
def window(home, _no_desktop_audio):
    from PySide6.QtWidgets import QApplication

    from simplegen import ui
    app = QApplication.instance() or QApplication([])
    instance = ui.MainWindow(make(home, BF16, template_file=str(home / "templates" / BF16)))
    yield instance, app
    instance.worker = None
    instance.close()
    instance.deleteLater()
    app.processEvents()


class Worker:
    def __init__(self, *args, **kwargs):
        from PySide6.QtCore import QObject, Signal

        class Signals(QObject):
            event = Signal(dict)
            succeeded = Signal(dict)
            failed = Signal(str)
            resource_choice = Signal(list)
            finished = Signal()
        self.signals = Signals()
        for name in ("event", "succeeded", "failed", "resource_choice", "finished"):
            setattr(self, name, getattr(self.signals, name))
        self.settings = args[0]

    def start(self):
        self.started = True


def pump(app, done, timeout=5):
    end = time.time() + timeout
    while not done() and time.time() < end:
        app.processEvents()
        time.sleep(0.01)


@pytest.mark.parametrize("status", ["fail", "unverified"])
def test_generate_blocks_on_fail_and_confirms_on_unverified(window, monkeypatch, status):
    from simplegen import ui
    main, app = window
    result = dict(PASSED, status=status, checks=[{
        "id": "target_compatibility", "status": status, "message": f"Model check {status}",
        "details": {"missing_models": [{"node_id": "1", "class_type": "UNETLoader", "input": "unet_name",
                                        "model": "bf16", "available": ["int8"]}]}}])
    monkeypatch.setattr(ui, "run_preflight", lambda settings, template_path=None: result)
    monkeypatch.setattr(ui, "GenerationWorker", Worker)
    dialogs = []
    answer = [False]
    monkeypatch.setattr(main, "preflight_dialog",
                        lambda res, confirm: dialogs.append(confirm) or answer[0])
    main.concept.setPlainText("A garden")
    main.start_generation()
    assert main.worker is None  # Nothing starts until preflight finishes.
    pump(app, lambda: main.preflight_worker is None)
    assert dialogs == [status == "unverified"]
    assert main.worker is None
    assert f"Preflight {status}: Model check {status}" in main.status.text()
    if status == "unverified":
        answer[0] = True
        main.start_generation()
        pump(app, lambda: main.preflight_worker is None)
        assert main.worker is not None and main.worker.started


def test_scheduled_dispatch_failure_shows_reason_without_starting(window, home, local_schema, monkeypatch):
    from simplegen import ui
    main, app = window
    settings = make(home, INT8)
    record = main.scheduler.add(settings, "Later", datetime.now(UTC) + timedelta(hours=1), preflight=PASSED)
    main.scheduler.change(record["id"], "now")
    main.schedule_timer.stop()
    removed = json.loads(json.dumps(local_schema))
    removed["UNETLoader"]["input"]["required"]["unet_name"][0].remove(
        "minimax_h3_ref2va_pruned_int8_convrot.safetensors")
    monkeypatch.setattr(ui, "local_queue_empty", lambda settings: True)
    monkeypatch.setattr(ui, "run_saved_preflight",
                        lambda folder: run_saved_preflight(folder, object_info=removed, resolve=tools))
    monkeypatch.setattr(main, "start_generation", lambda *a, **k: pytest.fail("dispatched after failed preflight"))
    main.poll_schedules()
    pump(app, lambda: main.schedule_probe is None)
    saved = main.scheduler.get(record["id"])
    assert saved["status"] == "attention" and not saved["started"]
    from simplegen.schedule_ui import ScheduleDialog
    dialog = ScheduleDialog(main.scheduler, lambda: None, main)
    row = dialog.rows.topLevelItem(0).text(2)
    assert "Preflight failed" in row and "unet_name" in row
    dialog.deleteLater()
