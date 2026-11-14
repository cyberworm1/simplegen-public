"""Offscreen UI checks; no agent, network, GPU, or video rendering is used."""

import json
import os
import sys
import threading
import time
import types
from dataclasses import asdict, replace
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from conftest import PASSED
from PySide6.QtCore import QObject, Signal
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import QApplication, QDialog, QMessageBox

from simplegen import ui
from simplegen.builtin_workflows import BUNDLED, REFMOD_LABEL, SCENE_LABEL, scene_template_path
from simplegen.settings import Settings


@pytest.fixture(scope="session")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def config(tmp_path, monkeypatch):
    monkeypatch.setenv("SIMPLEGEN_HOME", str(tmp_path / "config"))
    project = tmp_path / "references"
    templates = tmp_path / "templates"
    project.mkdir()
    templates.mkdir()
    (templates / "workflow.json").write_text("{}")
    return Settings(
        project_folder=str(project),
        template_folder=str(templates),
        template_file=str(templates / "workflow.json"),
        output_folder=str(tmp_path / "outputs"),
    )


@pytest.fixture(autouse=True)
def preflight_passes(monkeypatch):
    """These tests use placeholder workflows; preflight gates are tested in test_preflight."""
    monkeypatch.setattr(ui.MainWindow, "preflight_then",
                        lambda self, settings, template, proceed: proceed(PASSED, False))
    monkeypatch.setattr(ui, "run_saved_preflight", lambda folder: PASSED)


@pytest.fixture
def window(app, config, _no_desktop_audio):
    instance = ui.MainWindow(config)
    yield instance
    instance.worker = None
    instance.close()
    instance.deleteLater()
    app.processEvents()


def test_old_scene_run_edit_as_new_keeps_snapshot_and_source(window, config):
    from simplegen.editing import digest
    from simplegen.engine import create_run
    from simplegen.h3scene_compat import LEGACY_FILENAME

    fixtures = Path(__file__).parent / "fixtures/real/h3scene/legacy"
    state = json.loads((fixtures / "state.json").read_text())
    old_builtin = scene_template_path().with_name(LEGACY_FILENAME)
    state["settings"] = {**config.public_dict(), "template_file": str(old_builtin), "template_folder": ""}
    folder = Path(config.output_folder) / "old-saved-run"
    folder.mkdir(parents=True)
    snapshot = folder / "template.original.json"
    snapshot.write_bytes((fixtures / "template.json").read_bytes())
    state.update(template_path=str(old_builtin), template_sha256=digest(snapshot))
    saved = json.dumps(state).encode()
    (folder / "state.json").write_bytes(saved)
    window.stage_saved_run(folder, state, "Revised scene")
    settings = window.collect_settings()
    assert settings.template_file == str(scene_template_path())
    assert window.template.currentText() == SCENE_LABEL
    revision, child = create_run(settings, window.concept.toPlainText(), window.staged_fork)
    assert child["forked_from"]["run_id"] == state["run_id"]
    assert child["concept"] == "Revised scene"
    assert (revision / "template.original.json").read_bytes() == snapshot.read_bytes()
    assert (folder / "state.json").read_bytes() == saved


@pytest.mark.parametrize("absolute", [False, True])
def test_old_user_scene_template_uses_current_display_name(window, config, absolute):
    from simplegen.builtin_workflows import SCENE_FILENAME
    from simplegen.h3scene_compat import LEGACY_FILENAME

    source = Path(config.template_folder) / LEGACY_FILENAME
    source.write_bytes((Path(__file__).parent / "fixtures/real/h3scene/legacy/template.json").read_bytes())
    window.refresh_templates()
    index = window.template.findData(str(source))
    assert index >= 0 and window.template.itemText(index) == SCENE_FILENAME
    settings = replace(config, template_file=str(source) if absolute else source.name)
    window.show_workflow(settings)
    assert window.template.currentData() == str(source)


def test_templates_are_top_level_workflows(window, config):
    templates = Path(config.template_folder)
    (templates / "5minVid").mkdir()
    (templates / "5minVid" / "hidden.json").write_text("{}")
    (templates / "sample_prompts.json").write_text("{}")
    (templates / "example_prompt.txt").write_text("not a workflow")
    (templates / "H3_Persistent_References_Clonable_Chain.json").write_text("{}")
    window.refresh_templates()
    names = [window.template.itemText(index) for index in range(window.template.count())]
    assert names == ["H3_Persistent_References_Clonable_Chain.json", "workflow.json", SCENE_LABEL, REFMOD_LABEL,
                     *(label for _, label, _ in BUNDLED)]


def test_schedule_click_does_not_occupy_worker_and_cloud_disables_it(window, monkeypatch):
    monkeypatch.setattr(ui, "GenerationWorker", lambda *a: pytest.fail("worker started before deadline"))
    window.concept.setPlainText("Scheduled concept")
    window.schedule_enabled.setChecked(True)
    assert window.schedule_time.displayFormat() == "MM/dd/yy HH:mm"
    window.start_generation()
    assert window.worker is None and window.generate_button.isEnabled()
    assert not window.schedule_enabled.isChecked()
    records = window.scheduler.records()
    assert len(records) == 1 and records[0]["status"] == "scheduled"
    window.concept.setPlainText("Different concept")
    assert window.scheduler.records()[0]["concept"] == "Scheduled concept"
    window.select_location(True)
    assert not window.schedule_enabled.isEnabled() and not window.schedule_time.isEnabled()
    window.select_location(False)
    assert window.schedule_enabled.isEnabled()
    monkeypatch.setattr(ui, "GenerationWorker", FakeWorker)
    window.start_generation()
    assert window.worker.started and window.worker.concept == "Different concept"
    assert window.scheduler.records()[0]["status"] == "scheduled"
    assert window.schedule_enabled.isEnabled()  # Local work can still be scheduled during a run.
    window.worker.finished.emit()


def test_busy_schedule_keeps_generate_available_and_claims_once_when_idle(window, monkeypatch):
    from datetime import UTC, datetime, timedelta
    record = window.scheduler.add(window.collect_settings(), "Later", datetime.now(UTC) + timedelta(hours=1), preflight=PASSED)
    window.scheduler.change(record["id"], "now")
    window.schedule_ready(record["id"], False)
    assert window.generate_button.isEnabled()
    assert window.scheduler.get(record["id"])["status"] == "waiting"
    assert "1 waiting" in window.scheduled_button.text()
    assert "Waiting for local ComfyUI queue to clear" in window.schedule_status.text()
    starts = []

    def start(folder):
        starts.append(folder)
        window.worker = object()

    monkeypatch.setattr(window, "start_generation", start)
    window.schedule_ready(record["id"], True, None, PASSED)
    assert starts == []  # A blocked request waits for its saved retry instant.
    window.scheduler.change(record["id"], "now")
    window.schedule_ready(record["id"], True, None, PASSED)
    window.schedule_ready(record["id"], True, None, PASSED)
    assert starts == [Path(record["folder"])]
    window.worker = None
    window.finish_schedule(True)
    assert window.scheduler.get(record["id"])["status"] == "complete"
    assert window.schedule_status.isHidden()


def test_schedule_probe_displays_failure_then_dispatches_when_idle(window, app, monkeypatch):
    from datetime import UTC, datetime, timedelta
    record = window.scheduler.add(window.collect_settings(), "Due", datetime.now(UTC) + timedelta(hours=1), preflight=PASSED)
    window.scheduler.change(record["id"], "now")
    window.schedule_timer.stop()
    starts = []
    monkeypatch.setattr(window, "start_generation", lambda folder: starts.append(folder))

    def unavailable(settings):
        raise OSError("Connection refused")

    monkeypatch.setattr(ui, "local_queue_empty", unavailable)
    window.poll_schedules()
    pump(app, lambda: window.schedule_probe is None)
    assert not starts
    assert "Waiting for ComfyUI: Connection refused" in window.schedule_status.text()
    assert not window.scheduler.get(record["id"])["started"]

    window.scheduler.change(record["id"], "now")
    monkeypatch.setattr(ui, "local_queue_empty", lambda settings: False)
    window.poll_schedules()
    pump(app, lambda: window.schedule_probe is None)
    assert not starts
    assert "Waiting for local ComfyUI queue to clear" in window.schedule_status.text()

    window.scheduler.change(record["id"], "now")
    monkeypatch.setattr(ui, "local_queue_empty", lambda settings: True)
    window.poll_schedules()
    pump(app, lambda: window.schedule_probe is None)
    assert starts == [Path(record["folder"])]
    assert window.schedule_status.isHidden()


def test_schedule_wait_reason_for_busy_app_and_later_requests(window):
    from datetime import UTC, datetime, timedelta
    records = [window.scheduler.add(window.collect_settings(), name,
               datetime.now(UTC) + timedelta(hours=1), preflight=PASSED) for name in ("First", "Second")]
    for record in records:
        window.scheduler.change(record["id"], "now")
    window.worker = object()
    window.status.setText("Generating video")
    window.poll_schedules()
    assert "2 waiting" in window.scheduled_button.text()
    assert "current simpleGen operation" in window.schedule_status.text()
    assert window.status.text() == "Generating video"
    assert window.scheduler.get(records[1]["id"])["note"] == "Waiting for earlier scheduled requests"
    window.worker = None


def test_due_request_uses_saved_concept_and_restores_current_draft(window, monkeypatch):
    from datetime import UTC, datetime, timedelta
    monkeypatch.setattr(ui, "GenerationWorker", FakeWorker)
    record = window.scheduler.add(window.collect_settings(), "Saved concept", datetime.now(UTC) + timedelta(hours=1), preflight=PASSED)
    window.concept.setPlainText("Draft for later")
    window.scheduler.change(record["id"], "now")
    window.schedule_ready(record["id"], True, None, PASSED)
    worker = window.worker
    assert worker.concept == "Saved concept"
    assert window.concept.toPlainText() == "Draft for later"
    worker.failed.emit("Temporary failure")
    worker.finished.emit()
    assert window.concept.toPlainText() == "Draft for later"
    assert window.scheduler.get(record["id"])["status"] == "attention"


def test_schedule_edit_dialog_preserves_saved_due_and_relative_workflow(window, config):
    from datetime import UTC, datetime, timedelta
    record = window.scheduler.add(config, "Saved", datetime.now(UTC) + timedelta(hours=1), preflight=PASSED)
    saved = json.loads((Path(record["folder"]) / "state.json").read_text())
    dialog = ui.ScheduleEditDialog(record, Settings(**saved["settings"]), lambda *args: {}, window)
    try:
        settings, concept, due = dialog.values()
        assert settings.template_file == config.template_file
        assert concept == "Saved"
        assert due.isoformat() == record["due_at"]
        dialog.workflow.setText("another.json")
        settings, _, _ = dialog.values()
        assert settings.template_file == str(Path(config.template_folder) / "another.json")
    finally:
        dialog.deleteLater()


def test_stale_availability_probe_cannot_claim_edited_schedule(window, config):
    from datetime import UTC, datetime, timedelta

    from simplegen.scheduling import fingerprint

    record = window.scheduler.add(config, "Before", datetime.now(UTC) + timedelta(hours=1), preflight=PASSED)
    window.scheduler.change(record["id"], "now")
    expected = fingerprint(window.scheduler.get(record["id"]))
    window.scheduler.edit(record["id"], expected, config, "After", datetime.now(UTC) + timedelta(hours=2))
    window.schedule_ready(record["id"], True, expected)
    assert window.worker is None
    assert window.scheduler.get(record["id"])["concept"] == "After"
    assert not window.scheduler.get(record["id"])["started"]


def test_saved_run_choice_requires_new_run_for_edited_concept(window, config):
    from datetime import UTC, datetime, timedelta

    record = window.scheduler.add(config, "Original", datetime.now(UTC) + timedelta(hours=1), preflight=PASSED)
    state = json.loads((Path(record["folder"]) / "state.json").read_text())
    dialog = ui.SavedRunDialog(state, Path(record["folder"]), window)
    try:
        assert dialog.resume_button.isEnabled()
        dialog.concept.setPlainText("Revised")
        assert not dialog.resume_button.isEnabled()
        dialog.choose("new")
        assert dialog.choice == "new"
    finally:
        dialog.deleteLater()


def test_cancel_saved_run_choice_keeps_current_draft_and_settings(window, config, monkeypatch):
    from datetime import UTC, datetime, timedelta

    record = window.scheduler.add(config, "Original", datetime.now(UTC) + timedelta(hours=1), preflight=PASSED)
    window.concept.setPlainText("Unrelated draft")
    before = window.collect_settings()
    monkeypatch.setattr(ui.SavedRunDialog, "exec", lambda self: QDialog.DialogCode.Rejected)
    window.open_saved_run_choice(Path(record["folder"]))
    assert window.concept.toPlainText() == "Unrelated draft"
    assert window.collect_settings() == before
    assert window.staged_fork is None


def test_staged_saved_run_keeps_snapshot_when_linked_file_missing(window, config):
    from datetime import UTC, datetime, timedelta

    record = window.scheduler.add(config, "Original", datetime.now(UTC) + timedelta(hours=1), preflight=PASSED)
    folder = Path(record["folder"])
    state = json.loads((folder / "state.json").read_text())
    Path(config.template_file).unlink()
    window.stage_saved_run(folder, state, "Revised")
    assert window.worker is None
    assert window.concept.toPlainText() == "Revised"
    assert window.template.currentData() == config.template_file
    assert "Saved snapshot" in window.template.currentText()
    assert window.collect_settings().template_file == config.template_file
    assert window.staged_fork["template_sha256"] == state["template_sha256"]


def test_staging_managed_source_resets_old_endpoint_without_compute(window, config):
    from datetime import UTC, datetime, timedelta

    record = window.scheduler.add(config, "Original", datetime.now(UTC) + timedelta(hours=1), preflight=PASSED)
    folder = Path(record["folder"])
    state = json.loads((folder / "state.json").read_text())
    state["settings"].update(cloud=True, managed_runpod=True,
                             comfy_url="https://old-pod.example.test")
    state["settings"]["runpod_config_file"] = "/old/controller.json"
    state["settings"]["runpod_credentials_file"] = "/old/credentials.json"
    current = window.collect_settings()
    window.stage_saved_run(folder, state, "New concept")
    assert window.settings.runpod_config_file == current.runpod_config_file
    assert window.settings.runpod_credentials_file == current.runpod_credentials_file
    assert state["settings"]["runpod_config_file"] == "/old/controller.json"
    assert window.worker is None
    assert window.settings.managed_runpod
    assert window.settings.comfy_url == ""
    assert window.location_buttons[True].isChecked()
    assert not window.cloud_shutdown_button.isEnabled()
    assert window.concept.toPlainText() == "New concept"


def test_staged_new_run_can_be_scheduled_with_source_snapshot(window, config):
    from datetime import UTC, datetime, timedelta

    first = window.scheduler.add(config, "Original", datetime.now(UTC) + timedelta(hours=1), preflight=PASSED)
    first_folder = Path(first["folder"])
    state = json.loads((first_folder / "state.json").read_text())
    window.stage_saved_run(first_folder, state, "New concept")
    window.schedule_enabled.setChecked(True)
    window.start_generation()
    assert window.worker is None
    second = next(r for r in window.scheduler.records() if r["id"] != first["id"])
    second_folder = Path(second["folder"])
    assert (second_folder / "template.original.json").read_bytes() == (
        first_folder / "template.original.json").read_bytes()
    assert json.loads((second_folder / "state.json").read_text())["forked_from"]["run_id"] == first["id"]


def test_staged_new_run_starts_only_after_generate_click(window, config, monkeypatch):
    from datetime import UTC, datetime, timedelta

    first = window.scheduler.add(config, "Original", datetime.now(UTC) + timedelta(hours=1), preflight=PASSED)
    folder = Path(first["folder"])
    state = json.loads((folder / "state.json").read_text())
    window.stage_saved_run(folder, state, "Revised")
    assert window.worker is None

    class ForkWorker(FakeWorker):
        def __init__(self, settings, concept, resume_folder=None, parent=None, fork_seed=None):
            super().__init__(settings, concept, resume_folder, parent)
            self.fork_seed = fork_seed

    monkeypatch.setattr(ui, "GenerationWorker", ForkWorker)
    window.start_generation()
    assert window.worker.started
    assert window.worker.concept == "Revised"
    assert window.worker.fork_seed["run_id"] == first["id"]
    assert window.worker.resume_folder is None
    window.worker.finished.emit()


def test_attention_schedule_exposes_edit_as_new_run(window, config):
    from datetime import UTC, datetime, timedelta

    record = window.scheduler.add(config, "Failed", datetime.now(UTC) + timedelta(hours=1), preflight=PASSED)
    window.scheduler.change(record["id"], "now")
    window.scheduler.claim(record["id"], PASSED)
    window.scheduler.finish(record["id"], False)
    dialog = ui.ScheduleDialog(window.scheduler, lambda: None, window, edit=lambda _: None)
    try:
        assert dialog.selected()["id"] == record["id"]
        assert dialog.edit_button.isEnabled()
        assert dialog.edit_button.text() == "Edit as new run…"
    finally:
        dialog.timer.stop()
        dialog.deleteLater()


def test_schedule_edit_waits_for_durable_save_before_closing(window, app, config):
    from datetime import UTC, datetime, timedelta
    record = window.scheduler.add(config, "Saved", datetime.now(UTC) + timedelta(hours=1), preflight=PASSED)
    held = threading.Event()
    entered = threading.Event()

    def save(*_values):
        entered.set()
        held.wait(2)
        return {"saved": True}

    dialog = ui.ScheduleEditDialog(record, config, save, window)
    try:
        dialog.show()
        dialog.save()
        pump(app, entered.is_set)
        assert not dialog.buttons.isEnabled()
        dialog.reject()
        assert dialog.isVisible()
        held.set()
        pump(app, lambda: dialog.operation_worker is None)
        assert dialog.result() == QDialog.DialogCode.Accepted
    finally:
        held.set()
        dialog.deleteLater()


def test_override_inspection_blocks_other_generation_until_finished(window, app):
    entered = threading.Event()
    release = threading.Event()

    def inspect():
        entered.set()
        release.wait(2)
        return {"checked": True}

    window._override_operation(inspect, lambda _result: None)
    try:
        pump(app, entered.is_set)
        assert not window.generate_button.isEnabled()
        assert not window.resume_button.isEnabled()
        window.concept.setPlainText("New concept")
        window.start_generation()
        assert window.worker is None
        event = QCloseEvent()
        window.closeEvent(event)
        assert not event.isAccepted()
    finally:
        release.set()
        pump(app, lambda: window.override_worker is None)


def test_busy_override_drops_inflight_schedule_probe(window, app, config):
    from datetime import UTC, datetime, timedelta

    record = window.scheduler.add(config, "Due", datetime.now(UTC) + timedelta(hours=1), preflight=PASSED)
    window.scheduler.change(record["id"], "now")
    entered = threading.Event()
    release = threading.Event()

    def inspect():
        entered.set()
        release.wait(2)
        return {"checked": True}

    window._override_operation(inspect, lambda _result: None)
    try:
        pump(app, entered.is_set)
        window.schedule_ready(record["id"], True, None, PASSED)
        assert window.worker is None
        assert not window.scheduler.get(record["id"])["started"]
    finally:
        release.set()
        pump(app, lambda: window.override_worker is None)


def test_quantity_persists_and_watch_selects_final_cut(window, tmp_path):
    window.quantity.setValue(3)
    assert window.collect_settings().quantity == 3
    settings_path = tmp_path / "settings.json"
    window.collect_settings().save(settings_path)
    assert Settings.load(settings_path).quantity == 3
    window.display_settings = replace(window.settings, quantity=5)
    window.set_running(True)
    assert window.quantity.value() == 3 and window.quantity.isEnabled()  # The draft is not the running job.
    window.set_running(False)
    result = {"folder": str(tmp_path), "final_path": "one.mp4", "metadata": {}, "variations": [
        {"name": "Variation 1", "final_path": "one.mp4", "metadata": {}},
        {"name": "Variation 2", "final_path": "two.mp4", "metadata": {}},
    ]}
    window.show_result(result, switch_tab=False)
    assert window.tabs.currentIndex() == 0
    assert window.variation_picker.count() == 2
    window.variation_picker.setCurrentIndex(1)
    assert window.player.source().toLocalFile() == str(tmp_path / "two.mp4")


def test_steps_persist_in_settings_and_scheduled_requests(window, tmp_path):
    from datetime import UTC, datetime, timedelta
    window.steps.setValue(37)
    settings = window.collect_settings()
    path = tmp_path / "preferences.json"
    settings.save(path)
    assert Settings.load(path).steps == 37
    record = window.scheduler.add(settings, "Selected steps", datetime.now(UTC) + timedelta(hours=1), preflight=PASSED)
    state = json.loads((Path(record["folder"]) / "state.json").read_text())
    assert state["settings"]["steps"] == 37
    window.display_settings = replace(settings, steps=None)
    window.set_running(True)
    assert window.steps.isEnabled() and window.steps.value() == 37  # Draft steps, not the running job's.
    assert window.display_settings.steps is None


def test_workflow_polling_updates_files_preserves_selection_and_pauses_during_run(window, config, app):
    folder = Path(config.template_folder)
    selected = window.template.currentData()
    window.workflow_timer.setInterval(10)

    def wait_for_count(count):
        deadline = time.monotonic() + 1
        while window.template.count() != count and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(0.005)
        assert window.template.count() == count

    added = folder / "another.JSON"
    added.write_text("{}")
    wait_for_count(4 + len(BUNDLED))
    assert window.template.currentData() == selected
    window.worker = object()
    added.unlink()
    window.poll_workflows()
    assert window.template.count() == 4 + len(BUNDLED)
    window.worker = None
    wait_for_count(3 + len(BUNDLED))
    assert window.template.currentData() == selected
    Path(selected).unlink()
    wait_for_count(2 + len(BUNDLED))


def test_cloud_banner_visible_on_both_tabs(window):
    window.show()
    assert window.cloud_banner.isHidden()
    window.settings.cloud = True
    window.settings.cloud_name = "RunPod"
    window.update_summary()
    for tab in (0, 1):
        window.tabs.setCurrentIndex(tab)
        assert window.cloud_banner.isVisible()
        assert "RunPod" in window.cloud_banner.text()
        assert "Costs money" in window.cloud_banner.text()


def test_settings_include_model_reasoning_and_generation_controls(app, config, monkeypatch):
    agents = types.ModuleType("simplegen.agents")
    agents.model_options = lambda provider: [f"{provider}-small", f"{provider}-large"]
    agents.suggest_model = lambda provider, tier: f"{provider}-small"
    agents.reasoning_options = lambda provider, model: ["auto", "low", "high"]
    agents.agent_problem = lambda settings: None  # the agent rule has its own tests
    monkeypatch.setitem(sys.modules, "simplegen.agents", agents)
    dialog = ui.SettingsDialog(config)
    dialog.provider.setCurrentIndex(dialog.provider.findData("claude"))
    dialog.tier.setCurrentText("Custom")
    dialog.model.setCurrentText("custom-model")
    dialog.reasoning.setCurrentText("high")
    dialog.megapixels.setValue(0.75)
    dialog.duration.setValue(40)
    dialog.cloud.setChecked(True)
    dialog.accept()
    assert dialog.result() == QDialog.DialogCode.Accepted
    saved = Settings.load()
    assert (saved.provider, saved.model_tier, saved.model, saved.reasoning) == (
        "claude",
        "Custom",
        "custom-model",
        "high",
    )
    assert saved.megapixels == 0.75
    assert saved.duration == 40
    assert saved.cloud
    dialog.deleteLater()


class FakeWorker(QObject):
    event = Signal(dict)
    succeeded = Signal(dict)
    failed = Signal(str)
    finished = Signal()
    resource_choice = Signal(object)

    def __init__(self, settings, concept, resume_folder=None, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.concept = concept
        self.resume_folder = resume_folder
        self.cancel_event = threading.Event()
        self.detach_event = threading.Event()
        self.started = False

    def start(self):
        self.started = True


def test_generate_progress_success_and_finish(window, monkeypatch, tmp_path):
    monkeypatch.setattr(ui, "GenerationWorker", FakeWorker)
    shown = []
    monkeypatch.setattr(window, "show_result", shown.append)
    window.concept.setPlainText("A moonlit garden")
    window.start_generation()
    worker = window.worker
    assert worker.started
    assert worker.concept == "A moonlit garden"
    assert window.concept.isEnabled()
    window.concept.setPlainText("Next project draft")
    assert window.generate_button.isEnabled() and window.cancel_button.isEnabled()
    assert all(button.isEnabled() for button in window.provider_buttons.values())  # Draft stays editable.
    window.start_generation()
    assert window.worker is worker and "already in progress" in window.status.text()  # One local run.
    worker.event.emit(
        {"stage": "Generating", "message": "Shot 1", "progress": 24, "folder": str(tmp_path)}
    )
    assert window.progress.value() == 24
    assert window.report_button.isEnabled()
    result = {"folder": str(tmp_path), "final_path": str(tmp_path / "final.mp4"), "metadata": {}}
    worker.succeeded.emit(result)
    assert window.progress.value() == 100
    assert shown == [result]
    worker.finished.emit()
    assert window.worker is None
    assert window.generate_button.isEnabled()
    assert not window.cancel_button.isEnabled()
    assert all(button.isEnabled() for button in window.provider_buttons.values())
    assert window.concept.toPlainText() == "Next project draft"


def test_failure_restores_controls_and_keeps_report(window, monkeypatch, tmp_path):
    monkeypatch.setattr(ui, "GenerationWorker", FakeWorker)
    window.concept.setPlainText("A concept")
    window.start_generation()
    worker = window.worker
    window.concept.setPlainText("Draft survives failure")
    worker.event.emit({"folder": str(tmp_path), "stage": "Planning", "message": "Working"})
    worker.failed.emit("Agent unavailable; see report.txt")
    worker.finished.emit()
    assert "Agent unavailable" in window.status.text()
    assert str(tmp_path) in window.log.toPlainText()
    assert window.report_button.isEnabled()
    assert window.generate_button.isEnabled()
    assert window.concept.toPlainText() == "Draft survives failure"


def test_close_during_run_requests_detach_without_cancel_or_wait(window, monkeypatch):
    monkeypatch.setattr(ui, "GenerationWorker", FakeWorker)
    window.concept.setPlainText("A concept")
    window.start_generation()
    worker = window.worker
    event = QCloseEvent()
    window.closeEvent(event)
    assert not event.isAccepted()
    assert worker.detach_event.is_set()
    assert not worker.cancel_event.is_set()
    assert window.worker is worker
    worker.finished.emit()
    assert window.worker is None


def test_blank_concept_does_not_start(window, monkeypatch):
    warnings = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: warnings.append(args[-1]))
    window.start_generation()
    assert window.worker is None
    assert warnings == ["Enter a concept first."]


def test_resume_cloud_uses_saved_settings_without_replacing_preferences(
    window, config, monkeypatch, tmp_path
):
    monkeypatch.setattr(ui, "GenerationWorker", FakeWorker)
    run_folder = tmp_path / "saved-run"
    run_folder.mkdir()
    saved = replace(
        config,
        provider="claude",
        cloud=True,
        cloud_name="RunPod",
        comfy_url="https://saved.example.test",
    )
    (run_folder / "state.json").write_text(json.dumps({"settings": asdict(saved)}))
    config.save()
    window.start_generation(run_folder)
    assert window.worker.settings.cloud
    assert window.worker.settings.comfy_url == "https://saved.example.test"
    assert not window.cloud_banner.isHidden()
    assert "RunPod" in window.cloud_banner.text()
    assert not window.settings.cloud
    assert not Settings.load().cloud
    # The draft keeps its own agent, destination and endpoint; the running job shows in its own area.
    assert window.provider_buttons["codex"].isChecked()
    assert window.location_buttons[False].isChecked()
    assert window.endpoint.text() != "https://saved.example.test"
    assert "Cloud one-off" in window.jobs_status.text()
    window.worker.finished.emit()
    assert window.provider_buttons["codex"].isChecked()
    assert window.location_buttons[False].isChecked()


def test_provider_radio_buttons_switch_model_and_preserve_custom_choice(window, monkeypatch):
    monkeypatch.setattr(
        "simplegen.agents.suggest_model",
        lambda provider, tier: "haiku" if provider == "claude" else "",
    )
    monkeypatch.setattr(
        "simplegen.agents.reasoning_options",
        lambda provider, model: ["auto"] if provider == "claude" else ["low"],
    )
    window.settings = replace(
        window.settings, model_tier="Custom", model="codex-custom", reasoning="high"
    )
    window.provider_buttons["claude"].click()
    assert window.collect_settings().provider == "claude"
    assert window.settings.model == "haiku"
    assert window.settings.reasoning == "auto"
    assert sum(b.isChecked() for b in window.provider_buttons.values()) == 1
    window.provider_buttons["grok"].click()
    assert window.settings.model == ""
    assert window.settings.model_tier == "Custom"
    window.provider_buttons["codex"].click()
    assert window.settings.model == "codex-custom"
    assert window.settings.reasoning == "high"
    window.collect_settings().save()
    assert Settings.load().provider == "codex"


def test_radio_selection_is_used_by_worker(window, monkeypatch):
    monkeypatch.setattr(ui, "GenerationWorker", FakeWorker)
    window.provider_buttons["claude"].click()
    window.concept.setPlainText("An evening by the sea")
    window.start_generation()
    assert window.worker.settings.provider == "claude"
    assert Settings.load().provider == "claude"
    window.worker.finished.emit()


def test_local_selection_clears_cached_runpod_connection(window, config, monkeypatch):
    monkeypatch.setattr(ui, "GenerationWorker", FakeWorker)
    cloud = replace(config, cloud=True, comfy_url="https://old-8189.proxy.runpod.net",
                    browser_login_file="/private/cloud-login.json", token_env="RUNPOD_TOKEN")
    window.settings = window.display_settings = cloud
    window.update_summary()
    assert window.location_buttons[True].isChecked()
    window.location_buttons[False].click()
    assert window.endpoint.text() == "http://127.0.0.1:8188"
    assert window.template.currentIndex() == -1  # Local has no workflow of its own yet.
    window.template.setCurrentIndex(window.template.findData(config.template_file))
    window.concept.setPlainText("A garden")
    window.start_generation()
    selected = window.worker.settings
    assert not selected.cloud and not selected.managed_runpod
    assert selected.comfy_url == "http://127.0.0.1:8188"
    assert selected.browser_login_file == ""
    assert selected.token_env != "RUNPOD_TOKEN"
    assert all(b.isEnabled() for b in window.location_buttons.values())  # Draft stays editable.


def test_location_connections_survive_switches_and_restart(window, config):
    cloud = replace(config, cloud=True, comfy_url="https://old-8189.proxy.runpod.net",
                    browser_login_file="/private/cloud-login.json", token_env="CLOUD_TOKEN")
    window.settings = window.display_settings = cloud
    window.update_summary()
    window.location_buttons[False].click()
    window.endpoint.setText("http://192.168.1.10:8188")
    window.location_buttons[True].click()
    assert window.collect_settings().comfy_url == cloud.comfy_url
    assert window.collect_settings().browser_login_file == cloud.browser_login_file
    window.collect_settings().save()
    window.settings = window.display_settings = Settings.load()
    window.update_summary()
    window.location_buttons[False].click()
    assert window.collect_settings().comfy_url == "http://192.168.1.10:8188"
    assert window.collect_settings().browser_login_file == ""


def test_new_cloud_selection_uses_managed_setup(window):
    window.location_buttons[True].click()
    assert window.collect_settings().managed_runpod
    assert not window.endpoint.isEnabled()
    assert "Assigned" in window.endpoint.text()


def test_settings_local_toggle_removes_cloud_address(app, config):
    dialog = ui.SettingsDialog(replace(config, cloud=True,
                                      comfy_url="https://old-8189.proxy.runpod.net",
                                      browser_login_file="/private/cloud-login.json"))
    dialog.cloud.setChecked(False)
    selected = dialog.collect_settings()
    assert selected.comfy_url == "http://127.0.0.1:8188"
    assert selected.browser_login_file == ""
    assert not selected.cloud and not selected.managed_runpod
    dialog.deleteLater()


def test_settings_runpod_import_links_credentials_without_reading_them(
    app, config, monkeypatch, tmp_path
):
    (tmp_path / "gpu-session.json").write_text(
        json.dumps({"pod_id": "example", "url": "https://example-8189.proxy.runpod.net"})
    )
    # Import checks the path only. Credentials are read by the adapter at connection time.
    (tmp_path / "browser-login.json").write_text("not opened by the UI")
    monkeypatch.setattr(ui.QFileDialog, "getExistingDirectory", lambda *a: str(tmp_path))
    dialog = ui.SettingsDialog(config)
    dialog.import_runpod()
    dialog.accept()
    saved = Settings.load()
    assert saved.cloud
    assert saved.comfy_url == "https://example-8189.proxy.runpod.net"
    assert saved.browser_login_file == str(tmp_path / "browser-login.json")
    assert "not opened by the UI" not in json.dumps(saved.public_dict())
    dialog.deleteLater()


def test_empty_template_folder_does_not_browse_working_directory(window):
    window.template_folder.setText("")
    window.refresh_templates()
    assert [window.template.itemText(i) for i in range(window.template.count())] == [SCENE_LABEL, REFMOD_LABEL,
                                                                                  *(label for _, label, _ in BUNDLED)]


@pytest.mark.parametrize("folder", ["", "missing"])
def test_builtin_selection_persists_without_workflow_folder(window, tmp_path, folder):
    folder_text = str(tmp_path / folder) if folder else ""
    window.template_folder.setText(folder_text)
    window.refresh_templates()
    path = str(scene_template_path())
    assert [window.template.itemText(i) for i in range(window.template.count())] == [SCENE_LABEL, REFMOD_LABEL,
                                                                                  *(label for _, label, _ in BUNDLED)]
    window.template.setCurrentIndex(window.template.findData(path))
    selected = window.collect_settings()
    selected.validate()
    selected.save()
    loaded = Settings.load()
    assert loaded.template_file == path and loaded.template_folder == folder_text
    window.show_workflow(loaded)
    assert window.template.currentData() == path


def test_builtin_and_existing_user_scene_copy_are_distinct(window, config):
    from simplegen.h3scene.__main__ import template
    user_path = template(config.template_folder)
    window.refresh_templates()
    assert window.template.findText(user_path.name) >= 0
    assert window.template.findText(SCENE_LABEL) >= 0
    window.template.setCurrentIndex(window.template.findData(str(user_path)))
    window.refresh_templates()
    assert window.collect_settings().template_file == str(user_path)


def test_builtin_write_failure_omits_entry_without_crashing(window, monkeypatch, caplog):
    from simplegen import scheduling
    scene_template_path().unlink()

    def denied(*args):
        raise PermissionError("read-only app home")

    monkeypatch.setattr(scheduling, "atomic_bytes", denied)
    window.refresh_templates()
    assert window.template.findText(SCENE_LABEL) == -1
    assert window.template.findText("workflow.json") >= 0
    assert "read-only app home" in caplog.text


def test_builtin_stays_available_and_selection_is_per_destination(window, monkeypatch):
    monkeypatch.setattr(window, "refresh_runpod_catalog", lambda: None)
    path = str(scene_template_path())
    window.template.setCurrentIndex(window.template.findData(path))
    window.select_location(True)
    assert window.template.findText(SCENE_LABEL) >= 0
    assert window.template.currentData() != path
    window.template.setCurrentIndex(window.template.findData(path))
    window.select_location(False)
    assert window.template.currentData() == path
    window.collect_settings().save()
    window.settings = Settings.load()
    window.select_location(True)
    assert window.template.currentData() == path


def test_builtin_edit_as_new_run_keeps_snapshot_without_source(window, monkeypatch):
    from datetime import UTC, datetime, timedelta

    from simplegen import builtin_workflows

    window.template_folder.setText("")
    window.refresh_templates()
    path = str(scene_template_path())
    window.template.setCurrentIndex(window.template.findData(path))
    first = window.scheduler.add(window.collect_settings(), "Original", datetime.now(UTC) + timedelta(hours=1),
                                 preflight=PASSED)
    folder = Path(first["folder"])
    state = json.loads((folder / "state.json").read_text())
    scene_template_path().unlink()
    monkeypatch.setattr(builtin_workflows, "ensure_scene_template", lambda: None)
    window.stage_saved_run(folder, state, "Revised")
    assert window.template.currentData() == path
    assert "Saved snapshot" in window.template.currentText()
    window.schedule_enabled.setChecked(True)
    window.start_generation()
    second = next(r for r in window.scheduler.records() if r["id"] != first["id"])
    assert (Path(second["folder"]) / "template.original.json").read_bytes() == (
        folder / "template.original.json").read_bytes()


def test_worker_executes_off_ui_thread(app, config, monkeypatch, tmp_path):
    engine = types.ModuleType("simplegen.engine")
    thread_ids = []
    expected = {"folder": str(tmp_path), "final_path": "final.mp4", "metadata": {}}

    def run(settings, concept, on_event, cancel, resume_folder=None, detach=None, choose_gpu=None):
        assert isinstance(detach, threading.Event)
        assert callable(choose_gpu)
        thread_ids.append(threading.get_ident())
        on_event({"stage": "Complete", "progress": 100})
        return expected

    engine.run = run
    monkeypatch.setitem(sys.modules, "simplegen.engine", engine)
    worker = ui.GenerationWorker(config, "concept")
    results = []
    worker.succeeded.connect(results.append)
    worker.start()
    deadline = time.monotonic() + 3
    while worker.isRunning() and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.005)
    app.processEvents()
    assert worker.wait(1000)
    assert results == [expected]
    assert thread_ids != [threading.get_ident()]
    worker.deleteLater()


def pump(app, finished):
    deadline = time.monotonic() + 3
    while not finished() and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.005)
    app.processEvents()
    assert finished()


def test_managed_settings_enable_cloud_and_save_only_links(app, config):
    dialog = ui.SettingsDialog(config)
    dialog.managed_runpod.setChecked(True)
    dialog.url.setText("")
    dialog.runpod_config_file.setText("/linked/controller.json")
    dialog.runpod_credentials_file.setText("/linked/credentials.json")
    dialog.runpod_schema_file.setText("/linked/object_info.json")
    assert dialog.cloud.isChecked() and not dialog.cloud.isEnabled()
    assert not dialog.url.isEnabled()
    dialog.accept()
    saved = Settings.load()
    assert saved.managed_runpod and saved.cloud
    assert saved.runpod_credentials_file == "/linked/credentials.json"
    assert saved.comfy_url == ""
    dialog.deleteLater()


def test_scaffold_and_setup_checks_run_off_ui_thread(app, config, monkeypatch):
    credentials = types.ModuleType("simplegen.credentials")
    thread_ids = []
    def scaffold():
        thread_ids.append(threading.get_ident())
        return {"config_file": "/new/controller.json", "credentials_file": "/new/credentials.json",
                "schema_file": "/new/object_info.json"}
    def check(settings):
        thread_ids.append(threading.get_ident())
        assert settings.runpod_config_file == "/new/controller.json"
        return {"ready": False, "issues": ["Set the manager image digest."]}
    credentials.scaffold_runpod = scaffold
    credentials.check_setup = check
    monkeypatch.setitem(sys.modules, "simplegen.credentials", credentials)
    dialog = ui.SettingsDialog(config)
    dialog.scaffold_runpod()
    pump(app, lambda: dialog.operation_worker is None)
    assert dialog.runpod_credentials_file.text() == "/new/credentials.json"
    assert dialog.managed_runpod.isChecked()
    dialog.check_runpod()
    pump(app, lambda: dialog.operation_worker is None)
    assert "manager image digest" in dialog.setup_status.text()
    assert len(thread_ids) == 2 and threading.get_ident() not in thread_ids
    dialog.deleteLater()


def test_settings_close_waits_for_background_check_without_blocking(app, config):
    release = threading.Event()
    dialog = ui.SettingsDialog(config)
    dialog.start_operation(lambda: (release.wait(2) and {"ready": True}) or {}, dialog.setup_checked)
    event = QCloseEvent()
    dialog.closeEvent(event)
    assert not event.isAccepted() and dialog.operation_worker is not None
    release.set()
    pump(app, lambda: dialog.operation_worker is None)
    assert dialog.result() == QDialog.DialogCode.Rejected
    dialog.deleteLater()


def test_gpu_choice_requires_explicit_selection_and_shows_unknown_estimates(app):
    option = {"gpu_id": "GPU A", "gpu_count": 1, "vram_gb": 96, "hourly_usd": 2.5,
              "cpu_hourly_usd": 0.1}
    dialog = ui.GPUChoiceDialog([option])
    assert dialog.choice() is None and not dialog.rent_button.isEnabled()
    assert any("unknown" in label.text() for label in dialog.findChildren(ui.QLabel))
    dialog.group.button(0).click()
    assert dialog.rent_button.isEnabled()
    assert dialog.choice() == option
    dialog.reject()
    dialog.deleteLater()


@pytest.mark.parametrize("accepted", [False, True])
def test_choice_dialog_delivers_only_explicit_accepted_selection(window, monkeypatch, accepted):
    option = {"gpu_id": "GPU A", "hourly_usd": 2.5}
    window.worker = FakeWorker(window.settings, "concept", parent=window)
    def execute(dialog):
        if accepted:
            dialog.group.button(0).click()
        return QDialog.DialogCode.Accepted if accepted else QDialog.DialogCode.Rejected
    monkeypatch.setattr(ui.GPUChoiceDialog, "exec", execute)
    request = {"options": [option], "ready": threading.Event(), "choice": None}
    window.choose_gpu(request)
    assert request["ready"].is_set()
    assert request["choice"] == (option if accepted else None)
    window.worker = None


@pytest.mark.parametrize("signal", ["cancel_event", "detach_event"])
def test_worker_resource_wait_exits_without_selection_when_interrupted(app, config, signal):
    worker = ui.GenerationWorker(config, "concept")
    result = []
    waiter = threading.Thread(target=lambda: result.append(worker.choose_gpu([{"gpu_id": "GPU A"}])))
    waiter.start()
    getattr(worker, signal).set()
    waiter.join(timeout=1)
    assert not waiter.is_alive() and result == [None]
    worker.deleteLater()


def test_managed_shutdown_is_explicit_and_does_not_detach(window, monkeypatch):
    monkeypatch.setattr(ui, "GenerationWorker", FakeWorker)
    window.settings = replace(window.settings, managed_runpod=True)
    window.batch_mode.setCurrentIndex(window.batch_mode.findData("one_off"))
    window.concept.setPlainText("A concept")
    window.start_generation()
    assert window.cancel_button.text() == "Shut down compute"
    monkeypatch.setattr(window, "confirm_shutdown", lambda: False)
    window.cancel_generation()
    assert not window.worker.cancel_event.is_set()
    monkeypatch.setattr(window, "confirm_shutdown", lambda: True)
    window.cancel_generation()
    assert window.worker.cancel_event.is_set()
    assert not window.worker.detach_event.is_set()
    window.worker = None


def test_managed_resume_restores_saved_links_and_close_detaches(window, config, monkeypatch, tmp_path):
    monkeypatch.setattr(ui, "GenerationWorker", FakeWorker)
    folder = tmp_path / "managed-run"
    folder.mkdir()
    saved = replace(config, managed_runpod=True, runpod_credentials_file="/original/credentials.json")
    (folder / "state.json").write_text(json.dumps({"settings": asdict(saved)}))
    window.start_generation(folder)
    worker = window.worker
    assert worker.settings.managed_runpod
    assert worker.settings.runpod_credentials_file == "/original/credentials.json"
    assert not window.settings.managed_runpod
    event = QCloseEvent()
    window.closeEvent(event)
    assert worker.detach_event.is_set() and not worker.cancel_event.is_set()
    worker.finished.emit()


def test_cloud_actions_and_polling_run_in_background_preserve_video_status(window, app, monkeypatch, tmp_path):
    engine = types.ModuleType("simplegen.engine")
    calls = []
    state = {"phase": "ready", "downloads_verified": True, "idle_remaining_seconds": 240,
             "billing_confirmed_stopped": False}
    def action(folder, action):
        calls.append((folder, action, threading.get_ident()))
        return dict(state)
    engine.cloud_action = action
    monkeypatch.setitem(sys.modules, "simplegen.engine", engine)
    window.display_settings = replace(window.settings, managed_runpod=True)
    window.current_folder = tmp_path
    window.status.setText("Complete. Saved to /local/final.mp4")
    window.update_summary()
    window.poll_cloud_status()
    pump(app, lambda: window.cloud_worker is None)
    assert "240s" in window.cloud_banner.text()
    assert "verified" in window.cloud_banner.text()
    assert window.status.text() == "Complete. Saved to /local/final.mp4"
    assert calls[0][:2] == (tmp_path, "status")
    assert calls[0][2] != threading.get_ident()
    state.update(billing_confirmed_stopped=True, phase="closed")
    window.poll_cloud_status()
    pump(app, lambda: window.cloud_worker is None)
    assert "retirement confirmed" in window.cloud_banner.text()
    assert not window.cloud_timer.isActive()
    assert not window.cloud_shutdown_button.isEnabled()
    before = len(calls)
    window.poll_cloud_status()
    assert len(calls) == before


def test_cloud_status_error_keeps_completed_video_message(window):
    window.status.setText("Complete. Saved to /local/final.mp4")
    window.cloud_action_failed("Provider unavailable; retry later.")
    assert window.status.text() == "Complete. Saved to /local/final.mp4"
    assert "Billing may continue" in window.cloud_banner.text()


def test_idle_shutdown_calls_cloud_adapter_after_explicit_confirmation(window, app, monkeypatch, tmp_path):
    engine = types.ModuleType("simplegen.engine")
    calls = []
    engine.cloud_action = lambda folder, action: calls.append((folder, action)) or {"phase": "stopping"}
    monkeypatch.setitem(sys.modules, "simplegen.engine", engine)
    monkeypatch.setattr(window, "confirm_shutdown", lambda: True)
    window.current_folder = tmp_path
    window.display_settings = replace(window.settings, managed_runpod=True)
    window.cloud_action("cancel")
    pump(app, lambda: window.cloud_worker is None)
    assert calls == [(tmp_path, "cancel")]


def test_saving_an_edit_opened_from_the_scheduled_list_does_not_crash(window, app, config):
    """Regression (2026-09-30): Save in a schedule edit opened from the Scheduled list
    segfaulted. The finished save thread was deleteLater()'d and its last Python reference
    dropped; the deferred delete then ran in the list's outer event loop."""
    from datetime import UTC, datetime, timedelta

    from PySide6.QtCore import QTimer

    from simplegen.schedule_ui import ScheduleDialog
    for round_number in range(3):
        record = window.scheduler.add(config, f"Saved {round_number}", datetime.now(UTC) + timedelta(hours=1),
                                      preflight=PASSED)
        listing = ScheduleDialog(window.scheduler, window.refresh_schedule_count, window, edit=window.edit_schedule)

        def press_save(number=round_number):
            modal = QApplication.activeModalWidget()
            assert isinstance(modal, ui.ScheduleEditDialog)
            modal.concept.setPlainText(f"Edited {number}")
            modal.save()

        def pick_and_edit(target=record["id"], dialog=listing):
            for row in range(dialog.rows.topLevelItemCount()):
                item = dialog.rows.topLevelItem(row)
                if item.data(0, 0x0100)["id"] == target:
                    dialog.rows.setCurrentItem(item)
            QTimer.singleShot(150, press_save)
            dialog.edit_selected()
            QTimer.singleShot(50, dialog.accept)

        QTimer.singleShot(100, pick_and_edit)
        listing.exec()
        app.processEvents()
        assert window.scheduler.get(record["id"])["concept"] == f"Edited {round_number}"
        listing.deleteLater()
        app.processEvents()


def test_retire_worker_waits_and_keeps_the_worker_alive():
    from simplegen.qtutil import retire_worker

    class Owner:
        pass

    class Worker:
        waited = False

        def wait(self):
            self.waited = True

    owner, worker = Owner(), Worker()
    retire_worker(owner, worker)
    retire_worker(owner, None)
    assert worker.waited and owner._retired_workers == [worker]


def test_4k60_checkbox_needs_a_service_and_travels_with_the_request(window, config):
    assert not window.upscale.isEnabled() and not window.collect_settings().upscale
    window.settings = replace(window.settings, upscale_url="http://upscaler:8190")
    window.update_upscale_toggle()
    window.upscale.setChecked(True)
    assert window.collect_settings().upscale is True
    assert window.collect_settings().public_dict()["upscale"] is True


def test_upscale_method_sits_beside_4k60_with_speed_notes_and_travels_with_the_request(window, config):
    method = window.upscale_method
    labels = [method.itemText(i) for i in range(method.count())]
    assert labels == ["RTX Video Super Resolution (fastest)", "Remacri (fast)", "SeedVR2 3B (slow)"]
    assert method.currentData() == "remacri+rife" and not window.upscale.isChecked()
    assert not method.isEnabled()
    window.settings = replace(window.settings, upscale_url="http://upscaler:8190")
    window.update_upscale_toggle()
    assert not method.isEnabled()
    window.upscale.setChecked(True)
    assert method.isEnabled()
    method.setCurrentIndex(method.findData("seedvr2-3b+rife"))
    assert window.collect_settings().upscale_preset == "seedvr2-3b+rife"
    window.upscale.setChecked(False)
    assert not method.isEnabled()
    # A preset saved earlier that is not one of the preferred methods is kept, not silently replaced.
    window.show_upscale_method("ultrasharp+rife")
    assert method.currentData() == "ultrasharp+rife"
    assert window.collect_settings().upscale_preset == "ultrasharp+rife"
    # Settings saved before the method choice (empty preset) show and save the default, Remacri (fast).
    window.show_upscale_method("")
    assert method.currentData() == "remacri+rife"


def test_watch_offers_upscale_and_prefers_the_verified_4k60_version(window, tmp_path, monkeypatch):
    from simplegen import upscale
    folder = tmp_path / "run"
    (folder / "final" / "4k60").mkdir(parents=True)
    (folder / "state.json").write_text(json.dumps({"run_id": "r", "settings": {}}))
    final = folder / "final" / "a_final.mp4"
    final.write_bytes(b"x")
    window.display_video({"folder": str(folder), "final_path": "final/a_final.mp4",
                          "metadata": {"width": 672, "height": 384, "duration": 1.0, "size_bytes": 1}})
    assert not window.upscale_panel.upscale_button.isHidden()
    assert window.upscale_panel.version.isHidden()
    output = folder / "final" / "4k60" / "a_final_4k60.mp4"
    output.write_bytes(b"y")
    (folder / "final" / "4k60" / "manifest.json").write_text(json.dumps({"schema": 1, "outputs": [
        {"source": "final/a_final.mp4", "output": "final/4k60/a_final_4k60.mp4",
         "metadata": {"width": 3840, "height": 2194, "duration": 1.0, "size_bytes": 1}}]}))
    window.display_video({"folder": str(folder), "final_path": "final/a_final.mp4",
                          "metadata": {"width": 672, "height": 384, "duration": 1.0, "size_bytes": 1}})
    assert not window.upscale_panel.version.isHidden()
    assert window.upscale_panel.version.currentData() == "4k60"
    assert "4K60 version" in window.video_path.text() and "3840" in window.video_info.text()
    window.upscale_panel.version.setCurrentIndex(0)
    assert "Final video" in window.video_path.text()
    assert upscale.upscaled_version(final) == output.resolve()
