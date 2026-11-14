"""Batch UI behavior with no provider or paid calls."""

import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication, QDialog

from simplegen import ui
from simplegen.batch_ui import BatchDialog
from simplegen.settings import Settings


class Store:
    def __init__(self, folder):
        self.calls = []
        self.batch = {
            "id": "a" * 32, "folder": str(folder), "status": "prepared", "members": [],
            "gpu_options": [
                {"gpu_id": "GPU-A", "hourly_usd": 1.2, "vram_gb": 48,
                 "estimated_cost_low": 1, "estimated_cost_high": 2,
                 "runtime_seconds_low": 600, "runtime_seconds_high": 900},
                {"gpu_id": "GPU-B", "hourly_usd": 1.5, "vram_gb": 80,
                 "estimated_cost_low": 2, "estimated_cost_high": 3,
                 "runtime_seconds_low": 500, "runtime_seconds_high": 800},
            ],
            "gpu_choices": [], "priority": "high", "retry_minutes": 5,
            "due_at": None, "next_check_at": None, "waiting_reason": None,
        }

    def list(self):
        return [self.batch]

    def get(self, _id):
        return self.batch

    def configure(self, identity, **choices):
        self.calls.append(("configure", identity, choices))
        self.batch.update(choices)

    def arm(self, identity):
        self.calls.append(("arm", identity))
        self.batch["status"] = "armed"

    def run_now(self, identity):
        self.calls.append(("run_now", identity))
        self.batch["status"] = "armed"


def test_prepared_batch_requires_two_distinct_approved_gpus_and_saves_local_start(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    store = Store(tmp_path)
    dialog = BatchDialog(store, lambda: None)
    try:
        assert "No agent" in dialog.detail.text()
        dialog.gpu[0].setCurrentIndex(0)
        dialog.gpu[1].setCurrentIndex(1)
        assert dialog.retry_minutes.value() == 5
        future = datetime.now().astimezone() + timedelta(hours=2)
        dialog.due.setDate(future.date())
        dialog.due.setTime(future.time().replace(second=0, microsecond=0))
        dialog.activate(False)
        assert [call[0] for call in store.calls] == ["configure", "arm"]
        config = store.calls[0][2]
        assert [entry["gpu_id"] for entry in config["gpu_choices"]] == ["GPU-A", "GPU-B"]
        assert [entry["max_hourly"] for entry in config["gpu_choices"]] == [1.2, 1.5]
        assert datetime.fromisoformat(config["due_at"]).tzinfo is not None
        assert datetime.fromisoformat(config["due_at"]) > datetime.now(UTC)
        assert "Local start:" in dialog.detail.text()
    finally:
        dialog.close()
        dialog.deleteLater()
        app.processEvents()


def test_estimates_show_only_two_selected_gpu_types(tmp_path):
    app = QApplication.instance() or QApplication([])
    store = Store(tmp_path)
    store.batch["gpu_options"].append({
        "gpu_id": "GPU-C", "hourly_usd": 2.2, "vram_gb": 96,
        "estimated_cost_low": 3, "estimated_cost_high": 5,
        "estimated_session_seconds_low": 2400, "estimated_session_seconds_high": 4200,
    })
    dialog = BatchDialog(store, lambda: None)
    try:
        assert dialog.gpu[0].currentIndex() == -1
        assert "estimated" not in dialog.detail.text().lower()
        dialog.gpu[1].setCurrentIndex(2)
        assert dialog.gpu[1].currentData() == "GPU-C"
    finally:
        dialog.close()
        dialog.deleteLater()
        app.processEvents()


def test_start_now_does_not_invoke_provider_in_dialog(tmp_path):
    app = QApplication.instance() or QApplication([])
    store = Store(tmp_path)
    dialog = BatchDialog(store, lambda: None)
    try:
        dialog.gpu[0].setCurrentIndex(0)
        dialog.gpu[1].setCurrentIndex(1)
        dialog.activate(True)
        assert [call[0] for call in store.calls] == ["configure", "run_now"]
        assert store.calls[0][2]["due_at"] is None
    finally:
        dialog.close()
        dialog.deleteLater()
        app.processEvents()


def test_preparation_disables_membership_and_close_detaches_without_warning(tmp_path, monkeypatch):
    import json
    import time

    app = QApplication.instance() or QApplication([])
    run = tmp_path / "run"
    run.mkdir()
    (run / "state.json").write_text(json.dumps({"settings": {"quantity": 2}}))

    class PreparingStore(Store):
        def prepare(self, _id, _emit, _cancel, detach):
            while not detach.is_set():
                time.sleep(0.01)
            raise RuntimeError("Detached from preparation")

    store = PreparingStore(tmp_path)
    store.batch["status"] = "preparing"  # Persisted state after a prior desktop exited.
    store.batch["members"] = [{"id": "member", "folder": str(run), "concept": "Test", "status": "draft"}]
    warnings = []
    monkeypatch.setattr("simplegen.batch_ui.QMessageBox.warning", lambda *args: warnings.append(args))
    dialog = BatchDialog(store, lambda: None)
    try:
        assert dialog.prepare_button.isEnabled()  # Recovery is explicit, never automatic.
        dialog.prepare()
        assert dialog.worker is not None
        assert not dialog.batch_picker.isEnabled()
        assert not dialog.new_button.isEnabled()
        assert not dialog.edit_button.isEnabled()
        assert not dialog.remove_button.isEnabled()
        dialog.refresh()
        assert not dialog.edit_button.isEnabled() and not dialog.remove_button.isEnabled()
        dialog.reject()
        until = time.monotonic() + 3
        while dialog.worker is not None and time.monotonic() < until:
            app.processEvents()
            time.sleep(0.01)
        assert dialog.worker is None
        assert warnings == []
    finally:
        if dialog.worker is not None:
            dialog.worker.detach_event.set()
            dialog.worker.wait(3000)
        dialog.deleteLater()
        app.processEvents()


@pytest.mark.usefixtures("_no_desktop_audio")
def test_managed_generate_defaults_to_saved_batch_without_provider(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    monkeypatch.setenv("SIMPLEGEN_HOME", str(tmp_path / "home"))
    project = tmp_path / "project"
    templates = tmp_path / "templates"
    project.mkdir()
    templates.mkdir()
    workflow = templates / "workflow.json"
    workflow.write_text("{}")
    settings = Settings(project_folder=str(project), template_folder=str(templates),
                        template_file=str(workflow), output_folder=str(tmp_path / "outputs"),
                        managed_runpod=True, global_desktop=True)
    window = ui.MainWindow(settings)
    window.batch_timer.stop()
    opened = []
    monkeypatch.setattr(window, "show_batches", opened.append)
    monkeypatch.setattr(ui.QInputDialog, "getItem", lambda *args: ("New batch", True))
    monkeypatch.setattr("simplegen.engine.run", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("generation started")))
    try:
        assert window.batch_mode.currentData() == "batch"
        window.concept.setPlainText("A quiet lake")
        window.start_generation()
        import time
        until = time.monotonic() + 3
        while window.batch_stage_worker is not None and time.monotonic() < until:
            app.processEvents()
            time.sleep(0.01)
        assert window.batch_stage_worker is None
        assert len(opened) == 1
        batch = window.get_batch_store().get(opened[0])
        assert batch["status"] == "draft"
        assert len(batch["members"]) == 1
        member = batch["members"][0]
        assert member["concept"] == "A quiet lake"
        assert (Path(member["folder"]) / "state.json").is_file()
    finally:
        window.close()
        window.deleteLater()
        app.processEvents()


@pytest.mark.usefixtures("_no_desktop_audio")
def test_due_batch_does_not_wait_for_local_work_and_dispatches_off_ui_thread(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    monkeypatch.setenv("SIMPLEGEN_HOME", str(tmp_path / "home"))
    project = tmp_path / "project"
    templates = tmp_path / "templates"
    project.mkdir()
    templates.mkdir()
    workflow = templates / "workflow.json"
    workflow.write_text("{}")
    settings = Settings(project_folder=str(project), template_folder=str(templates),
                        template_file=str(workflow), output_folder=str(tmp_path / "outputs"))
    window = ui.MainWindow(settings)
    window.batch_timer.stop()
    window.schedule_timer.stop()
    thread_ids = []
    import threading
    release = threading.Event()
    row = {"id": "a" * 32, "folder": str(tmp_path / "batch"), "status": "armed",
           "settings": Settings(**{**settings.public_dict(), "managed_runpod": True}).public_dict(),
           "next_check_at": (datetime.now(UTC) - timedelta(minutes=1)).isoformat(),
           "members": []}
    class DueStore:
        def list(self):
            return [row]
        def recover_handoffs(self):
            pass
        def service_inbox(self, _id):
            return None
        def run_batch(self, _id, _emit, _cancel, _detach):
            thread_ids.append(threading.get_ident())
            release.wait(2)
            return {**row, "status": "waiting", "waiting_reason": "Both GPUs unavailable",
                    "next_check_at": (datetime.now(UTC) + timedelta(minutes=5)).isoformat()}
    monkeypatch.setattr("simplegen.batches.registered_stores", lambda: [DueStore()])
    due_local = [{"status": "scheduled", "due_at": (datetime.now(UTC) - timedelta(minutes=1)).isoformat()}]
    monkeypatch.setattr(window.scheduler, "records", lambda: due_local)
    try:
        # Phase 4: separate slots. A due local schedule (or a local run) no longer blocks cloud dispatch.
        window.worker = object()
        window.cloud_worker = object()
        window.poll_batches()
        assert window.batch_worker is None
        window.cloud_worker = None
        window.override_worker = object()
        window.poll_batches()
        assert window.batch_worker is None
        window.override_worker = None
        modal = QDialog(window)
        modal.setModal(True)
        modal.show()
        app.processEvents()
        window.poll_batches()
        assert window.batch_worker is None
        modal.close()
        modal.deleteLater()
        app.processEvents()
        window.poll_batches()
        import time
        until = time.monotonic() + 3
        while not thread_ids and time.monotonic() < until:
            app.processEvents()
            time.sleep(0.01)
        assert thread_ids and window.batch_worker is not None
        assert window.generate_button.isEnabled()  # The draft stays available while the batch runs.
        window.worker = None
        window.update_jobs_status()
        # The main Cancel never targets the batch; it has its own explicit control.
        assert not window.cancel_button.isEnabled() and not window.stop_batch_button.isHidden()
        release.set()
        until = time.monotonic() + 3
        while window.batch_worker is not None and time.monotonic() < until:
            app.processEvents()
            time.sleep(0.01)
        assert len(thread_ids) == 1 and thread_ids[0] != threading.get_ident()
        assert window.generate_button.isEnabled()
        assert not window.cancel_button.isEnabled()
        assert "Both GPUs unavailable" in window.status.text()
        assert "Next check" in window.status.text()
    finally:
        release.set()
        window.close()
        window.deleteLater()
        app.processEvents()


@pytest.mark.usefixtures("_no_desktop_audio")
def test_active_batch_owns_cloud_controls_and_complete_status(tmp_path, monkeypatch):
    import threading
    from types import SimpleNamespace

    app = QApplication.instance() or QApplication([])
    monkeypatch.setenv("SIMPLEGEN_HOME", str(tmp_path / "home"))
    window = ui.MainWindow(Settings(output_folder=str(tmp_path / "outputs")))
    window.batch_timer.stop()
    window.cloud_timer.stop()
    window.global_timer.stop()
    owner = tmp_path / "batch-owner"
    owner.mkdir()
    window.current_folder = owner
    window.display_settings = Settings(managed_runpod=True, global_desktop=True)
    worker = SimpleNamespace(cancel_event=threading.Event(), detach_event=threading.Event())
    window.batch_worker = worker
    shown = []
    monkeypatch.setattr(window, "show_result", lambda result, **kwargs: shown.append(result))
    monkeypatch.setattr(window, "confirm_shutdown", lambda batch=False: True)
    try:
        window.set_running(True)
        window.update_summary()
        assert window.cancel_button.isEnabled()
        assert not window.cloud_status_button.isEnabled()
        assert not window.cloud_shutdown_button.isEnabled()
        window.poll_cloud_status()
        window.cloud_action("status")
        assert window.cloud_worker is None
        window.cloud_action("cancel")
        assert worker.cancel_event.is_set()
        result = {"folder": str(tmp_path / "child"), "final_path": str(tmp_path / "child/final.mp4")}
        window.batch_dispatched({"status": "complete", "folder": str(owner),
                                 "members": [{"result": result}]})
        assert shown == [result]
        assert window.current_folder == owner  # The batch never replaces the draft/local context.
        assert "Batch complete" in window.status.text()
    finally:
        window.batch_worker = None
        window.close()
        window.deleteLater()
        app.processEvents()
