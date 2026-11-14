"""GPU selection is manual; preparation gates dispatch, not catalog selection."""

import os
import threading
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QDialog

from simplegen.batch_ui import BatchDialog


def pump_until(app, predicate):
    deadline = time.monotonic() + 3
    while not predicate() and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.01)
    app.processEvents()
    assert predicate()


class CatalogStore:
    def __init__(self, folder):
        self.calls = []
        self.catalog_threads = []
        self.batch = {
            "id": "a" * 32, "folder": str(folder), "status": "draft",
            "members": [{"id": "job", "folder": str(folder), "concept": "Scene", "status": "draft"}],
            "gpu_options": [], "gpu_choices": [], "priority": "high", "retry_minutes": 5,
        }

    def list(self):
        return [self.batch]

    def get(self, _identity):
        return self.batch

    def catalog(self, _identity):
        self.calls.append("catalog")
        self.catalog_threads.append(threading.get_ident())
        return [
            {"gpu_id": "GPU-EXPENSIVE", "hourly_usd": 4.5, "vram_gb": 80,
             "availability": "HIGH", "available_data_centers": ["US-TX-1", "AP-IN-1", "EU-RO-1"],
             "available_location_count": 3},
            {"gpu_id": "GPU-CHEAP", "hourly_usd": 1.5, "vram_gb": 48,
             "availability": "UNAVAILABLE"},
        ]

    def configure(self, _identity, **values):
        self.calls.append(("configure", values))
        self.batch.update(values)

    def arm(self, _identity):
        self.calls.append("arm")

    def run_now(self, _identity):
        self.calls.append("run_now")


def test_catalog_lists_unassessed_gpus_without_enabling_approval(tmp_path):
    app = QApplication.instance() or QApplication([])
    store = CatalogStore(tmp_path)
    dialog = BatchDialog(store, lambda: None)
    try:
        pump_until(app, lambda: dialog.catalog.topLevelItemCount() == 2)
        assert dialog.catalog.topLevelItem(1).text(0) == "GPU-EXPENSIVE"
        assert dialog.catalog.topLevelItem(1).text(1) == "$4.50"
        assert dialog.catalog.topLevelItem(1).text(2) == "HIGH"
        assert dialog.catalog.topLevelItem(1).text(5) == "3"
        assert "AP-IN-1" in dialog.catalog.topLevelItem(1).toolTip(5)
        assert "across 3 locations" in dialog.catalog_status.text()
        assert "Manual choice" in dialog.catalog.topLevelItem(1).text(4)
        assert not dialog.arm_button.isEnabled()
        assert dialog.gpu[0].count() == 2
        assert dialog.gpu[0].currentIndex() == -1
        assert store.catalog_threads[0] != threading.get_ident()
        dialog.catalog_refresh.click()
        pump_until(app, lambda: store.calls.count("catalog") >= 2 and dialog.catalog_refresh.isEnabled())
        assert all(call == "catalog" for call in store.calls)
    finally:
        dialog.close()
        dialog.deleteLater()
        app.processEvents()


def test_batch_ceiling_can_exceed_profile_default_and_is_saved_on_approval(tmp_path):
    app = QApplication.instance() or QApplication([])
    store = CatalogStore(tmp_path)
    store.batch["status"] = "prepared"
    store.batch["gpu_options"] = [
        {"gpu_id": "GPU-CHEAP", "hourly_usd": 1.5, "vram_gb": 48},
        {"gpu_id": "GPU-EXPENSIVE", "hourly_usd": 4.5, "vram_gb": 80},
    ]
    dialog = BatchDialog(store, lambda: None)
    try:
        dialog.max_hourly.setValue(5.0)
        dialog.gpu[1].setCurrentIndex(1)
        assert dialog.caps[1].value() == 4.5
        dialog.gpu[0].setCurrentIndex(1)
        dialog.gpu[1].setCurrentIndex(0)
        selected = [combo.currentData() for combo in dialog.gpu]
        dialog.refresh_catalog()
        pump_until(app, lambda: dialog.catalog.topLevelItemCount() == 2)
        assert [combo.currentData() for combo in dialog.gpu] == selected
        dialog.gpu[0].setCurrentIndex(0)
        dialog.gpu[1].setCurrentIndex(1)
        dialog.activate(True)
        configure = next(call[1] for call in store.calls if isinstance(call, tuple))
        assert configure["max_hourly"] == 5.0
        assert configure["gpu_choices"][1]["max_hourly"] == 4.5
        assert "run_now" in store.calls
    finally:
        dialog.close()
        dialog.deleteLater()
        app.processEvents()


def test_catalog_failure_is_visible_without_provider_side_effects(tmp_path):
    app = QApplication.instance() or QApplication([])
    store = CatalogStore(tmp_path)
    def fail(_identity):
        raise RuntimeError("catalog unavailable")
    store.catalog = fail
    dialog = BatchDialog(store, lambda: None)
    try:
        pump_until(app, lambda: "Catalog check failed" in dialog.catalog_status.text())
        assert "catalog unavailable" in dialog.catalog_status.text()
        assert not dialog.arm_button.isEnabled()
    finally:
        dialog.close()
        dialog.deleteLater()
        app.processEvents()


def test_close_waits_for_catalog_thread_without_blocking_ui(tmp_path):
    app = QApplication.instance() or QApplication([])
    store = CatalogStore(tmp_path)
    started = threading.Event()
    release = threading.Event()

    def slow_catalog(_identity):
        started.set()
        release.wait(2)
        return []

    store.catalog = slow_catalog
    dialog = BatchDialog(store, lambda: None)
    dialog.show()
    try:
        pump_until(app, started.is_set)
        dialog.close()
        assert dialog.catalog_worker is not None
        assert dialog.isVisible()
        assert "Finishing catalog check" in dialog.status.text()
        release.set()
        pump_until(app, lambda: dialog.catalog_worker is None)
        assert dialog.result() == QDialog.DialogCode.Rejected
        assert not dialog.isVisible()
    finally:
        release.set()
        dialog.close()
        dialog.deleteLater()
        app.processEvents()


def test_start_now_waits_for_catalog_and_preserves_accept_result(tmp_path):
    app = QApplication.instance() or QApplication([])
    store = CatalogStore(tmp_path)
    store.batch['status'] = 'prepared'
    store.batch['gpu_options'] = store.catalog(store.batch['id'])
    started, release = threading.Event(), threading.Event()
    def delayed(_identity):
        started.set()
        release.wait(5)
        return store.batch['gpu_options']
    store.catalog = delayed
    dialog = BatchDialog(store, lambda: None)
    dialog.show()
    try:
        pump_until(app, started.is_set)
        dialog.max_hourly.setValue(5)
        dialog.gpu[0].setCurrentIndex(0)
        dialog.gpu[1].setCurrentIndex(1)
        dialog.activate(True)
        assert 'run_now' in store.calls
        assert dialog.isVisible()
        assert dialog.catalog_worker is not None
        assert 'Finishing catalog check' in dialog.status.text()
        assert not dialog.now_button.isEnabled()
        release.set()
        pump_until(app, lambda: dialog.catalog_worker is None and not dialog.isVisible())
        assert dialog.result() == QDialog.DialogCode.Accepted
    finally:
        release.set()
        pump_until(app, lambda: dialog.catalog_worker is None)
        dialog.deleteLater()
        app.processEvents()
