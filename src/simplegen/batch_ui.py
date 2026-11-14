"""Desktop controls for durable managed RunPod batches."""

from __future__ import annotations

import json
import threading
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QDate, QDateTime, Qt, QThread, QTime, QTimer, QTimeZone, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
)

from .qtutil import retire_worker
from .schedule_ui import date_field, selected_time
from .scheduling import display_time
from .settings import Settings


def catalog_summary(rows):
    available = sum(row.get("availability") in {"LOW", "MEDIUM", "HIGH"} for row in rows)
    locations = {place for row in rows for place in row.get("available_data_centers", [])}
    coverage = f" across {len(locations)} locations" if locations else ""
    return (f"Live catalog checked: {len(rows)} secure GPU types; {available} available{coverage}. "
            "Global Volume · all regions. Location counts are not GPU inventory counts.")


class CatalogWorker(QThread):
    succeeded = Signal(object)
    failed = Signal(str)

    def __init__(self, operation, parent=None):
        super().__init__(parent)
        self.operation = operation

    def run(self):
        try:
            self.succeeded.emit(self.operation())
        except Exception as error:  # noqa: BLE001 — report provider failures in the dialog.
            self.failed.emit(str(error))


class BatchWorker(QThread):
    event = Signal(dict)
    succeeded = Signal(object)
    failed = Signal(str)

    def __init__(self, operation, parent=None):
        super().__init__(parent)
        self.operation = operation
        self.cancel_event = threading.Event()
        self.detach_event = threading.Event()

    def run(self):
        try:
            value = self.operation(self.event.emit, self.cancel_event, self.detach_event)
        except Exception as error:  # noqa: BLE001 — worker boundary reports background failures.
            self.failed.emit(str(error))
        else:
            self.succeeded.emit(value)


def batch_label(batch: dict) -> str:
    members = batch.get("members", [])
    name = members[0].get("concept", "Untitled")[:32] if members else "Empty batch"
    return f"{name} · {len(members)} job(s) · {batch.get('status', 'draft')}"


class MemberEditDialog(QDialog):
    def __init__(self, state: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Edit batch job as new revision")
        self.resize(610, 560)
        self.original = Settings(**state["settings"])
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Save creates a new run revision. Earlier files and history remain available."))
        form = QFormLayout()
        self.concept = QPlainTextEdit(state["concept"])
        form.addRow("Concept", self.concept)
        self.paths = {}
        for key, label in (("project_folder", "Project references"),
                           ("template_folder", "Workflow folder"),
                           ("template_file", "Workflow JSON")):
            field = QLineEdit(getattr(self.original, key))
            self.paths[key] = field
            form.addRow(label, field)
        self.provider = QComboBox()
        for name in ("codex", "claude", "grok"):
            self.provider.addItem(name.title(), name)
        self.provider.setCurrentIndex(self.provider.findData(self.original.provider))
        form.addRow("Agent", self.provider)
        self.model = QLineEdit(self.original.model)
        form.addRow("Model", self.model)
        self.reasoning = QLineEdit(self.original.reasoning)
        form.addRow("Reasoning", self.reasoning)
        self.steps = QSpinBox()
        self.steps.setRange(1, 100)
        self.steps.setValue(self.original.steps or 25)
        form.addRow("Steps", self.steps)
        self.megapixels = QDoubleSpinBox()
        self.megapixels.setRange(0.05, 16)
        self.megapixels.setValue(self.original.megapixels)
        form.addRow("Megapixels", self.megapixels)
        self.duration = QDoubleSpinBox()
        self.duration.setRange(1, 3600)
        self.duration.setValue(self.original.duration)
        form.addRow("Duration seconds", self.duration)
        self.aspect = QComboBox()
        from .dimensions import ASPECT_RATIOS
        for value, _ in ASPECT_RATIOS:
            self.aspect.addItem(value, value)
        self.aspect.setCurrentIndex(max(0, self.aspect.findData(self.original.aspect_ratio)))
        form.addRow("Aspect ratio", self.aspect)
        self.quantity = QSpinBox()
        self.quantity.setRange(1, 100)
        self.quantity.setValue(self.original.quantity)
        form.addRow("Qty", self.quantity)
        layout.addLayout(form)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def values(self):
        concept = self.concept.toPlainText().strip()
        if not concept:
            raise ValueError("Enter a concept.")
        settings = replace(self.original,
            **{key: field.text().strip() for key, field in self.paths.items()},
            provider=self.provider.currentData(), model=self.model.text().strip(),
            model_tier="Custom" if self.model.text().strip() != self.original.model else self.original.model_tier,
            reasoning=self.reasoning.text().strip(), steps=self.steps.value(),
            megapixels=self.megapixels.value(), duration=self.duration.value(),
            aspect_ratio=self.aspect.currentData(), quantity=self.quantity.value())
        settings.validate()
        from .agents import agent_problem
        problem = agent_problem(settings)
        if problem:
            raise ValueError(problem)
        workflow = Path(settings.template_file).expanduser()
        if not workflow.is_absolute():
            workflow = Path(settings.template_folder).expanduser() / workflow
        if not workflow.is_file():
            raise ValueError("Choose an existing workflow JSON.")
        return concept, settings


class BatchDialog(QDialog):
    """Manage membership and explicit GPU approval before arming a batch."""

    def __init__(self, store, changed, initial_id=None, parent=None, settings: Settings | None = None):
        super().__init__(parent)
        self.store, self.changed = store, changed
        self.worker = None
        self.catalog_worker = None
        self.pending_confirm = None
        self._closing = False
        self._close_result = QDialog.DialogCode.Rejected
        self.catalog_settings = settings
        self._catalog_identity = None
        self._catalog_closed = False
        self.setWindowTitle("RunPod batches & schedule")
        self.resize(830, 650)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Prepare every job before renting compute. Keep simpleGen open and this computer awake for scheduled starts."))
        top = QHBoxLayout()
        self.batch_picker = QComboBox()
        self.batch_picker.currentIndexChanged.connect(self.refresh)
        self.batch_picker.currentIndexChanged.connect(lambda *_: self.refresh_catalog())
        top.addWidget(self.batch_picker, 1)
        self.new_button = QPushButton("New batch")
        self.new_button.clicked.connect(self.new_batch)
        top.addWidget(self.new_button)
        layout.addLayout(top)
        self.jobs = QTreeWidget()
        self.jobs.setHeaderLabels(["Concept", "Qty", "Status", "Project folder"])
        self.jobs.setRootIsDecorated(False)
        self.jobs.setColumnWidth(0, 210)
        self.jobs.setColumnWidth(3, 240)
        layout.addWidget(self.jobs, 1)
        member_actions = QHBoxLayout()
        self.edit_button = QPushButton("Edit job…")
        self.remove_button = QPushButton("Remove from batch")
        self.edit_button.clicked.connect(self.edit_member)
        self.remove_button.clicked.connect(self.remove_member)
        member_actions.addWidget(self.edit_button)
        member_actions.addWidget(self.remove_button)
        member_actions.addStretch()
        layout.addLayout(member_actions)
        self.detail = QLabel()
        self.detail.setWordWrap(True)
        layout.addWidget(self.detail)
        catalog_heading = QHBoxLayout()
        catalog_heading.addWidget(QLabel("Global Volume GPU catalog — all Secure Cloud regions"))
        self.catalog_refresh = QPushButton("Refresh catalog")
        self.catalog_refresh.clicked.connect(self.refresh_catalog)
        catalog_heading.addWidget(self.catalog_refresh)
        layout.addLayout(catalog_heading)
        self.catalog_status = QLabel("Catalog has not been checked.")
        self.catalog_status.setWordWrap(True)
        layout.addWidget(self.catalog_status)
        self.catalog_options = {}
        self.catalog = QTreeWidget()
        self.catalog.setHeaderLabels(["GPU type", "Price / hour", "Availability", "VRAM", "Selection", "Locations"])
        self.catalog.setRootIsDecorated(False)
        self.catalog.setMaximumHeight(150)
        self.catalog.setMinimumHeight(95)
        for column, width in enumerate((320, 100, 100, 70, 210)):
            self.catalog.setColumnWidth(column, width)
        layout.addWidget(self.catalog)
        form = QFormLayout()
        self.max_hourly = QDoubleSpinBox()
        self.max_hourly.setRange(0.01, 1000)
        self.max_hourly.setDecimals(3)
        self.max_hourly.setPrefix("$ ")
        self.max_hourly.setSuffix(" / hour")
        self.max_hourly.setValue(3.0)
        self.max_hourly.setToolTip("This batch's maximum hourly rate. It does not change the global profile or an existing session.")
        self.max_hourly.valueChanged.connect(self._update_choice_cap_ranges)
        form.addRow("Batch hourly ceiling", self.max_hourly)
        self.max_total = QDoubleSpinBox()
        self.max_total.setRange(0, 100000)
        self.max_total.setDecimals(2)
        self.max_total.setPrefix("$ ")
        self.max_total.setSpecialValueText("No cap")
        self.max_total.setToolTip("Optional total spend cap. Additions that would exceed the projected total are "
                                  "rejected; crossing it alerts. Queued or running renders are never cancelled.")
        form.addRow("Spend cap (total)", self.max_total)
        self.gpu = [QComboBox(), QComboBox()]
        self.caps = [QDoubleSpinBox(), QDoubleSpinBox()]
        for index in range(2):
            row = QHBoxLayout()
            row.addWidget(self.gpu[index], 1)
            self.caps[index].setRange(0.01, 1000)
            self.caps[index].setDecimals(3)
            self.caps[index].setPrefix("$ ")
            self.caps[index].setSuffix(" / hour maximum")
            row.addWidget(self.caps[index])
            form.addRow(f"GPU choice {index + 1}", row)
        self.priority = QComboBox()
        self.priority.addItem("High — either approved available GPU", "high")
        self.priority.addItem("Low — wait for the cheaper approved GPU", "low")
        form.addRow("Priority", self.priority)
        self.retry_minutes = QSpinBox()
        self.retry_minutes.setRange(1, 1440)
        self.retry_minutes.setValue(5)
        self.retry_minutes.setSuffix(" minutes")
        form.addRow("Availability retry", self.retry_minutes)
        self.due = date_field()
        form.addRow("Local start (24h)", self.due)
        layout.addLayout(form)
        actions = QHBoxLayout()
        self.prepare_button = QPushButton("Prepare scripts & workflows")
        self.arm_button = QPushButton("Schedule batch")
        self.now_button = QPushButton("Start now")
        self.resume_button = QPushButton("Resume reviewed failure")
        self.pause_button = QPushButton("Pause waiting batch")
        self.cancel_button = QPushButton("Cancel batch…")
        self.add_running_button = QPushButton("Add to running batch / warm pod…")
        self.add_running_button.setToolTip("Prepare a new concept now and queue it on this batch's GPU "
                                           "without starting another pod.")
        self.add_running_button.clicked.connect(self.add_to_running)
        self.prepare_button.clicked.connect(self.prepare)
        self.arm_button.clicked.connect(lambda: self.activate(False))
        self.now_button.clicked.connect(lambda: self.activate(True))
        self.resume_button.clicked.connect(self.resume)
        self.pause_button.clicked.connect(self.pause)
        self.cancel_button.clicked.connect(self.cancel)
        for button in (self.prepare_button, self.arm_button, self.now_button, self.resume_button,
                       self.pause_button, self.cancel_button, self.add_running_button):
            actions.addWidget(button)
        layout.addLayout(actions)
        self.status = QLabel()
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        close = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        close.rejected.connect(self.reject)
        layout.addWidget(close)
        self.live_timer = QTimer(self)
        self.live_timer.setInterval(3000)
        self.live_timer.timeout.connect(self.refresh_live)
        self.live_timer.start()
        self.reload(initial_id)
        for index, selector in enumerate(self.gpu):
            selector.currentIndexChanged.connect(lambda _value, index=index: self._gpu_changed(index))
        self.refresh_catalog()

    def _catalog_source(self):
        identity = self.selected_id()
        if identity:
            batch = self.store.get(identity)
            if batch.get("members") and hasattr(self.store, "catalog"):
                return identity, lambda: self.store.catalog(identity)
        settings = self.catalog_settings
        if settings is not None:
            from .batches import _catalog
            return identity, lambda: _catalog(settings)
        return identity, None

    def refresh_catalog(self):
        """Fetch read-only catalog data away from the Qt UI thread."""
        if self._catalog_closed:
            return
        identity, operation = self._catalog_source()
        if self.catalog_worker is not None:
            return
        if operation is None:
            self.catalog_status.setText("Add a job or open with managed RunPod settings to check the catalog.")
            self.catalog.clear()
            return
        self._catalog_identity = identity
        self.catalog_refresh.setEnabled(False)
        self.catalog_status.setText("Checking live RunPod GPU prices and availability…")
        worker = CatalogWorker(operation, self)
        self.catalog_worker = worker
        worker.succeeded.connect(lambda rows, source=identity: self._catalog_succeeded(source, rows))
        worker.failed.connect(lambda message, source=identity: self._catalog_failed(source, message))
        worker.finished.connect(lambda worker=worker: self._catalog_finished(worker))
        worker.finished.connect(worker.deleteLater)
        worker.start()

    def _catalog_succeeded(self, identity, rows):
        if self._catalog_closed or identity != self.selected_id():
            return
        self.catalog.clear()
        batch = self.store.get(identity) if identity else None
        self.catalog_options[identity] = rows
        for row in sorted(rows, key=lambda entry: (entry["hourly_usd"], entry["gpu_id"])):
            rate = row["hourly_usd"]
            availability = row.get("availability") or "UNAVAILABLE"
            assessment = "Manual choice"
            item = QTreeWidgetItem([row["gpu_id"], f"${rate:.2f}", availability,
                                    f"{row.get('vram_gb', '?')} GB", assessment,
                                    str(row.get("available_location_count", "—"))])
            item.setToolTip(0, row["gpu_id"])
            item.setToolTip(5, "Available data centers: " + (", ".join(row.get("available_data_centers", [])) or "none reported"))
            self.catalog.addTopLevelItem(item)
        self.catalog_status.setText(catalog_summary(rows))
        for index, combo in enumerate(self.gpu):
            selected = combo.currentData()
            combo.blockSignals(True)
            combo.clear()
            for row in rows:
                combo.addItem(f"{row['gpu_id']} · ${row['hourly_usd']:.2f}/h · {row.get('vram_gb', '?')} GB VRAM", row['gpu_id'])
            chosen = combo.findData(selected)
            combo.setCurrentIndex(chosen if chosen >= 0 else -1)
            combo.blockSignals(False)
        ready = bool(batch) and len(rows) >= 2 and batch.get("status") in {"prepared", "armed", "waiting", "paused"}
        self.arm_button.setEnabled(ready and self.worker is None)
        self.now_button.setEnabled(ready and self.worker is None)
        self.show_estimates()

    def _catalog_failed(self, identity, message):
        if self._catalog_closed or identity != self.selected_id():
            return
        self.catalog.clear()
        self.catalog_status.setText(f"Catalog check failed: {message}")

    def _catalog_finished(self, worker):
        if self.catalog_worker is not worker:
            return
        self.catalog_worker = None
        if self._catalog_closed:
            if self.worker is None:
                super().done(self._close_result)
            return
        self.catalog_refresh.setEnabled(True)
        if self._catalog_identity != self.selected_id():
            self.refresh_catalog()

    def _update_choice_cap_ranges(self):
        for index, cap in enumerate(self.caps):
            cap.setMaximum(self.max_hourly.value())
            self._suggest_quoted_cap(index)

    def _suggest_quoted_cap(self, index):
        identity = self.selected_id()
        if not identity:
            return
        gpu_id = self.gpu[index].currentData()
        if gpu_id is None:
            return
        batch = self.store.get(identity)
        option = next((row for row in self.catalog_options.get(identity, batch.get("gpu_options") or []) if row["gpu_id"] == gpu_id), None)
        rate = option.get("hourly_usd") if option else None
        if isinstance(rate, (int, float)) and rate <= self.max_hourly.value() and rate > self.caps[index].value():
            self.caps[index].setValue(rate)

    def _gpu_changed(self, index):
        self._suggest_quoted_cap(index)
        self.show_estimates()

    def show_estimates(self, *_args):
        identity = self.selected_id()
        if not identity:
            return
        try:
            batch = self.store.get(identity)
        except (OSError, ValueError):
            return
        status = batch.get("status", "draft")
        wait = batch.get("waiting_reason") or ""
        next_check = batch.get("next_check_at")
        due = batch.get("due_at")
        detail = (f"{status} · {wait}" +
                  (f" · Local start: {display_time(due)}" if due else "") +
                  (f" · Next check: {display_time(next_check)}" if next_check else ""))
        detail += "\nChoose two GPUs manually from the catalog. Prepare workflows before scheduling or starting. No agent selects or assesses GPUs."
        self.detail.setText(detail)

    def refresh_live(self):
        """Show durable progress without changing GPU, rate, or date fields in use."""
        identity = self.selected_id()
        if not identity or self.worker is not None:
            return
        try:
            batch = self.store.get(identity)
        except (OSError, ValueError):
            return
        index = self.batch_picker.currentIndex()
        if index >= 0:
            self.batch_picker.setItemText(index, batch_label(batch))
        reason = batch.get("waiting_reason") or ""
        next_check = batch.get("next_check_at")
        due = batch.get("due_at")
        flag = "NEEDS REVIEW — " if batch.get("status") in {"failed", "preparation_failed", "detached"} else ""
        self.status.setText(flag + f"{batch.get('status', 'draft')} · {reason}" +
                            (f" · Local start: {display_time(due)}" if due else "") +
                            (f" · Next check: {display_time(next_check)}" if next_check else ""))
        by_id = {member["id"]: member for member in batch.get("members", [])}
        for row in range(self.jobs.topLevelItemCount()):
            item = self.jobs.topLevelItem(row)
            member = by_id.get(item.data(0, Qt.ItemDataRole.UserRole))
            if member:
                item.setText(2, member.get("status", ""))

    def selected_id(self):
        return self.batch_picker.currentData()

    def selected_member(self):
        item = self.jobs.currentItem()
        return item.data(0, Qt.ItemDataRole.UserRole) if item else None

    def reload(self, selected=None):
        selected = selected or self.selected_id()
        batches = self.store.list()
        self.batch_picker.blockSignals(True)
        self.batch_picker.clear()
        for batch in batches:
            self.batch_picker.addItem(batch_label(batch), batch["id"])
        index = self.batch_picker.findData(selected)
        self.batch_picker.setCurrentIndex(index if index >= 0 else (0 if batches else -1))
        self.batch_picker.blockSignals(False)
        self.refresh()
        self.changed()
        if self.catalog_worker is None:
            QTimer.singleShot(0, self.refresh_catalog)

    def refresh(self):
        identity = self.selected_id()
        batch = self.store.get(identity) if identity else None
        self.jobs.clear()
        if not batch:
            self.detail.setText("Create a batch or add a concept from Generate.")
            for button in (self.prepare_button, self.arm_button, self.now_button, self.resume_button,
                           self.pause_button, self.cancel_button, self.edit_button, self.remove_button):
                button.setEnabled(False)
            self._enforce_busy()
            return
        for member in batch.get("members", []):
            folder = Path(member["folder"])
            try:
                state = json.loads((folder / "state.json").read_text())
                qty = state.get("settings", {}).get("quantity", 1)
            except (OSError, ValueError):
                qty = "?"
            item = QTreeWidgetItem([member.get("concept", ""), str(qty), member.get("status", ""), str(folder)])
            item.setData(0, Qt.ItemDataRole.UserRole, member["id"])
            self.jobs.addTopLevelItem(item)
        status = batch.get("status", "draft")
        due = batch.get("due_at")
        options = self.catalog_options.get(identity, batch.get("gpu_options") or [])
        choices = batch.get("gpu_choices") or []
        profile = batch.get("settings", {}).get("runpod_config_file")
        ceiling = 3.0
        if profile and Path(profile).expanduser().is_file():
            try:
                ceiling = float(json.loads(Path(profile).expanduser().read_text())["gpu_max_hourly"])
            except (OSError, ValueError, KeyError, TypeError):
                pass
        self.max_hourly.setValue(float(batch.get("max_hourly") or ceiling))
        self._update_choice_cap_ranges()
        for index in range(2):
            combo = self.gpu[index]
            combo.clear()
            for option in options:
                rate = option.get("hourly_usd")
                price = f"${rate:.2f}/h" if isinstance(rate, (int, float)) else "price pending"
                combo.addItem(f"{option.get('gpu_id')} · {price} · {option.get('vram_gb', '?')} GB VRAM", option.get("gpu_id"))
            if len(choices) > index:
                chosen = combo.findData(choices[index].get("gpu_id"))
                combo.setCurrentIndex(chosen)
                self.caps[index].setValue(float(choices[index].get("max_hourly", 0.01)))
            else:
                combo.setCurrentIndex(-1)
        self.show_estimates()
        self.priority.setCurrentIndex(max(0, self.priority.findData(batch.get("priority", "high"))))
        self.retry_minutes.setValue(int(batch.get("retry_minutes") or 5))
        if due:
            local = datetime.fromisoformat(due).astimezone()
            self.due.setDateTime(QDateTime(QDate(local.year, local.month, local.day),
                                           QTime(local.hour, local.minute), QTimeZone.utc()))
        self.prepare_button.setEnabled(bool(batch.get("members")) and status in
                                       {"draft", "prepared", "preparation_failed", "preparing"})
        ready = bool(options) and len(options) >= 2 and status in {"prepared", "armed", "waiting", "paused"}
        self.arm_button.setEnabled(ready)
        self.now_button.setEnabled(ready)
        self.resume_button.setEnabled(status in {"failed", "detached"})
        self.pause_button.setEnabled(status in {"armed", "waiting"})
        from .batches import owns_gpu, reopen_refusal
        self.cancel_button.setEnabled(status not in {"complete", "cancelled", "running"} and not owns_gpu(batch))
        self.add_running_button.setEnabled(self.worker is None and (
            status in {"running", "armed", "waiting"} or reopen_refusal(batch) is None))
        self.max_total.setValue(float(batch.get("max_total_usd") or 0))
        if batch.get("managed_session", {}).get("pod_id") or status == "running":
            from .cost_safety import batch_projection, projection_text
            self.detail.setText((self.detail.text() + "\n" if self.detail.text() else "")
                                + "Spend: " + projection_text(batch_projection(batch)))
        self.edit_button.setEnabled(status in {"draft", "prepared", "preparation_failed", "failed", "paused"})
        self.remove_button.setEnabled(status in {"draft", "prepared", "preparation_failed", "failed", "paused"})
        self._enforce_busy()

    def _enforce_busy(self):
        busy = self.worker is not None
        for widget in (self.batch_picker, self.new_button, self.jobs, self.edit_button,
                       self.remove_button, self.prepare_button, self.arm_button,
                       self.now_button, self.resume_button, self.pause_button,
                       self.cancel_button, *self.gpu, *self.caps, self.max_hourly,
                       self.priority, self.retry_minutes, self.due, self.add_running_button, self.max_total):
            if busy:
                widget.setEnabled(False)
            elif widget in (self.batch_picker, self.new_button, self.jobs, self.max_total,
                            *self.gpu, *self.caps, self.max_hourly, self.priority, self.retry_minutes, self.due):
                widget.setEnabled(True)

    def new_batch(self):
        try:
            batch = self.store.create()
            self.reload(batch["id"])
        except Exception as error:  # noqa: BLE001 — present persistence error.
            QMessageBox.warning(self, "Cannot create batch", str(error))

    def edit_member(self):
        member_id = self.selected_member()
        if not member_id:
            return
        batch = self.store.get(self.selected_id())
        member = next(m for m in batch["members"] if m["id"] == member_id)
        try:
            state = json.loads((Path(member["folder"]) / "state.json").read_text())
            dialog = MemberEditDialog(state, self)
        except (OSError, ValueError, KeyError, TypeError) as error:
            QMessageBox.warning(self, "Cannot open batch job", str(error))
            return
        if dialog.exec() == QDialog.DialogCode.Accepted:
            try:
                concept, settings = dialog.values()
                self.store.edit(batch["id"], member_id, concept, settings)
                self.reload(batch["id"])
            except Exception as error:  # noqa: BLE001
                QMessageBox.warning(self, "Cannot edit batch job", str(error))

    def remove_member(self):
        member_id = self.selected_member()
        if member_id:
            try:
                self.store.remove(self.selected_id(), member_id)
                self.reload()
            except Exception as error:  # noqa: BLE001
                QMessageBox.warning(self, "Cannot remove batch job", str(error))

    def prepare(self):
        if self.worker is not None:
            return
        identity = self.selected_id()
        self.worker = BatchWorker(lambda emit, cancel, detach: self.store.prepare(identity, emit, cancel, detach), self)
        self.worker.event.connect(lambda event: self.status.setText(str(event.get("message", "Preparing…"))))
        self.worker.succeeded.connect(lambda _result: self.reload(identity))
        self.worker.failed.connect(self.preparation_failed)
        self.worker.finished.connect(self.operation_finished)
        self._enforce_busy()
        self.status.setText("Preparing saved scripts and workflows…")
        self.worker.start()

    def add_to_running(self, *_args, concept=None, quantity=None, confirm=None):
        """Prepare an addition off the UI thread (agent pass outside the batch lock), then request it."""
        from PySide6.QtWidgets import QInputDialog
        if self.worker is not None:
            return
        identity = self.selected_id()
        if concept is None:
            concept, ok = QInputDialog.getMultiLineText(self, "Add to running batch", "Concept")
            if not ok or not concept.strip():
                return
        if quantity is None:
            quantity, ok = QInputDialog.getInt(self, "Add to running batch", "Qty", 1, 1, 100)
            if not ok:
                return
        from .additions import prepare_addition
        self.pending_confirm = confirm
        self.worker = BatchWorker(lambda emit, cancel, detach: prepare_addition(
            self.store, identity, concept.strip(), quantity, emit, cancel, detach), self)
        self.worker.event.connect(lambda event: self.status.setText(str(event.get("message", "Preparing…"))))
        self.worker.succeeded.connect(self.addition_prepared)
        self.worker.failed.connect(lambda message: QMessageBox.warning(self, "Cannot add to this batch", message)
                                   if not self._closing else None)
        self.worker.finished.connect(self.operation_finished)
        self._enforce_busy()
        self.status.setText("Preparing the addition (the pod keeps rendering)…")
        self.worker.start()

    def addition_prepared(self, prepared):
        from .additions import submit_addition
        from .cost_safety import confirmation_text
        text = confirmation_text(prepared["projection"])
        confirm = self.pending_confirm or (lambda message: QMessageBox.question(
            self, "Add to running batch?", message + "\n\nQueue it on this batch's GPU?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No) == QMessageBox.StandardButton.Yes)
        if not confirm(text):
            self.status.setText("Addition prepared but not queued; its run folder is kept.")
            return
        try:
            submit_addition(self.store, prepared["batch_id"], Path(prepared["folder"]))
        except (OSError, ValueError, RuntimeError) as error:
            QMessageBox.warning(self, "Cannot add to this batch", str(error))
            return
        self.status.setText("Addition queued; the running batch accepts it at its next step.")
        self.changed()

    def preparation_failed(self, message):
        if not self._closing:
            QMessageBox.warning(self, "Batch preparation needs review", message)

    def operation_finished(self):
        worker, self.worker = self.worker, None
        retire_worker(self, worker)
        if self._closing:
            if self.catalog_worker is None:
                super().done(self._close_result)
        else:
            self.reload()
            self.refresh_catalog()

    def _approved_choices(self):
        result = []
        for combo, cap in zip(self.gpu, self.caps):
            if combo.currentIndex() < 0:
                raise ValueError("Refresh the catalog and choose two GPU types first.")
            result.append({"gpu_id": combo.currentData(), "max_hourly": cap.value()})
        if result[0]["gpu_id"] == result[1]["gpu_id"]:
            raise ValueError("Choose two different GPU types.")
        batch = self.store.get(self.selected_id())
        by_id = {option["gpu_id"]: option for option in self.catalog_options.get(self.selected_id(), batch.get("gpu_options", []))}
        for choice in result:
            rate = by_id[choice["gpu_id"]].get("hourly_usd")
            if isinstance(rate, (int, float)) and choice["max_hourly"] < rate:
                raise ValueError(f"The maximum rate for {choice['gpu_id']} is below its current quoted rate.")
            if choice["max_hourly"] > self.max_hourly.value():
                raise ValueError("A GPU choice exceeds this batch's hourly ceiling.")
        return result

    def activate(self, now: bool):
        try:
            due = None if now else selected_time(self.due).isoformat()
            identity = self.selected_id()
            cap = self.max_total.value() or None
            self.store.configure(identity, gpu_choices=self._approved_choices(),
                                 max_hourly=self.max_hourly.value(),
                                 catalog_options=self.catalog_options.get(identity),
                                 priority=self.priority.currentData(),
                                 retry_minutes=self.retry_minutes.value(), due_at=due, max_total_usd=cap)
            if now:
                self.store.run_now(identity)
            else:
                self.store.arm(identity)
            self.reload(identity)
            self.status.setText("Batch armed. GPU availability and approved rates are checked at dispatch."
                                + ("" if cap else " Warning: no spend cap is set for this batch."))
            if now:
                self.accept()
        except Exception as error:  # noqa: BLE001
            QMessageBox.warning(self, "Cannot arm batch", str(error))

    def resume(self):
        try:
            self.store.resume(self.selected_id())
            self.reload()
        except Exception as error:  # noqa: BLE001
            QMessageBox.warning(self, "Cannot resume batch", str(error))

    def pause(self):
        try:
            self.store.pause(self.selected_id())
            self.reload()
        except Exception as error:  # noqa: BLE001
            QMessageBox.warning(self, "Cannot pause batch", str(error))

    def cancel(self):
        answer = QMessageBox.question(self, "Cancel batch?",
            "Remove this batch from dispatch? Saved run folders and history remain."
            " Active compute requires the separate Shut down compute action.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No)
        if answer != QMessageBox.StandardButton.Yes:
            return
        try:
            self.store.cancel(self.selected_id())
            self.reload()
        except Exception as error:  # noqa: BLE001
            QMessageBox.warning(self, "Cannot cancel batch", str(error))

    def done(self, result):
        if self._closing:
            return
        self._closing = True
        self._close_result = result
        self._catalog_closed = True
        for button in (self.prepare_button, self.arm_button, self.now_button,
                       self.resume_button, self.pause_button, self.cancel_button):
            button.setEnabled(False)
        if self.worker is not None:
            self.worker.detach_event.set()
            self.status.setText("Detaching from preparation… saved work remains available.")
            return
        if self.catalog_worker is not None:
            self.status.setText("Finishing catalog check…")
            return
        super().done(result)

    def accept(self):
        self.done(QDialog.DialogCode.Accepted)

    def reject(self):
        self.done(QDialog.DialogCode.Rejected)

    def closeEvent(self, event):
        if self.worker is not None or self.catalog_worker is not None:
            self.reject()
            event.ignore()
        else:
            self.reject()
            event.accept()
