"""Small local date/time controls and a list of saved scheduled requests."""

from datetime import datetime, timedelta
from pathlib import Path

from PySide6.QtCore import QDate, QDateTime, Qt, QTime, QTimer, QTimeZone
from PySide6.QtWidgets import (
    QDateTimeEdit,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
)

from .scheduling import VISIBLE, display_time, resolve_local, system_zone


def date_field():
    field = QDateTimeEdit()
    # Hold wall-clock components without Qt silently normalizing a DST gap/fold.
    field.setTimeZone(QTimeZone.utc())
    value = datetime.now(system_zone()) + timedelta(hours=1)
    field.setDateTime(QDateTime(QDate(value.year, value.month, value.day),
                                QTime(value.hour, value.minute), QTimeZone.utc()))
    field.setDisplayFormat("MM/dd/yy HH:mm")
    field.setCalendarPopup(True)
    field.setToolTip("Local system time, 24-hour format. Keep simpleGen open and the computer awake.")
    return field


def selected_time(field):
    date, time = field.date(), field.time()
    value = datetime(date.year(), date.month(), date.day(), time.hour(), time.minute())
    return resolve_local(value)


class ScheduleDialog(QDialog):
    def __init__(self, scheduler, changed, parent=None, edit=None, repair=None):
        super().__init__(parent)
        self.scheduler, self.changed, self.edit_callback = scheduler, changed, edit
        self.repair_callback = repair
        self.setWindowTitle("Scheduled local generations")
        self.resize(860, 390)
        layout = QVBoxLayout(self)
        label = QLabel("Local system time · Keep simpleGen open and this computer awake.\n"
                       "Pending requests retry every 30 minutes when blocked. Other work is never interrupted.")
        label.setWordWrap(True)
        layout.addWidget(label)
        self.rows = QTreeWidget()
        self.rows.setRootIsDecorated(False)
        self.rows.setHeaderLabels(["Local start", "Concept", "Status"])
        self.rows.setColumnWidth(0, 145)
        self.rows.setColumnWidth(1, 240)
        self.rows.itemSelectionChanged.connect(self.selection_changed)
        layout.addWidget(self.rows)
        self.help_label = QLabel("Select a request to see available actions.")
        self.help_label.setWordWrap(True)
        layout.addWidget(self.help_label)
        actions = QHBoxLayout()
        self.now = QPushButton("Run now")
        self.reschedule = QPushButton("Reschedule…")
        self.edit_button = QPushButton("Edit scheduled job…")
        self.edit_button.clicked.connect(self.edit_selected)
        self.repair_button = QPushButton("Repair…")
        self.repair_button.setToolTip("Replace unavailable models; the accepted script and prompts are kept.")
        self.repair_button.clicked.connect(self.repair_selected)
        self.remove = QPushButton("Remove")
        self.remove.setToolTip("Remove the schedule only. Saved files and any ComfyUI work are retained.")
        for button, action in ((self.now, "now"), (self.reschedule, "reschedule"), (self.remove, "remove")):
            button.clicked.connect(lambda _checked=False, value=action: self.change(value))
            actions.addWidget(button)
        actions.addWidget(self.edit_button)
        actions.addWidget(self.repair_button)
        actions.addStretch()
        close = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        close.rejected.connect(self.reject)
        actions.addWidget(close)
        layout.addLayout(actions)
        self.timer = QTimer(self)
        self.timer.setInterval(1000)
        self.timer.timeout.connect(self.refresh)
        self.timer.start()
        self.refresh()

    def selected(self):
        item = self.rows.currentItem()
        return item.data(0, Qt.ItemDataRole.UserRole) if item else None

    def refresh(self):
        previous = self.selected()
        try:
            records = self.scheduler.records()
        except (OSError, ValueError) as error:
            self.setWindowTitle(f"Cannot read schedules: {error}")
            return
        self.rows.blockSignals(True)
        self.rows.clear()
        labels = {"scheduled": "Scheduled", "waiting": "Waiting for availability", "overdue": "Overdue",
                  "running": "Running", "attention": "Needs attention — Resume", "paused": "Paused",
                  "repaired": "Repaired"}
        for record in records:
            if record["status"] not in VISIBLE:
                continue
            status = labels[record["status"]]
            if record["status"] == "waiting":
                status = record.get("note") or "Waiting — checking availability"
                if record.get("next_retry_at"):
                    status += f" · next retry {display_time(record['next_retry_at'])}"
            reason = ""
            if record["status"] == "repaired":
                target = record.get("repaired_to", {})
                status = f"Repaired → {Path(target.get('folder', '')).name or target.get('run_id', '')}"
            if record["status"] == "attention" and record.get("preflight", {}).get("status") == "fail":
                reason = record.get("note") or f"Preflight failed: {record['preflight'].get('reason', '')}"
                status = f"{labels['attention']} · {reason[:160]}{'…' if len(reason) > 160 else ''}"
            item = QTreeWidgetItem([display_time(record["due_at"]), record["concept"].splitlines()[0][:100],
                                    status])
            item.setData(0, Qt.ItemDataRole.UserRole, record)
            item.setToolTip(1, record["concept"])
            item.setToolTip(2, f"{reason}\n{record['folder']}" if reason else record["folder"])
            self.rows.addTopLevelItem(item)
            if ((previous and previous["id"] == record["id"])
                    or (previous is None and record["status"] == "attention"
                        and self.rows.currentItem() is None)):
                self.rows.setCurrentItem(item)
        if previous is None and self.rows.currentItem() is None and self.rows.topLevelItemCount():
            self.rows.setCurrentItem(self.rows.topLevelItem(0))
        self.rows.blockSignals(False)
        self.selection_changed()

    def selection_changed(self):
        record = self.selected()
        enabled = bool(record and record["status"] != "running" and self.scheduler.owner)
        self.now.setEnabled(enabled)
        self.now.setText("Resume" if record and record["started"] else "Run now")
        self.reschedule.setEnabled(enabled and not record["started"])
        self.edit_button.setEnabled(bool(enabled and not record["started"]
                                         and record["status"] in {"scheduled", "waiting", "overdue"}
                                         and self.edit_callback)
                                    or bool(enabled and record["status"] == "attention" and self.edit_callback))
        self.edit_button.setText("Edit as new run…" if record and record["status"] == "attention"
                                 else "Edit scheduled job…")
        self.help_label.setText(
            "Needs attention: Resume unchanged or edit the concept as a new run; saved work stays intact."
            if record and record["status"] == "attention" else
            "Pending: Edit scheduled job changes its saved request before any work starts."
            if record and record["status"] in {"scheduled", "waiting", "overdue"} else
            "Select a request to see available actions."
        )
        self.remove.setEnabled(enabled)
        repaired = bool(record and record["status"] == "repaired")
        self.now.setEnabled(enabled and not repaired)
        self.reschedule.setEnabled(self.reschedule.isEnabled() and not repaired)
        self.edit_button.setEnabled(self.edit_button.isEnabled() and not repaired)
        self.repair_button.setEnabled(bool(enabled and record["status"] == "attention" and self.repair_callback))
        if record and record["status"] == "attention":
            self.help_label.setText("Needs attention: Resume unchanged, Repair unavailable models, or edit the "
                                    "concept as a new run; saved work stays intact.")
        elif repaired:
            self.help_label.setText("Repaired: this request is replaced by its repaired revision and will not run.")

    def repair_selected(self):
        record = self.selected()
        if record and record["status"] == "attention" and self.repair_callback:
            self.repair_callback(record)
            self.refresh()
            self.changed()

    def edit_selected(self):
        record = self.selected()
        if record and self.edit_callback:
            self.edit_callback(record)
            self.refresh()
            self.changed()

    def change(self, action):
        record = self.selected()
        if not record:
            return
        try:
            due = None
            if action == "reschedule":
                dialog = QDialog(self)
                dialog.setWindowTitle("Reschedule — local time")
                layout = QVBoxLayout(dialog)
                field = date_field()
                layout.addWidget(field)
                buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
                buttons.accepted.connect(dialog.accept)
                buttons.rejected.connect(dialog.reject)
                layout.addWidget(buttons)
                if dialog.exec() != QDialog.DialogCode.Accepted:
                    return
                due = selected_time(field)
            self.scheduler.change(record["id"], action, due)
            self.refresh()
            self.changed()
        except (OSError, ValueError, RuntimeError) as error:
            QMessageBox.warning(self, "Cannot change schedule", str(error))
