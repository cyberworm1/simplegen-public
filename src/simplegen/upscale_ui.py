"""Qt pieces for the 4K60 upscale (Phase 8). Network work runs in OperationWorker threads."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from . import upscale
from .dimensions import ASPECT_RATIOS
from .qtutil import retire_worker
from .settings import Settings


def framing_field(value: str, *, default_label: str | None = None) -> QComboBox:
    field = QComboBox()
    if default_label is not None:
        field.addItem(default_label, "")
    for key, label in upscale.FRAMINGS:
        field.addItem(label, key)
    field.setCurrentIndex(max(0, field.findData(value)))
    return field


def method_field(value: str) -> QComboBox:
    """Upscale method: the preferred methods with speed notes, and any other saved preset.

    An empty preset (settings saved before the method choice) shows as the default method, Remacri (fast).
    """
    value = value or upscale.DEFAULT_METHOD
    field = QComboBox()
    for key, _label, _speed in upscale.METHODS:
        field.addItem(upscale.method_label(key), key)
    if value and field.findData(value) < 0:
        field.addItem(value, value)
    field.setCurrentIndex(max(0, field.findData(value)))
    return field


class UpscaleSettings(QWidget):
    """Settings rows: service URL, token, preset, framing defaults, test and host clean-up."""

    def __init__(self, settings: Settings, form: QFormLayout, parent=None):
        super().__init__(parent)
        from .ui import OperationWorker, selectable
        self._worker_class = OperationWorker
        self.settings = settings
        self.worker = None
        self.health: dict | None = None
        self.url = QLineEdit(settings.upscale_url)
        self.url.setPlaceholderText("e.g. http://upscale-host:8190 (empty turns 4K60 off)")
        self.token = QLineEdit()
        self.token.setEchoMode(QLineEdit.EchoMode.Password)
        try:
            saved = bool(upscale.load_token())
        except upscale.UpscaleError:
            saved = False
        self.token.setPlaceholderText("Saved; type to replace" if saved else "Token shown by the upscale host")
        self.preset = method_field(settings.upscale_preset)
        self.preset.setEditable(True)
        self.framing = framing_field(settings.upscale_framing)
        overrides = QWidget()
        grid = QGridLayout(overrides)
        grid.setContentsMargins(0, 0, 0, 0)
        self.aspect_framing: dict[str, QComboBox] = {}
        for index, (ratio, _label) in enumerate(ASPECT_RATIOS):
            field = framing_field(settings.upscale_framing_by_aspect.get(ratio, ""), default_label="Default")
            self.aspect_framing[ratio] = field
            grid.addWidget(QLabel(ratio), index // 2, (index % 2) * 2)
            grid.addWidget(field, index // 2, (index % 2) * 2 + 1)
        actions = QHBoxLayout()
        self.test_button = QPushButton("Test connection")
        self.test_button.clicked.connect(self.test)
        self.clear_button = QPushButton("Clear host cache…")
        self.clear_button.setToolTip("Delete finished jobs on the upscale host that no saved upscale still needs.")
        self.clear_button.setEnabled(False)
        self.clear_button.clicked.connect(self.clear_host)
        actions.addWidget(self.test_button)
        actions.addWidget(self.clear_button)
        actions.addStretch()
        self.status = selectable("4K60 upscales run on a separate host; the render never waits for it.")
        self.status.setObjectName("muted")
        column = QVBoxLayout(self)
        column.setContentsMargins(0, 0, 0, 0)
        column.addLayout(actions)
        column.addWidget(self.status)
        form.addRow("Upscale service", self.url)
        form.addRow("Upscale token", self.token)
        form.addRow("Upscale preset", self.preset)
        form.addRow("4K framing", self.framing)
        form.addRow("4K framing by aspect", overrides)
        form.addRow("", self)

    def values(self) -> dict:
        index = self.preset.currentIndex()
        listed = index >= 0 and self.preset.currentText() == self.preset.itemText(index)
        preset = (self.preset.currentData() or "") if listed else self.preset.currentText().strip()
        return {
            "upscale_url": self.url.text().strip(),
            "upscale_preset": preset,
            "upscale_framing": self.framing.currentData(),
            "upscale_framing_by_aspect": {ratio: field.currentData()
                                          for ratio, field in self.aspect_framing.items() if field.currentData()},
        }

    def save_token(self) -> None:
        if self.token.text().strip():
            upscale.save_token(self.token.text())

    def busy(self) -> bool:
        return self.worker is not None

    def _start(self, operation, succeeded) -> None:
        if self.worker is not None:
            return
        self.test_button.setEnabled(False)
        self.clear_button.setEnabled(False)
        worker = self._worker_class(operation, self)
        self.worker = worker
        worker.succeeded.connect(succeeded)
        worker.failed.connect(lambda message: self.status.setText(message))
        worker.finished.connect(self._finished)
        worker.start()

    def _finished(self) -> None:
        worker, self.worker = self.worker, None
        retire_worker(self, worker)
        self.test_button.setEnabled(True)
        self.clear_button.setEnabled(bool(self.health and upscale.clearable_host_jobs(self.health)))

    def test(self) -> None:
        self.save_token()
        settings = replace(self.settings, upscale_url=self.url.text().strip())

        def check():
            client = upscale.Client.from_settings(settings, timeout=5.0)
            try:
                return client.health()
            finally:
                client.close()
        self.status.setText("Contacting the upscale host…")
        self._start(check, self.tested)

    def tested(self, health: dict) -> None:
        self.health = health
        self.status.setText(upscale.host_summary(health))
        current = self.preset.currentText()
        self.preset.clear()
        for entry in health.get("presets", []):
            self.preset.addItem(upscale.method_label(entry["id"], entry.get("label") or ""), entry["id"])
        index = self.preset.findData(current)
        if index < 0:
            index = self.preset.findText(current)
        self.preset.setCurrentIndex(index if index >= 0 else max(0, self.preset.findData(upscale.DEFAULT_METHOD)))

    def clear_host(self) -> None:
        jobs = upscale.clearable_host_jobs(self.health or {})
        if not jobs:
            return
        answer = QMessageBox.question(
            self, "Clear host cache",
            f"Delete {len(jobs)} job(s) from the upscale host? Failed or cancelled upscales you have not "
            "dismissed, and anything still running, are kept.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel, QMessageBox.StandardButton.Cancel)
        if answer != QMessageBox.StandardButton.Yes:
            return
        settings = replace(self.settings, upscale_url=self.url.text().strip())

        def clear():
            cleared = upscale.clear_host(settings, jobs)
            client = upscale.Client.from_settings(settings, timeout=5.0)
            try:
                return {"cleared": cleared, "health": client.health()}
            finally:
                client.close()
        self._start(clear, lambda result: (self.tested(result["health"]), self.status.setText(
            f"Cleared {result['cleared']} job(s).\n" + upscale.host_summary(result["health"]))))


class UpscaleDialog(QDialog):
    """Explicit Upscale… for any completed final cut, with framing, preset and a time estimate."""

    def __init__(self, final: Path, settings: Settings, metadata: dict, parent=None):
        super().__init__(parent)
        from .ui import OperationWorker, selectable
        self.setWindowTitle("Upscale to 4K60")
        self.final = final
        self.settings = settings
        self.metadata = metadata
        self.health: dict | None = None
        layout = QVBoxLayout(self)
        layout.addWidget(selectable(f"Source: {final}"))
        form = QFormLayout()
        aspect = settings.aspect_ratio
        self.framing = framing_field(upscale.framing_for(settings) if aspect else settings.upscale_framing)
        self.preset = method_field(settings.upscale_preset)
        form.addRow("4K framing", self.framing)
        form.addRow("Preset", self.preset)
        layout.addLayout(form)
        self.output = selectable("")
        self.estimate = selectable("Checking the upscale host…")
        layout.addWidget(self.output)
        layout.addWidget(self.estimate)
        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Upscale")
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)
        self.framing.currentIndexChanged.connect(self.update_text)
        self.preset.currentIndexChanged.connect(self.update_text)
        self.update_text()

        def check():
            client = upscale.Client.from_settings(settings, timeout=5.0)
            try:
                return client.health()
            finally:
                client.close()
        self.worker = OperationWorker(check, self)
        self.worker.succeeded.connect(self.received)
        self.worker.failed.connect(lambda message: self.estimate.setText(
            f"{message} The request is saved and waits until the host is reachable."))
        self.worker.start()

    def received(self, health: dict) -> None:
        self.health = health
        current = self.preset.currentData()
        self.preset.blockSignals(True)
        self.preset.clear()
        for entry in health.get("presets", []):
            if entry.get("ready"):
                self.preset.addItem(upscale.method_label(entry["id"], entry.get("label") or ""), entry["id"])
        index = self.preset.findData(current)
        self.preset.setCurrentIndex(index if index >= 0 else max(0, self.preset.findData(upscale.DEFAULT_METHOD)))
        self.preset.blockSignals(False)
        self.update_text()

    def update_text(self) -> None:
        width, height = self.metadata.get("width"), self.metadata.get("height")
        if width and height:
            out = upscale.target_size(int(width), int(height), self.framing.currentData())
            self.output.setText(f"{width} × {height} → {out[0]} × {out[1]} at 60 fps (HEVC 10-bit)")
        if self.health is None:
            return
        duration = float(self.metadata.get("duration") or 0)
        seconds = upscale.estimate_seconds(self.health, self.preset.currentData() or "", duration)
        preset = self.preset.currentData() or self.health.get("default_preset")
        if not preset:
            self.estimate.setText("The host has no default preset yet. Choose a preset, or the request waits.")
        elif seconds is None:
            self.estimate.setText("No time estimate yet for this preset. Large videos can take hours.")
        else:
            self.estimate.setText(f"Estimated {upscale._duration(seconds)} on the host, after any queued work. "
                                  "You can keep working; the result is added to this run's folder.")

    def choice(self) -> dict:
        return {"framing": self.framing.currentData(), "preset": self.preset.currentData() or ""}

    def done(self, result: int) -> None:
        self.worker.wait()
        super().done(result)


class UpscalePanel(QWidget):
    """Watch tab row: version picker, status, and Upscale…/Cancel/Retry/Dismiss for the shown final cut."""

    play = Signal(str)
    action = Signal(str, str)  # (action, job id) for cancel/retry/dismiss; ("upscale", "") for a new request.

    def __init__(self, parent=None):
        super().__init__(parent)
        from .ui import selectable
        self.final: Path | None = None
        self.record: dict | None = None
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        self.version = QComboBox()
        self.version.addItem("Original", "original")
        self.version.addItem("4K60", "4k60")
        self.version.setVisible(False)
        self.version.currentIndexChanged.connect(self.version_changed)
        self.status = selectable("")
        self.status.setObjectName("muted")
        self.upscale_button = QPushButton("Upscale…")
        self.upscale_button.setToolTip("Make a 4K/60 version of this final cut on the upscale host.")
        self.upscale_button.clicked.connect(lambda: self.action.emit("upscale", ""))
        self.cancel_button = QPushButton("Cancel upscale")
        self.retry_button = QPushButton("Retry upscale")
        self.dismiss_button = QPushButton("Dismiss")
        for button, name in ((self.cancel_button, "cancel"), (self.retry_button, "retry"),
                             (self.dismiss_button, "dismiss")):
            button.clicked.connect(lambda _checked=False, value=name: self.record and self.action.emit(
                value, self.record["id"]))
        row.addWidget(self.version)
        row.addWidget(self.status, 1)
        for button in (self.upscale_button, self.cancel_button, self.retry_button, self.dismiss_button):
            row.addWidget(button)
        self.refresh()

    def show_final(self, final: Path | None) -> None:
        self.final = final.resolve() if final else None
        self.version.blockSignals(True)
        self.version.setCurrentIndex(0)
        self.version.blockSignals(False)
        self.refresh()
        if self.upscaled():
            self.version.setCurrentIndex(1)  # Prefer the 4K60 version when it exists.

    def upscaled(self) -> Path | None:
        return upscale.upscaled_version(self.final) if self.final else None

    def refresh(self, records: list[dict] | None = None) -> None:
        if self.final is None:
            self.record = None
        else:
            matching = ([r for r in records if Path(r["run_folder"], r["source"]).resolve() == self.final]
                        if records is not None else upscale.records_for(self.final))
            matching = [r for r in matching if r["status"] != "dismissed"]
            self.record = matching[-1] if matching else None
        status = (self.record or {}).get("status")
        ready = self.upscaled() is not None
        self.version.setVisible(ready)
        self.upscale_button.setVisible(self.final is not None and status not in upscale.ACTIVE + upscale.FLAGGED)
        self.cancel_button.setVisible(status in ("pending", "remote", "downloading"))
        self.retry_button.setVisible(status in upscale.FLAGGED)
        self.dismiss_button.setVisible(status in upscale.FLAGGED)
        self.status.setText(upscale.describe(self.record) if self.record and status != "complete" else "")

    def version_changed(self, _index: int) -> None:
        if self.final is None:
            return
        path = self.upscaled() if self.version.currentData() == "4k60" else self.final
        self.play.emit(str(path or self.final))
