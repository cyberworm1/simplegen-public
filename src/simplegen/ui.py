"""The small desktop front end. All long-running work belongs to the worker."""

from __future__ import annotations

import sys
import threading
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

from PySide6.QtCore import QDate, QDateTime, Qt, QThread, QTime, QTimer, QTimeZone, QUrl, Signal
from PySide6.QtGui import QCloseEvent, QDesktopServices
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtMultimediaWidgets import QVideoWidget
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .dimensions import ASPECT_RATIOS
from .h3scene_compat import normalize_state, read_state
from .preflight import FAIL, UNVERIFIED, run_preflight, run_saved_preflight, summary
from .qtutil import retire_worker
from .schedule_ui import ScheduleDialog, date_field, selected_time
from .scheduling import (
    VISIBLE,
    Scheduler,
    display_time,
    fingerprint,
    local_queue_empty,
    system_zone,
)
from .settings import Settings, with_generation_location, with_runpod_config
from .upscale_ui import UpscaleDialog, UpscalePanel, UpscaleSettings, method_field

PROVIDERS = (("Codex", "codex"), ("Claude Code", "claude"), ("Grok Build", "grok"))


def change_agent(settings: Settings, provider: str) -> Settings:
    """Keep the requested tier where available; never reuse another provider's model ID."""
    from .agents import reasoning_options, suggest_model

    tier = settings.model_tier if settings.model_tier != "Custom" else "Economy"
    model = suggest_model(provider, tier)
    options = reasoning_options(provider, model)
    reasoning = settings.reasoning
    if reasoning not in options:
        reasoning = "low" if "low" in options else options[0] if options else "auto"
    return replace(
        settings,
        provider=provider,
        model_tier=tier if model else "Custom",
        model=model,
        reasoning=reasoning,
    )


STYLE = """
QWidget { background: #171b23; color: #e6eaf2; font-size: 13px; }
QMainWindow, QDialog { background: #171b23; }
QLabel#title { font-size: 26px; font-weight: 700; color: #ffffff; }
QLabel#muted { color: #a5afbf; }
QLabel#cloud { background: #493517; color: #ffd383; border: 1px solid #a47629;
              border-radius: 6px; padding: 10px; font-weight: 600; }
QLineEdit, QPlainTextEdit, QComboBox, QDoubleSpinBox {
    background: #222936; border: 1px solid #3c4657; border-radius: 5px; padding: 7px;
    selection-background-color: #385e92;
}
QPushButton { background: #2b3545; border: 1px solid #435169;
              border-radius: 5px; padding: 8px 14px; }
QPushButton:hover { background: #3a4960; }
QPushButton:disabled { color: #788391; background: #242a34; }
QPushButton#primary { background: #4b79be; border-color: #6492d5; color: white;
                      font-weight: 700; }
QPushButton#primary:disabled { background: #344459; color: #93a0b1; }
QRadioButton { spacing: 7px; }
QRadioButton::indicator { width: 14px; height: 14px; border-radius: 8px;
                          border: 2px solid #8996aa; background: #222936; }
QRadioButton::indicator:checked { background: #89b6f5; border-color: #89b6f5; }
QRadioButton:disabled { color: #788391; }
QRadioButton::indicator:disabled { border-color: #566174; }
QTabWidget::pane { border: 1px solid #354054; border-radius: 5px; }
QTabBar::tab { background: #222936; padding: 10px 28px; }
QTabBar::tab:selected { background: #33435a; color: white; }
QProgressBar { border: 1px solid #3c4657; border-radius: 5px; min-height: 22px;
               text-align: center; }
QProgressBar::chunk { background: #4b79be; }
QSlider::groove:horizontal { height: 5px; background: #3c4657; }
QSlider::handle:horizontal { background: #89b6f5; width: 12px; margin: -4px 0;
                            border-radius: 6px; }
"""


def selectable(text: str = "") -> QLabel:
    label = QLabel(text)
    label.setWordWrap(True)
    label.setTextFormat(Qt.TextFormat.PlainText)
    label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
    return label


def number_field(value: float, minimum: float, maximum: float, suffix: str) -> QDoubleSpinBox:
    field = QDoubleSpinBox()
    field.setRange(minimum, maximum)
    field.setDecimals(2)
    field.setSingleStep(0.05 if maximum == 16 else 1)
    field.setSuffix(suffix)
    field.setValue(value)
    return field


def steps_field(value: int | None) -> QSpinBox:
    field = QSpinBox()
    field.setRange(1, 100)
    field.setValue(value if value is not None else 25)
    field.setToolTip("Sampling steps for every generation segment. Higher values take longer.")
    return field


def aspect_field(value: str) -> QComboBox:
    field = QComboBox()
    for ratio, label in ASPECT_RATIOS:
        field.addItem(f"{ratio} ({label})", ratio)
    field.setCurrentIndex(field.findData(value or "16:9"))
    field.setPlaceholderText("Saved template ratio")
    field.setToolTip("Target aspect ratio. Dimensions are rounded to H3's 32-pixel grid.")
    field.setMinimumWidth(field.sizeHint().width())
    return field


class OperationWorker(QThread):
    succeeded = Signal(dict)
    failed = Signal(str)

    def __init__(self, operation, parent=None):
        super().__init__(parent)
        self.operation = operation

    def run(self) -> None:
        try:
            result = self.operation()
        except Exception as exc:  # noqa: BLE001 — UI thread boundary.
            self.failed.emit(str(exc))
        else:
            self.succeeded.emit(result)


class GPUChoiceDialog(QDialog):
    """No rental is authorized until a user explicitly selects and accepts a GPU."""

    def __init__(self, options: list[dict], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Choose RunPod compute")
        self.resize(620, 420)
        self.options = options
        self.group = QButtonGroup(self)
        layout = QVBoxLayout(self)
        quantity = options[0].get("quantity", 1) if options else 1
        layout.addWidget(selectable(f"Choose compute for {quantity} video(s), generated one at a time. Select a GPU manually. RunPod usage costs money."))
        content = QWidget()
        rows = QVBoxLayout(content)
        for index, option in enumerate(options):
            button = QRadioButton(f"{option.get('gpu_count', 1)} × {option.get('gpu_id', 'GPU')}")
            self.group.addButton(button, index)
            rows.addWidget(button)
            def value(key, suffix="", prefix="", item=option):
                raw = item.get(key)
                return f"{prefix}{raw:g}{suffix}" if isinstance(raw, (float, int)) else "unknown"
            rows.addWidget(selectable(
                f"VRAM {value('vram_gb', ' GB')} · GPU {value('hourly_usd', '/hour', '$')}"
                f" · CPU {value('cpu_hourly_usd', '/hour', '$')}\n"
                f"Availability: {option.get('availability', 'unknown')}"
            ))
        rows.addStretch()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(content)
        layout.addWidget(scroll)
        layout.addWidget(selectable("Estimates are not a spending cap. Missing estimates are shown as unknown."))
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.rent_button = buttons.button(QDialogButtonBox.StandardButton.Ok)
        self.rent_button.setText("Rent selected compute")
        self.rent_button.setEnabled(False)
        self.group.idClicked.connect(lambda _id: self.rent_button.setEnabled(True))
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def choice(self) -> dict | None:
        index = self.group.checkedId()
        return dict(self.options[index]) if index >= 0 else None


class SettingsDialog(QDialog):
    def __init__(self, settings: Settings, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("simpleGen · Settings")
        self.setMinimumWidth(530)
        self.settings = replace(settings)
        self.operation_worker = None
        self.pending_close = False
        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.provider = QComboBox()
        for label, value in PROVIDERS:
            self.provider.addItem(label, value)
        self.provider.setCurrentIndex(max(0, self.provider.findData(settings.provider)))
        self.tier = QComboBox()
        self.tier.addItems(["Economy", "Balanced", "Strongest", "Custom"])
        self.tier.setCurrentText(settings.model_tier)
        self.model = QComboBox()
        self.model.setEditable(True)
        self.reasoning = QComboBox()
        self.reasoning.setEditable(True)
        self.steps = steps_field(settings.steps)
        self.megapixels = number_field(settings.megapixels, 0.05, 16, " MP")
        self.duration = number_field(settings.duration, 1, 3600, " seconds")
        self.aspect_ratio = aspect_field(settings.aspect_ratio)
        self.url = QLineEdit(settings.comfy_url)
        self.token_env = QLineEdit(settings.token_env)
        self.token_env.setPlaceholderText("Environment variable name, never the token itself")
        self.browser_login_file = QLineEdit(settings.browser_login_file)
        self.browser_login_file.setPlaceholderText("Optional path to browser-login.json")
        self.cloud = QCheckBox("Use cloud ComfyUI (paid GPU)")
        self.cloud.setChecked(settings.cloud)
        self.cloud_name = QLineEdit(settings.cloud_name)
        self.managed_runpod = QCheckBox("Manage Global Volume GPU from this desktop")
        self.managed_runpod.setChecked(settings.managed_runpod)
        self.runpod_config_file = QLineEdit(settings.runpod_config_file)
        self.runpod_credentials_file = QLineEdit(settings.runpod_credentials_file)
        self.runpod_schema_file = QLineEdit(settings.runpod_schema_file)
        self.runpod_default_workflow = QLineEdit(settings.runpod_default_workflow)
        self.runpod_default_model = QLineEdit(settings.runpod_default_model)
        self.runpod_default_clip = QLineEdit(settings.runpod_default_clip)
        self.runpod_default_workflow.setToolTip("Absolute JSON path, or a filename relative to your workflow folder. Used when copying a local OOM run to RunPod.")
        for label, widget in (
            ("Agent", self.provider),
            ("Model tier", self.tier),
            ("Model", self.model),
            ("Reasoning", self.reasoning),
            ("Steps", self.steps),
            ("Megapixel", self.megapixels),
            ("Duration", self.duration),
            ("Aspect ratio", self.aspect_ratio),
            ("ComfyUI URL", self.url),
            ("Token environment variable", self.token_env),
            ("Browser login file", self.browser_login_file),
            ("Generation location", self.cloud),
            ("Cloud provider name", self.cloud_name),
            ("RunPod", self.managed_runpod),
        ):
            form.addRow(label, widget)
        for label, field in (("Global GPU profile", self.runpod_config_file),
                             ("Credentials file", self.runpod_credentials_file),
                             ("ComfyUI schema cache", self.runpod_schema_file)):
            row = QWidget()
            path_layout = QHBoxLayout(row)
            path_layout.setContentsMargins(0, 0, 0, 0)
            path_layout.addWidget(field)
            browse = QPushButton("Browse…")
            browse.clicked.connect(lambda _checked=False, edit=field: self.browse_file(edit))
            path_layout.addWidget(browse)
            form.addRow(label, row)
        form.addRow("RunPod default workflow", self.runpod_default_workflow)
        form.addRow("RunPod default diffusion model", self.runpod_default_model)
        form.addRow("RunPod default CLIP", self.runpod_default_clip)
        self.failure_shutdown = QSpinBox()
        self.failure_shutdown.setRange(0, 240)
        self.failure_shutdown.setSuffix(" min")
        self.failure_shutdown.setSpecialValueText("Off")
        self.failure_shutdown.setValue(settings.failure_shutdown_minutes)
        self.failure_shutdown.setToolTip("After a failure, shut the GPU down once every finished output is downloaded "
                                         "and the queues have been empty this long. 0 turns it off.")
        self.startup_timeout = QSpinBox()
        self.startup_timeout.setRange(5, 1440)
        self.startup_timeout.setSuffix(" min")
        self.startup_timeout.setValue(max(5, settings.startup_timeout_minutes))
        self.alert_webhook = QLineEdit(settings.alert_webhook_url)
        self.alert_webhook.setPlaceholderText("Optional, e.g. https://ntfy.sh/your-topic (plain-text POST)")
        form.addRow("Failure shutdown", self.failure_shutdown)
        form.addRow("GPU startup timeout", self.startup_timeout)
        self.keep_warm = QSpinBox()
        self.keep_warm.setRange(5, 60)
        self.keep_warm.setSuffix(" min")
        self.keep_warm.setValue(min(60, max(5, settings.keep_warm_minutes)))
        self.keep_warm.setToolTip("After verified downloads, keep the GPU this long with empty queues so additions or "
                                  "another batch can use it. Applies to new sessions.")
        form.addRow("Keep-warm window", self.keep_warm)
        form.addRow("Alert webhook", self.alert_webhook)
        self.adopt_schema_button = QPushButton("Use latest live schema…")
        self.adopt_schema_button.setToolTip("Install the last live RunPod node schema as the cache; the old file is kept.")
        self.adopt_schema_button.clicked.connect(self.adopt_live_schema)
        form.addRow("Schema cache", self.adopt_schema_button)
        cache_row = QHBoxLayout()
        self.reference_cache_label = QLabel()
        self.clear_cache_button = QPushButton("Clear unused")
        self.clear_cache_button.setToolTip("Delete cached reference copies that no saved run needs. Copies kept for "
                                           "failed, cancelled or not-yet-run work stay.")
        self.clear_cache_button.clicked.connect(self.clear_reference_cache)
        cache_row.addWidget(self.reference_cache_label, 1)
        cache_row.addWidget(self.clear_cache_button)
        form.addRow("Reference cache", cache_row)
        self.upscale_section = UpscaleSettings(settings, form)
        self.cache_worker = None
        self.show_reference_cache()
        body = QWidget()
        body.setLayout(form)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(body)
        layout.addWidget(scroll, 1)
        self.resize(690, 800)
        managed_actions = QHBoxLayout()
        self.scaffold_button = QPushButton("Create RunPod configuration…")
        self.check_button = QPushButton("Check RunPod setup")
        self.scaffold_button.clicked.connect(self.scaffold_runpod)
        self.check_button.clicked.connect(self.check_runpod)
        managed_actions.addWidget(self.scaffold_button)
        managed_actions.addWidget(self.check_button)
        layout.addLayout(managed_actions)
        self.setup_status = selectable("Configuration contains file paths only. Setup checks do not rent compute.")
        layout.addWidget(self.setup_status)
        runpod_button = QPushButton("Use existing RunPod setup…")
        runpod_button.clicked.connect(self.import_runpod)
        layout.addWidget(runpod_button)
        note = selectable(
            "Uses your installed, signed-in agent CLI. Subscription usage limits apply. "
            "Selected references may be sent to that AI provider. "
            "Desktop-managed RunPod keeps compute until verified downloads and five minutes idle, "
            "or explicit shutdown. Keep simpleGen running for automatic shutdown."
        )
        note.setObjectName("muted")
        layout.addWidget(note)
        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)
        self.managed_runpod.toggled.connect(self.managed_changed)
        self.cloud.toggled.connect(self.location_changed)
        self.managed_changed(settings.managed_runpod)
        self.provider.currentIndexChanged.connect(self.change_provider)
        self.tier.currentTextChanged.connect(self.refresh_models)
        self.model.currentTextChanged.connect(self.refresh_reasoning)
        self.refresh_models()
        if settings.model:
            self.model.setCurrentText(settings.model)
        self.reasoning.setCurrentText(settings.reasoning)

    def managed_changed(self, enabled: bool) -> None:
        if enabled:
            self.cloud.setChecked(True)
            self.cloud_name.setText("RunPod")
        self.cloud.setEnabled(not enabled)
        for field in (self.url, self.token_env, self.browser_login_file, self.cloud_name):
            field.setEnabled(not enabled)

    def location_changed(self, cloud: bool) -> None:
        previous = replace(
            self.settings, comfy_url=self.url.text().strip(),
            token_env=self.token_env.text().strip(),
            browser_login_file=self.browser_login_file.text().strip(),
        )
        self.settings = with_generation_location(previous, cloud)
        self.url.setText(self.settings.comfy_url)
        self.token_env.setText(self.settings.token_env)
        self.browser_login_file.setText(self.settings.browser_login_file)
        self.managed_runpod.setChecked(self.settings.managed_runpod)
        self.managed_changed(self.settings.managed_runpod)

    def browse_file(self, field) -> None:
        path, _filter = QFileDialog.getOpenFileName(self, "Choose JSON file", field.text(), "JSON files (*.json)")
        if path:
            field.setText(path)

    def scaffold_runpod(self) -> None:
        from .credentials import scaffold_runpod

        self.start_operation(scaffold_runpod, self.scaffold_finished)

    def scaffold_finished(self, result: dict) -> None:
        for key, field in (("config_file", self.runpod_config_file),
                           ("credentials_file", self.runpod_credentials_file),
                           ("schema_file", self.runpod_schema_file)):
            field.setText(str(result[key]))
        self.managed_runpod.setChecked(True)
        self.setup_status.setText("Configuration files are ready. Enter your RunPod template ID and Global Volume ID "
                                  "in the profile, fill in the protected credentials, and check setup.")

    def check_runpod(self) -> None:
        from .credentials import check_setup

        candidate = self.collect_settings()
        self.start_operation(lambda: check_setup(candidate), self.setup_checked)

    def setup_checked(self, result: dict) -> None:
        issues = "\n".join(str(item) for item in result.get("issues", []))
        self.setup_status.setText("RunPod setup is ready." if result.get("ready") else "Setup needs attention.\n" + issues)

    def start_operation(self, operation, succeeded) -> None:
        if self.operation_worker is not None:
            return
        self.setup_status.setText("Checking configuration…")
        self.scaffold_button.setEnabled(False)
        self.check_button.setEnabled(False)
        self.buttons.setEnabled(False)
        self.operation_worker = OperationWorker(operation, self)
        self.operation_worker.succeeded.connect(succeeded)
        self.operation_worker.failed.connect(self.setup_status.setText)
        self.operation_worker.finished.connect(self.operation_finished)
        self.operation_worker.start()

    def operation_finished(self) -> None:
        worker, self.operation_worker = self.operation_worker, None
        self.scaffold_button.setEnabled(True)
        self.check_button.setEnabled(True)
        self.buttons.setEnabled(True)
        retire_worker(self, worker)
        if self.pending_close:
            super().reject()

    def reject(self) -> None:
        if self.operation_worker is not None or self.upscale_section.busy():
            self.pending_close = True
            return
        super().reject()

    def closeEvent(self, event: QCloseEvent) -> None:
        if self.operation_worker is not None or self.upscale_section.busy():
            self.pending_close = True
            event.ignore()
            return
        super().closeEvent(event)

    def change_provider(self) -> None:
        selected = change_agent(
            replace(
                self.settings,
                model_tier=self.tier.currentText(),
                reasoning=self.reasoning.currentText(),
            ),
            self.provider.currentData(),
        )
        self.tier.setCurrentText(selected.model_tier)
        self.refresh_models()
        self.model.setCurrentText(selected.model)
        self.reasoning.setCurrentText(selected.reasoning)

    def import_runpod(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self,
            "Choose existing runpodConfig folder",
            str(Path.home() / "Documents" / "runpodConfig"),
        )
        if not folder:
            return
        try:
            config = with_runpod_config(self.settings, Path(folder))
        except ValueError as exc:
            QMessageBox.warning(self, "Cannot read RunPod setup", str(exc))
            return
        self.cloud.setChecked(True)
        self.url.setText(config.comfy_url)
        self.browser_login_file.setText(config.browser_login_file)
        self.cloud_name.setText(config.cloud_name)
        self.runpod_config_file.setText(config.runpod_config_file)
        self.runpod_schema_file.setText(config.runpod_schema_file)
        self.runpod_credentials_file.setText(config.runpod_credentials_file)
        self.managed_runpod.setChecked(config.managed_runpod)

    def refresh_models(self, *_args) -> None:
        from .agents import model_options, suggest_model

        provider = self.provider.currentData()
        current = self.model.currentText()
        if self.tier.currentText() != "Custom" and not suggest_model(provider, self.tier.currentText()):
            # This agent has no model for that tier (e.g. Grok): Custom = its own default.
            self.tier.blockSignals(True)
            self.tier.setCurrentText("Custom")
            self.tier.blockSignals(False)
            current = ""
        self.model.blockSignals(True)
        self.model.clear()
        self.model.addItems(model_options(provider))
        self.model.setCurrentText(
            current
            if self.tier.currentText() == "Custom"
            else suggest_model(provider, self.tier.currentText())
        )
        self.model.blockSignals(False)
        self.refresh_reasoning()

    def refresh_reasoning(self, *_args) -> None:
        from .agents import reasoning_options

        current = self.reasoning.currentText() or self.settings.reasoning
        options = reasoning_options(self.provider.currentData(), self.model.currentText())
        self.reasoning.clear()
        self.reasoning.addItems(options)
        self.reasoning.setCurrentText(
            current
            if current in options
            else ("low" if "low" in options else options[0] if options else "auto")
        )

    def collect_settings(self) -> Settings:
        return replace(
            self.settings,
            provider=self.provider.currentData(),
            model_tier=self.tier.currentText(),
            model=self.model.currentText().strip(),
            reasoning=self.reasoning.currentText().strip(),
            steps=self.steps.value(),
            megapixels=self.megapixels.value(),
            duration=self.duration.value(),
            aspect_ratio=self.aspect_ratio.currentData() or "",
            comfy_url=self.url.text().strip(),
            token_env=self.token_env.text().strip(),
            browser_login_file=self.browser_login_file.text().strip(),
            cloud=self.cloud.isChecked(),
            cloud_name=self.cloud_name.text().strip(),
            managed_runpod=self.managed_runpod.isChecked(),
            runpod_config_file=self.runpod_config_file.text().strip(),
            runpod_credentials_file=self.runpod_credentials_file.text().strip(),
            runpod_schema_file=self.runpod_schema_file.text().strip(),
            runpod_default_workflow=self.runpod_default_workflow.text().strip(),
            runpod_default_model=self.runpod_default_model.text().strip(),
            runpod_default_clip=self.runpod_default_clip.text().strip(),
            failure_shutdown_minutes=self.failure_shutdown.value(),
            startup_timeout_minutes=self.startup_timeout.value(),
            keep_warm_minutes=self.keep_warm.value(),
            alert_webhook_url=self.alert_webhook.text().strip(),
            **self.upscale_section.values(),
        )

    def show_reference_cache(self) -> None:
        from .refcache import usage
        used = usage()
        self.reference_cache_label.setText(f"{_size(used['bytes'])} cached · {_size(used['unused_bytes'])} unused")

    def clear_reference_cache(self) -> None:
        """Collect unpinned reference copies off the UI thread."""
        from .refcache import gc
        if self.cache_worker is not None:
            return
        self.clear_cache_button.setEnabled(False)
        worker = OperationWorker(gc, self)
        self.cache_worker = worker
        worker.failed.connect(lambda message: QMessageBox.warning(self, "Reference cache", message))
        worker.finished.connect(self.reference_cache_cleared)
        worker.start()

    def reference_cache_cleared(self) -> None:
        worker, self.cache_worker = self.cache_worker, None
        retire_worker(self, worker)
        self.clear_cache_button.setEnabled(True)
        self.show_reference_cache()

    def adopt_live_schema(self, *_args, confirm=None) -> None:
        """Explicit: keep the old cache under a timestamped name and install the live schema."""
        from .cost_safety import adopt_live_schema, pointer_path
        schema_file = self.runpod_schema_file.text().strip()
        if not schema_file or not pointer_path().is_file():
            QMessageBox.information(self, "No live schema", "No live RunPod schema has been saved yet.")
            return
        ask = confirm or (lambda: QMessageBox.question(
            self, "Use latest live schema",
            "Install the latest live RunPod node schema as the cache? The current file is kept under a "
            "timestamped name. Armed and prepared batches will need preparing again.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel) == QMessageBox.StandardButton.Yes)
        if not ask():
            return
        try:
            result = adopt_live_schema(replace(self.settings, runpod_schema_file=schema_file))
        except (OSError, ValueError, KeyError, TypeError) as error:
            QMessageBox.warning(self, "Schema not installed", str(error))
            return
        self.adopted_schema = result

    def accept(self) -> None:
        if self.operation_worker is not None:
            return
        candidate = self.collect_settings()
        from .agents import agent_problem
        problem = agent_problem(candidate)
        if problem:
            QMessageBox.warning(self, "Agent settings can't run", problem)
            return
        if self.upscale_section.busy():
            return
        try:
            candidate.save()
            self.upscale_section.save_token()
        except (ValueError, OSError) as exc:
            QMessageBox.warning(self, "Settings could not be saved", str(exc))
            return
        self.settings = candidate
        super().accept()


class GenerationWorker(QThread):
    event = Signal(dict)
    succeeded = Signal(dict)
    failed = Signal(str)
    resource_choice = Signal(object)

    def __init__(
        self,
        settings: Settings,
        concept: str,
        resume_folder: Path | None = None,
        parent: QWidget | None = None,
        fork_seed: dict | None = None,
    ):
        super().__init__(parent)
        self.settings = replace(settings)
        self.concept = concept
        self.resume_folder = resume_folder
        self.fork_seed = fork_seed
        self.cancel_event = threading.Event()
        self.detach_event = threading.Event()

    def choose_gpu(self, options: list[dict]) -> dict | None:
        request = {"options": options, "ready": threading.Event(), "choice": None}
        self.resource_choice.emit(request)
        while not request["ready"].wait(0.1):
            if self.cancel_event.is_set() or self.detach_event.is_set():
                return None
        if self.cancel_event.is_set() or self.detach_event.is_set():
            return None
        return request["choice"]

    def run(self) -> None:
        try:
            from .engine import run

            options = {"fork_seed": self.fork_seed} if self.fork_seed is not None else {}
            result = run(
                self.settings,
                self.concept,
                self.event.emit,
                self.cancel_event,
                resume_folder=self.resume_folder,
                detach=self.detach_event,
                choose_gpu=self.choose_gpu,
                **options,
            )
        except Exception as exc:  # noqa: BLE001 — worker boundary must surface every failure to the UI.
            self.failed.emit(str(exc))
        else:
            self.succeeded.emit(result)


def _size(value: int) -> str:
    for unit in ("bytes", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.0f} {unit}" if unit == "bytes" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} GB"


class SavedRunDialog(QDialog):
    """Choose between reconnecting saved work and staging a fresh revision."""

    def __init__(self, state: dict, folder: Path, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Saved run")
        self.setMinimumWidth(560)
        self.choice = None
        state = normalize_state(state)
        self.original = state["concept"]
        layout = QVBoxLayout(self)
        layout.addWidget(selectable(f"Saved run: {folder.name}\nEdit the concept below to prepare a new run."))
        self.concept = QPlainTextEdit(self.original)
        self.concept.setMinimumHeight(150)
        layout.addWidget(self.concept)
        self.explanation = selectable("Resume unchanged reconnects saved work. Edit as new run stages a new project; "
                                      "agent preparation starts only when you click Generate.")
        layout.addWidget(self.explanation)
        buttons = QHBoxLayout()
        self.resume_button = QPushButton("Resume unchanged")
        self.resume_button.clicked.connect(lambda: self.choose("resume"))
        self.new_button = QPushButton("Edit as new run")
        self.new_button.clicked.connect(lambda: self.choose("new"))
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        for button in (self.resume_button, self.new_button, cancel):
            buttons.addWidget(button)
        layout.addLayout(buttons)
        from .oom_transfer import local_oom
        self.rp_button = QPushButton("Add OOM run to RunPod batch…")
        self.rp_button.setVisible(local_oom(folder, state))
        self.rp_button.clicked.connect(lambda: self.choose("runpod"))
        layout.addWidget(self.rp_button)
        from .repair import refusal
        self.repair_button = QPushButton("Repair model selection…")
        self.repair_button.setToolTip("Replace unavailable models and keep the accepted script and prompts.")
        self.repair_button.setVisible(refusal(folder, state) is None)
        self.repair_button.clicked.connect(lambda: self.choose("repair"))
        layout.addWidget(self.repair_button)
        from .repair import salvage_refusal
        self.salvage_button = QPushButton("Rebuild from saved agent work…")
        self.salvage_button.setToolTip("Rebuild the rejected attempt's script and prompts into a valid workflow; "
                                       "no agent pass.")
        self.salvage_button.setVisible(salvage_refusal(folder, state) is None)
        self.salvage_button.clicked.connect(lambda: self.choose("salvage"))
        layout.addWidget(self.salvage_button)
        from .refcache import MANIFEST
        from .repair import submitted
        self.cache_button = QPushButton("Cache references now")
        self.cache_button.setToolTip("Copy this run's references to local disk (checked by hash) so it can run "
                                     "while the linked project folder is unavailable.")
        self.cache_button.setVisible(bool(state.get("workflow_acceptance")) and not (folder / MANIFEST).exists()
                                     and state.get("schema_version", 1) >= 2 and not state.get("batch_parent")
                                     and not submitted(folder, state))
        self.cache_button.clicked.connect(lambda: self.choose("cache"))
        layout.addWidget(self.cache_button)
        self.scene_pending = bool(state.get("h3scene_review")) and not state.get("h3scene_decision")
        if self.scene_pending:
            self.explanation.setText("H3 Scene Chain plan needs overflow review: " + state["h3scene_review"]["reason"])
            self.resume_button.setText("Approve")
            self.resume_button.clicked.disconnect()
            self.resume_button.clicked.connect(lambda: self.choose("scene_approve"))
            deny = QPushButton("Deny")
            deny.clicked.connect(lambda: self.choose("scene_deny"))
            buttons.addWidget(deny)
        if state.get("h3scene_decision", {}).get("action") == "deny":
            self.resume_button.hide()
        self.concept.textChanged.connect(self.update_actions)

    def update_actions(self):
        self.resume_button.setEnabled(self.concept.toPlainText() == self.original)

    def choose(self, choice: str):
        if choice == "resume" and self.concept.toPlainText() != self.original:
            return
        if choice == "new" and not self.concept.toPlainText().strip():
            QMessageBox.warning(self, "Enter a concept", "Enter a concept for the new run.")
            return
        self.choice = choice
        self.accept()


class ScheduleEditDialog(QDialog):
    """Edit one saved local request without touching the Generate page draft."""

    def __init__(self, record: dict, settings: Settings, save, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Edit scheduled request")
        self.setMinimumWidth(570)
        self.original_due = record["due_at"]
        self.settings = settings
        self.save_callback = save
        self.operation_worker = None
        self.saved = False
        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.concept = QPlainTextEdit(record["concept"])
        self.concept.setMinimumHeight(120)
        form.addRow("Concept", self.concept)
        self.workflow = QLineEdit(settings.template_file)
        browse = QPushButton("Browse…")
        browse.clicked.connect(self.browse_workflow)
        workflow_row = QWidget()
        workflow_layout = QHBoxLayout(workflow_row)
        workflow_layout.setContentsMargins(0, 0, 0, 0)
        workflow_layout.addWidget(self.workflow)
        workflow_layout.addWidget(browse)
        form.addRow("Workflow JSON", workflow_row)
        self.provider = QComboBox()
        for label, value in PROVIDERS:
            self.provider.addItem(label, value)
        self.provider.setCurrentIndex(self.provider.findData(settings.provider))
        self.tier = QComboBox()
        self.tier.addItems(["Economy", "Balanced", "Strongest", "Custom"])
        self.tier.setCurrentText(settings.model_tier)
        self.model = QLineEdit(settings.model)
        self.reasoning = QComboBox()
        self.reasoning.setEditable(True)
        self.reasoning.addItems(["auto", "low", "medium", "high"])
        self.reasoning.setCurrentText(settings.reasoning)
        self.provider.currentIndexChanged.connect(self.agent_changed)
        self.tier.currentIndexChanged.connect(self.agent_changed)
        for label, field in (("Agent", self.provider), ("Model tier", self.tier),
                             ("Model", self.model), ("Reasoning", self.reasoning)):
            form.addRow(label, field)
        self.steps = steps_field(settings.steps)
        self.megapixels = number_field(settings.megapixels, 0.05, 16, " MP")
        self.duration = number_field(settings.duration, 1, 3600, " seconds")
        self.aspect_ratio = aspect_field(settings.aspect_ratio)
        self.quantity = QSpinBox()
        self.quantity.setRange(1, 100)
        self.quantity.setValue(settings.quantity)
        for label, field in (("Steps", self.steps), ("Megapixel", self.megapixels),
                             ("Duration", self.duration), ("Aspect ratio", self.aspect_ratio),
                             ("Qty", self.quantity)):
            form.addRow(label, field)
        self.due = date_field()
        local = datetime.fromisoformat(record["due_at"]).astimezone(system_zone())
        self.original_local = (local.year, local.month, local.day, local.hour, local.minute)
        self.due.setDateTime(QDateTime(QDate(local.year, local.month, local.day),
                                       QTime(local.hour, local.minute), QTimeZone.utc()))
        form.addRow("Local date/time", self.due)
        layout.addLayout(form)
        note = selectable("Only a pending local request can be edited. Save creates a new revision.")
        layout.addWidget(note)
        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save |
                                   QDialogButtonBox.StandardButton.Cancel)
        self.buttons.accepted.connect(self.save)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)

    def browse_workflow(self):
        path, _ = QFileDialog.getOpenFileName(self, "Choose workflow JSON", self.workflow.text(), "JSON files (*.json)")
        if path:
            self.workflow.setText(path)

    def agent_changed(self):
        from .agents import reasoning_options, suggest_model
        provider = self.provider.currentData()
        if self.tier.currentText() != "Custom" and not suggest_model(provider, self.tier.currentText()):
            self.tier.blockSignals(True)
            self.tier.setCurrentText("Custom")
            self.tier.blockSignals(False)
        model = suggest_model(provider, self.tier.currentText()) if self.tier.currentText() != "Custom" else self.model.text()
        self.model.setText(model or "")
        options = reasoning_options(provider, model or "")
        current = self.reasoning.currentText()
        self.reasoning.clear()
        self.reasoning.addItems(options)
        self.reasoning.setCurrentText(current if current in options else (options[0] if options else "auto"))

    def values(self):
        selected = self.workflow.text().strip()
        if selected == self.settings.template_file:
            folder, workflow = self.settings.template_folder, self.settings.template_file
        else:
            path = Path(selected).expanduser()
            if not path.is_absolute():
                path = Path(self.settings.template_folder).expanduser() / path
            path = path.resolve()
            folder, workflow = str(path.parent), str(path)
        settings = replace(self.settings, template_folder=folder, template_file=workflow,
                           provider=self.provider.currentData(), model_tier=self.tier.currentText(),
                           model=self.model.text().strip(), reasoning=self.reasoning.currentText().strip(),
                           steps=self.steps.value(), megapixels=self.megapixels.value(),
                           duration=self.duration.value(), aspect_ratio=self.aspect_ratio.currentData() or "",
                           quantity=self.quantity.value())
        from .agents import agent_problem
        problem = agent_problem(settings)
        if problem:
            raise ValueError(problem)
        value = (self.due.date().year(), self.due.date().month(), self.due.date().day(),
                 self.due.time().hour(), self.due.time().minute())
        due = datetime.fromisoformat(self.original_due) if value == self.original_local else selected_time(self.due)
        return settings, self.concept.toPlainText().strip(), due

    def save(self):
        if self.operation_worker is not None:
            return
        try:
            values = self.values()
        except (ValueError, OSError, TypeError) as error:
            QMessageBox.warning(self, "Cannot edit schedule", str(error))
            return
        self.buttons.setEnabled(False)
        worker = OperationWorker(lambda: self.save_callback(*values), self)
        self.operation_worker = worker
        worker.succeeded.connect(lambda _result: setattr(self, "saved", True))
        worker.failed.connect(lambda message: QMessageBox.warning(self, "Cannot edit schedule", message))
        worker.finished.connect(self.operation_finished)
        worker.start()

    def operation_finished(self):
        worker, self.operation_worker = self.operation_worker, None
        self.buttons.setEnabled(True)
        retire_worker(self, worker)
        if self.saved:
            super().accept()

    def reject(self):
        if self.operation_worker is None:
            super().reject()

    def closeEvent(self, event):
        if self.operation_worker is not None:
            event.ignore()
        else:
            super().closeEvent(event)


class MainWindow(QMainWindow):
    def __init__(self, settings: Settings | None = None):
        super().__init__()
        self.settings = settings or Settings.load()
        self.display_settings = self.settings
        self.agent_choices: dict[str, Settings] = {}
        self.worker: GenerationWorker | None = None
        self.cloud_worker = None
        self.choice_dialog = None
        self.cloud_state = None
        self.current_folder: Path | None = None
        self.closing = False
        self.scheduler = Scheduler()
        self.schedule_probe = None
        self.preflight_worker = None
        self.override_worker = None
        self.staged_fork = None
        self.active_fork_seed = None
        self.active_schedule = None
        self.batch_worker = None
        self.batch_worker_folder = None
        self.batch_line = ""  # Running cloud batch status (its own slot; never the draft).
        self.addition_worker = None
        self.salvage_worker = None
        self.batch_stage_worker = None
        from .refcache import gc_quietly
        threading.Thread(target=gc_quietly, name="reference-cache-gc", daemon=True).start()
        self.batch_store = None
        self.catalog_worker = None
        self.oom_source = None
        self.upscale_worker = None
        self.upscale_note = ""
        self.setWindowTitle("simpleGen")
        self.resize(960, 860)
        self.setMinimumSize(740, 650)
        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(22, 18, 22, 18)
        layout.setSpacing(12)
        header = QHBoxLayout()
        title = QLabel("simpleGen")
        title.setObjectName("title")
        header.addWidget(title)
        tagline = QLabel("Simple. Effective. Useful.")
        tagline.setObjectName("muted")
        header.addWidget(tagline)
        header.addStretch()
        self.settings_button = QPushButton("Settings")
        self.settings_button.clicked.connect(self.edit_settings)
        header.addWidget(self.settings_button)
        layout.addLayout(header)
        self.cloud_banner = selectable()
        self.cloud_banner.setObjectName("cloud")
        layout.addWidget(self.cloud_banner)
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)
        cloud_actions = QHBoxLayout()
        self.cloud_status_button = QPushButton("Cloud session status")
        self.cloud_shutdown_button = QPushButton("Shut down compute")
        self.cloud_status_button.clicked.connect(lambda: self.cloud_action("status"))
        self.cloud_shutdown_button.clicked.connect(lambda: self.cloud_action("cancel"))
        cloud_actions.addWidget(self.cloud_status_button)
        cloud_actions.addWidget(self.cloud_shutdown_button)
        cloud_actions.addStretch()
        layout.addLayout(cloud_actions)
        self.build_generate()
        self.build_watch()
        self.workflow_timer = QTimer(self)
        self.workflow_timer.setInterval(3000)
        self.workflow_timer.timeout.connect(self.poll_workflows)
        self.workflow_timer.start()
        self.status = selectable("Ready. Enter a concept and generate.")
        self.status.setObjectName("muted")
        layout.addWidget(self.status)
        self.global_status = selectable("")
        self.global_status.setObjectName("cloud")
        self.global_status.setVisible(False)
        layout.addWidget(self.global_status)
        self.cloud_timer = QTimer(self)
        self.cloud_timer.setInterval(10000)
        self.cloud_timer.timeout.connect(self.poll_cloud_status)
        self.cloud_timer.start()
        self.global_worker = None
        self.shown_alerts: set = set()
        self.tray = None
        self.alert_text = ""
        self.global_timer = QTimer(self)
        self.global_timer.setInterval(10000)
        self.global_timer.timeout.connect(self.poll_registered_global_sessions)
        self.global_timer.start()
        self.update_summary()
        self.schedule_timer = QTimer(self)
        self.schedule_timer.setInterval(5000)
        self.schedule_timer.timeout.connect(self.poll_schedules)
        self.schedule_timer.start()
        self.refresh_schedule_count()
        self.batch_timer = QTimer(self)
        self.batch_timer.setInterval(5000)
        self.batch_timer.timeout.connect(self.poll_batches)
        self.batch_timer.start()
        self.upscale_timer = QTimer(self)
        self.upscale_timer.setInterval(15000)
        self.upscale_timer.timeout.connect(self.poll_upscales)
        self.upscale_timer.start()
        QTimer.singleShot(0, self.poll_upscales)  # Reconcile upscales left from an earlier session.

    def build_generate(self) -> None:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(16, 16, 16, 16)
        form = QFormLayout()
        self.project_folder = QLineEdit(self.settings.project_folder)
        self.template_folder = QLineEdit(self.settings.template_folder)
        self.output_folder = QLineEdit(self.settings.output_folder)
        self.folder_buttons = []
        for label, field in (
            ("Project references", self.project_folder),
            ("Workflow folder", self.template_folder),
            ("Save projects in", self.output_folder),
        ):
            row = QWidget()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.addWidget(field)
            button = QPushButton("Browse…")
            button.clicked.connect(lambda _checked=False, edit=field: self.choose_folder(edit))
            self.folder_buttons.append(button)
            row_layout.addWidget(button)
            form.addRow(label, row)
        self.template = QComboBox()
        self.template.setMinimumContentsLength(30)
        self.template.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self.template_folder.editingFinished.connect(self.refresh_templates)
        form.addRow("Workflow", self.template)
        layout.addLayout(form)
        self.refresh_templates()
        location = QHBoxLayout()
        location.addWidget(QLabel("Generate on"))
        self.location_group = QButtonGroup(self)
        self.location_buttons = {}
        for label, cloud in (("Local", False), ("RunPod / Cloud", True)):
            button = QRadioButton(label)
            self.location_group.addButton(button)
            self.location_buttons[cloud] = button
            location.addWidget(button)
            button.clicked.connect(lambda _checked=False, value=cloud: self.select_location(value))
        location.addStretch()
        layout.addLayout(location)
        endpoint = QHBoxLayout()
        endpoint.addWidget(QLabel("ComfyUI URL"))
        self.endpoint = QLineEdit(self.settings.comfy_url)
        self.endpoint.setPlaceholderText("http://127.0.0.1:8188")
        self.endpoint.textEdited.connect(self.endpoint_edited)
        endpoint.addWidget(self.endpoint)
        layout.addLayout(endpoint)
        self.concept_label = QLabel("Your concept")
        layout.addWidget(self.concept_label)
        self.concept = QPlainTextEdit()
        self.concept.setPlaceholderText(
            "Describe the video you want. Your agent will use the linked "
            "show notes, character references, and workflow to create it."
        )
        self.concept.setMinimumHeight(150)
        layout.addWidget(self.concept, 1)
        specs = QHBoxLayout()
        specs.setSpacing(4)
        self.steps = steps_field(self.settings.steps)
        self.megapixels = number_field(self.settings.megapixels, 0.05, 16, " MP")
        self.duration = number_field(self.settings.duration, 1, 3600, " seconds")
        self.aspect_ratio = aspect_field(self.settings.aspect_ratio)
        self.quantity = QSpinBox()
        self.quantity.setRange(1, 100)
        self.quantity.setValue(self.settings.quantity)
        self.quantity.setToolTip("Number of variations. Each video renders individually using the same workflow settings.")
        specs.addWidget(QLabel("Steps"))
        specs.addWidget(self.steps)
        specs.addWidget(QLabel("Megapixel"))
        specs.addWidget(self.megapixels)
        specs.addWidget(QLabel("Duration"))
        specs.addWidget(self.duration)
        specs.addWidget(QLabel("Aspect ratio"))
        specs.addWidget(self.aspect_ratio)
        specs.addWidget(QLabel("Qty"))
        specs.addWidget(self.quantity)
        self.upscale = QCheckBox("4K60")
        self.upscale.setChecked(self.settings.upscale)
        specs.addWidget(self.upscale)
        self.upscale_method = method_field(self.settings.upscale_preset)
        self.upscale_method.setToolTip("Upscale method for this request's 4K60 version.")
        specs.addWidget(self.upscale_method)
        self.upscale.toggled.connect(lambda _checked: self.update_upscale_method())
        self.update_upscale_toggle()
        specs.addStretch()
        layout.addLayout(specs)
        providers = QHBoxLayout()
        providers.addWidget(QLabel("Agent"))
        self.provider_group = QButtonGroup(self)
        self.provider_buttons: dict[str, QRadioButton] = {}
        for label, provider in PROVIDERS:
            button = QRadioButton(label)
            self.provider_buttons[provider] = button
            self.provider_group.addButton(button)
            providers.addWidget(button)
            button.clicked.connect(lambda _checked=False, value=provider: self.select_agent(value))
        providers.addStretch()
        layout.addLayout(providers)
        self.batch_mode = QComboBox()
        self.batch_mode.addItem("Batch & schedule (managed RunPod)", "batch")
        self.batch_mode.addItem("One-off (managed RunPod)", "one_off")
        self.batch_mode.setToolTip("A batch prepares all jobs before renting one GPU session. One-off starts a single run with its own approval.")
        layout.addWidget(self.batch_mode)
        self.catalog_status = selectable("")
        self.catalog_status.hide()
        layout.addWidget(self.catalog_status)
        self.summary = selectable()
        self.summary.setObjectName("muted")
        layout.addWidget(self.summary)
        actions = QHBoxLayout()
        self.generate_button = QPushButton("Generate")
        self.generate_button.setObjectName("primary")
        self.generate_button.clicked.connect(lambda: self.start_generation())
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self.cancel_generation)
        self.resume_button = QPushButton("Resume saved run…")
        self.resume_button.clicked.connect(self.resume_generation)
        self.anyway_button = QPushButton("Generate anyway…")
        self.anyway_button.setToolTip("Recheck saved agent work rejected only for response format; no agent call is repeated.")
        self.anyway_button.clicked.connect(self.generate_anyway)
        actions.addWidget(self.generate_button)
        actions.addWidget(self.cancel_button)
        self.queue_local_button = QPushButton("Queue locally")
        self.queue_local_button.setToolTip("Schedule this draft on Local now; it starts when the local queue is free.")
        self.queue_local_button.clicked.connect(self.queue_locally)
        actions.addWidget(self.queue_local_button)
        self.add_running_button = QPushButton("Add to running batch…")
        self.add_running_button.setToolTip("Prepare this draft and queue it on a running (or warm) RunPod batch's GPU.")
        self.add_running_button.clicked.connect(self.add_draft_to_running_batch)
        actions.addWidget(self.add_running_button)
        self.schedule_enabled = QCheckBox("Schedule")
        self.schedule_time = date_field()
        self.schedule_time.setEnabled(False)
        self.schedule_enabled.toggled.connect(self.update_schedule_controls)
        self.scheduled_button = QPushButton("Scheduled (0)")
        self.scheduled_button.clicked.connect(self.show_schedules)
        self.batches_button = QPushButton("Batches (0)")
        self.batches_button.clicked.connect(self.show_batches)
        actions.addWidget(self.schedule_enabled)
        actions.addWidget(self.schedule_time)
        actions.addWidget(self.scheduled_button)
        actions.addWidget(self.batches_button)
        actions.addStretch()
        actions.addWidget(self.resume_button)
        actions.addWidget(self.anyway_button)
        layout.addLayout(actions)
        self.jobs_status = selectable("")
        self.jobs_status.setObjectName("muted")
        self.jobs_box = QWidget()
        jobs_row = QHBoxLayout(self.jobs_box)
        jobs_row.setContentsMargins(0, 0, 0, 0)
        jobs_row.addWidget(self.jobs_status, 1)
        # One explicit control per running job; nothing shuts down a batch's pod implicitly.
        self.cancel_local_button = QPushButton("Cancel local run")
        self.cancel_local_button.clicked.connect(self.cancel_generation)
        self.stop_batch_button = QPushButton("Shut down batch…")
        self.stop_batch_button.setToolTip("Deliberately shut down the running batch's GPU (asks first).")
        self.stop_batch_button.clicked.connect(self.cancel_batch)
        jobs_row.addWidget(self.cancel_local_button)
        jobs_row.addWidget(self.stop_batch_button)
        self.jobs_box.hide()
        layout.addWidget(self.jobs_box)
        self.schedule_status = selectable("")
        self.schedule_status.setObjectName("muted")
        self.schedule_status.hide()
        layout.addWidget(self.schedule_status)
        self.upscale_status = selectable("")
        self.upscale_status.setObjectName("muted")
        self.upscale_status.hide()
        layout.addWidget(self.upscale_status)
        self.oom_button = QPushButton("Local run out of memory — Add to RunPod batch…")
        self.oom_button.clicked.connect(lambda: self.add_oom_to_batch(self.oom_source))
        self.oom_button.hide()
        layout.addWidget(self.oom_button)
        self.repair_button = QPushButton("Repair…")
        self.repair_button.setToolTip("Replace unavailable models and keep the accepted script and prompts.")
        self.repair_button.clicked.connect(lambda: self.current_folder and self.open_repair(self.current_folder))
        self.repair_button.hide()
        layout.addWidget(self.repair_button)
        self.salvage_button = QPushButton("Rebuild from saved agent work…")
        self.salvage_button.clicked.connect(lambda: self.current_folder and self.salvage_saved(self.current_folder))
        self.salvage_button.hide()
        layout.addWidget(self.salvage_button)
        self.progress = QProgressBar()
        self.progress.setValue(0)
        layout.addWidget(self.progress)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(90)
        self.log.setPlaceholderText("Progress and the success or failure report appear here.")
        self.log.setMaximumBlockCount(500)
        layout.addWidget(self.log)
        self.report_button = QPushButton("Open run folder / report")
        self.report_button.setEnabled(False)
        self.report_button.clicked.connect(self.open_run_folder)
        layout.addWidget(self.report_button, alignment=Qt.AlignmentFlag.AlignRight)
        self.tabs.addTab(page, "Generate")

    def build_watch(self) -> None:
        page = QWidget()
        layout = QVBoxLayout(page)
        open_button = QPushButton("Open completed project…")
        open_button.clicked.connect(self.open_completed_project)
        layout.addWidget(open_button, alignment=Qt.AlignmentFlag.AlignRight)
        self.variation_picker = QComboBox()
        self.variation_picker.setVisible(False)
        self.variation_picker.currentIndexChanged.connect(self.select_variation)
        layout.addWidget(self.variation_picker)
        self.video = QVideoWidget()
        self.video.setMinimumHeight(280)
        layout.addWidget(self.video, 1)
        self.player = QMediaPlayer(self)
        self.audio = QAudioOutput(self)
        self.player.setAudioOutput(self.audio)
        self.player.setVideoOutput(self.video)
        self.player.errorOccurred.connect(self.playback_error)
        controls = QHBoxLayout()
        self.play_button = QPushButton("Play")
        self.play_button.setEnabled(False)
        self.play_button.clicked.connect(self.toggle_playback)
        self.player.playbackStateChanged.connect(
            lambda state: self.play_button.setText(
                "Pause" if state == QMediaPlayer.PlaybackState.PlayingState else "Play"
            )
        )
        controls.addWidget(self.play_button)
        self.seek = QSlider(Qt.Orientation.Horizontal)
        self.seek.setRange(0, 0)
        self.seek.sliderMoved.connect(self.player.setPosition)
        self.player.durationChanged.connect(lambda duration: self.seek.setRange(0, duration))
        self.player.positionChanged.connect(self.update_position)
        controls.addWidget(self.seek, 1)
        self.time_label = QLabel("0:00 / 0:00")
        controls.addWidget(self.time_label)
        layout.addLayout(controls)
        self.video_info = selectable("Your final cut will appear here.")
        self.video_path = selectable()
        self.folder_path = selectable()
        layout.addWidget(self.video_info)
        layout.addWidget(self.video_path)
        layout.addWidget(self.folder_path)
        self.upscale_panel = UpscalePanel()
        self.upscale_panel.play.connect(self.play_version)
        self.upscale_panel.action.connect(self.upscale_action)
        layout.addWidget(self.upscale_panel)
        self.shown_final: Path | None = None
        self.shown_metadata: dict = {}
        self.result_folder_button = QPushButton("Open folder")
        self.result_folder_button.setEnabled(False)
        self.result_folder_button.clicked.connect(self.open_result_folder)
        layout.addWidget(self.result_folder_button, alignment=Qt.AlignmentFlag.AlignRight)
        self.result_folder: Path | None = None
        self.tabs.addTab(page, "Watch")

    def choose_folder(self, field: QLineEdit) -> None:
        selected = QFileDialog.getExistingDirectory(self, "Choose folder", field.text())
        if selected:
            field.setText(selected)
            if field is self.template_folder:
                self.refresh_templates()

    def poll_workflows(self) -> None:
        if self.worker is None and not self.template.view().isVisible():
            self.refresh_templates()

    def refresh_templates(self) -> None:
        from .builtin_workflows import (
            REFMOD_LABEL,
            SCENE_LABEL,
            ensure_bundled_templates,
            ensure_refmod_template,
            ensure_scene_template,
            template_label,
        )
        selected = self.template.currentData() or self.settings.template_file
        folder_text = self.template_folder.text().strip()
        folder = Path(folder_text).expanduser()
        entries = []
        try:
            candidates = sorted(folder.iterdir()) if folder_text and folder.is_dir() else []
            for path in candidates:
                name = path.name.casefold()
                if (
                    path.is_file()
                    and path.suffix.casefold() == ".json"
                    and "5minvid" not in name
                    and not ("prompt" in name and ("sample" in name or "example" in name))
                ):
                    entries.append((template_label(path), str(path)))
        except OSError:
            pass
        builtin = ensure_scene_template()
        if builtin is not None:
            entries.append((SCENE_LABEL, str(builtin)))
        builtin_refmod = ensure_refmod_template()
        if builtin_refmod is not None:
            entries.append((REFMOD_LABEL, str(builtin_refmod)))
        entries.extend((label, str(path)) for label, path in ensure_bundled_templates())
        seed = self.staged_fork
        if (seed and folder_text == seed["template_folder"]
                and not any(value == seed["template_file"] for _, value in entries)):
            entries.insert(0, (f"Saved snapshot — {template_label(seed['template_file'])}", seed["template_file"]))
        current = [(self.template.itemText(i), self.template.itemData(i))
                   for i in range(self.template.count())]
        if entries == current:
            return
        self.template.clear()
        for label, value in entries:
            self.template.addItem(label, value)
        index = self.template.findData(selected)
        if index < 0 and selected:
            index = self.template.findText(template_label(selected))
        if index >= 0:
            self.template.setCurrentIndex(index)

    def show_workflow(self, settings: Settings) -> None:
        """Show the destination's own workflow; an unset one stays unselected."""
        from .builtin_workflows import template_label

        self.template_folder.setText(settings.template_folder)
        self.template.setCurrentIndex(-1)
        self.refresh_templates()
        index = self.template.findData(settings.template_file)
        if index < 0 and settings.template_file:
            index = self.template.findText(template_label(settings.template_file))
        self.template.setCurrentIndex(index)

    def collect_settings(self) -> Settings:
        return replace(
            self.settings,
            comfy_url=self.endpoint.text().strip() if not self.settings.managed_runpod else self.settings.comfy_url,
            project_folder=self.project_folder.text().strip(),
            template_folder=self.template_folder.text().strip(),
            template_file=self.template.currentData() or (self.settings.template_file if self.staged_fork else ""),
            output_folder=self.output_folder.text().strip(),
            steps=self.steps.value(),
            megapixels=self.megapixels.value(),
            duration=self.duration.value(),
            aspect_ratio=self.aspect_ratio.currentData() or "",
            quantity=self.quantity.value(),
            upscale=self.upscale.isChecked() and self.upscale.isEnabled(),
            upscale_preset=self.upscale_method.currentData(),
        )

    def update_upscale_toggle(self) -> None:
        configured = bool(self.settings.upscale_url.strip())
        self.upscale.setEnabled(configured)
        if not configured:
            self.upscale.setChecked(False)
        self.upscale.setToolTip(
            "After the final cut is saved, make a 4K/60 version on the upscale host. The render never waits for it; "
            "upscaling a long video can take hours." if configured
            else "Set up an upscale service in Settings to enable 4K60.")
        self.update_upscale_method()

    def update_upscale_method(self) -> None:
        self.upscale_method.setEnabled(self.upscale.isEnabled() and self.upscale.isChecked())

    def show_upscale_method(self, preset: str) -> None:
        field = method_field(preset)
        self.upscale_method.clear()
        for index in range(field.count()):
            self.upscale_method.addItem(field.itemText(index), field.itemData(index))
        self.upscale_method.setCurrentIndex(field.currentIndex())

    def endpoint_edited(self, value: str) -> None:
        if self.worker is None and not self.settings.managed_runpod:
            self.settings = replace(self.settings, comfy_url=value.strip())
            self.display_settings = self.settings

    def select_location(self, cloud: bool) -> None:
        """Change the draft's destination only; a running job keeps its own settings and monitoring."""
        self.settings = with_generation_location(self.collect_settings(), cloud)
        if self.worker is None:
            self.display_settings = self.settings
            self.cloud_state = None
            self.current_folder = None
            self.report_button.setEnabled(False)
        self.show_workflow(self.settings)
        self.update_summary()
        if self.settings.managed_runpod:
            self.refresh_runpod_catalog()
        else:
            self.catalog_status.hide()

    def refresh_runpod_catalog(self) -> None:
        if self.catalog_worker is not None or not self.settings.managed_runpod:
            return
        from .batches import _catalog
        config = self.collect_settings()
        self.catalog_status.setText("Checking RunPod GPU catalog… No compute is being rented.")
        self.catalog_status.show()
        self.catalog_worker = OperationWorker(lambda: {"rows": _catalog(config)}, self)
        def received(result):
            if self.settings.managed_runpod:
                rows = result["rows"]
                from .batch_ui import catalog_summary
                self.catalog_status.setText(catalog_summary(rows) + " Open Batches for prices and GPU choices.")
        self.catalog_worker.succeeded.connect(received)
        self.catalog_worker.failed.connect(lambda message: self.catalog_status.setText(f"RunPod catalog unavailable: {message}. Retry by opening Batches.") if self.settings.managed_runpod else None)
        self.catalog_worker.finished.connect(self.catalog_finished)
        self.catalog_worker.start()

    def catalog_finished(self):
        worker, self.catalog_worker = self.catalog_worker, None
        if worker:
            worker.deleteLater()
        if self.closing:
            self.close()

    def edit_settings(self) -> None:
        dialog = SettingsDialog(self.collect_settings(), self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.settings = dialog.settings
            if self.worker is None:
                self.display_settings = self.settings
            self.steps.setValue(self.settings.steps or 25)
            self.megapixels.setValue(self.settings.megapixels)
            self.duration.setValue(self.settings.duration)
            self.aspect_ratio.setCurrentIndex(self.aspect_ratio.findData(self.settings.aspect_ratio or "16:9"))
            self.quantity.setValue(self.settings.quantity)
            self.upscale.setChecked(self.settings.upscale)
            self.show_upscale_method(self.settings.upscale_preset)
            self.update_upscale_toggle()
            self.update_summary()
            QTimer.singleShot(0, self.poll_upscales)

    def select_agent(self, provider: str) -> None:
        if provider != self.settings.provider:
            self.agent_choices[self.settings.provider] = replace(self.settings)
            choice = self.agent_choices.get(provider) or change_agent(self.settings, provider)
            self.settings = replace(
                self.settings,
                provider=provider,
                model_tier=choice.model_tier,
                model=choice.model,
                reasoning=choice.reasoning,
            )
        if self.worker is None:
            self.display_settings = self.settings
        self.update_summary()

    def update_agent_selection(self, config: Settings) -> None:
        self.provider_buttons[config.provider].setChecked(True)
        self.summary.setText(
            f"{config.model_tier} · {config.model or 'provider model'} · reasoning: {config.reasoning}"
        )

    def update_summary(self) -> None:
        config = self.display_settings
        active = self.settings  # Draft controls always show the draft, never a running job.
        self.location_buttons[active.cloud].setChecked(True)
        self.endpoint.setText(active.comfy_url if not active.managed_runpod else "Assigned when the RunPod session connects")
        self.endpoint.setEnabled(not active.managed_runpod)
        self.update_agent_selection(self.settings)
        self.cloud_banner.setText(
            f"CLOUD GPU — {config.cloud_name or 'Cloud instance'} — "
            "Costs money. Billing may continue while the instance runs."
        )
        self.cloud_banner.setVisible(config.cloud)
        self.cancel_button.setText("Shut down compute" if config.managed_runpod else "Cancel")
        for button in (self.cloud_status_button, self.cloud_shutdown_button):
            button.setVisible(config.managed_runpod)
        self.cloud_status_button.setEnabled(self.cloud_worker is None and self.batch_worker is None)
        self.cloud_shutdown_button.setEnabled(
            self.worker is None and self.batch_worker is None and self.cloud_worker is None
            and self.current_folder is not None
        )
        if self.cloud_state is not None and config.managed_runpod:
            self.show_cloud_status(self.cloud_state)
        self.update_schedule_controls()
        self.batch_mode.setVisible(active.managed_runpod)
        self.batches_button.setVisible(active.managed_runpod or config.managed_runpod)
        self.queue_local_button.setVisible(not active.cloud)
        self.add_running_button.setVisible(active.managed_runpod)
        self.refresh_batch_count()
        self.update_jobs_status()

    def set_running(self, running: bool) -> None:
        """The local slot's run state. Draft controls stay editable and keep showing the draft."""
        self.cancel_button.setEnabled(running)  # Only ever the local-slot run.
        self.resume_button.setEnabled(not running)
        self.anyway_button.setEnabled(not running)
        self.oom_button.setEnabled(not running)
        self.repair_button.setEnabled(not running)
        self.salvage_button.setEnabled(not running)
        self.concept.setEnabled(True)
        self.concept_label.setText("Next concept (the running job uses its saved request)" if running
                                   else "Your concept")
        self.cloud_shutdown_button.setEnabled(
            not running and self.batch_worker is None and self.cloud_worker is None
            and self.current_folder is not None
        )
        self.update_schedule_controls()
        self.update_jobs_status()

    def update_jobs_status(self) -> None:
        """Running jobs, each in its own slot: one local run and one cloud batch at most."""
        lines = []
        if self.worker is not None:
            where = "Cloud one-off" if self.display_settings.cloud else "Local run"
            folder = f" · {self.current_folder.name}" if self.current_folder else ""
            lines.append(f"{where}{folder}: {self.status.text()}")
        if self.batch_worker is not None:
            name = self.batch_worker_folder.name[:8] if self.batch_worker_folder else ""
            lines.append(f"Cloud batch {name}: {self.batch_line or 'running'}")
        self.jobs_status.setText("RUNNING — " + "\n".join(lines) if lines else "")
        self.cancel_local_button.setVisible(self.worker is not None)
        self.cancel_local_button.setText("Shut down cloud run…" if self.worker is not None
                                         and self.display_settings.managed_runpod else "Cancel local run")
        self.stop_batch_button.setVisible(self.batch_worker is not None)
        self.jobs_box.setVisible(bool(lines))

    def update_schedule_controls(self, *_args) -> None:
        # Local work can be scheduled while any job runs (it waits for the local queue).
        enabled = not self.settings.cloud and self.scheduler.owner
        self.schedule_enabled.setEnabled(enabled)
        if self.settings.cloud:
            self.schedule_enabled.setChecked(False)
        self.schedule_time.setEnabled(enabled and self.schedule_enabled.isChecked())
        self.schedule_enabled.setToolTip("Local generation only. Keep simpleGen open and the computer awake."
                                        if self.scheduler.owner else "Schedules are managed by another open simpleGen window.")

    def refresh_schedule_count(self) -> None:
        records = self.scheduler.records()
        count = sum(r["status"] in VISIBLE for r in records)
        overdue = sum(r["status"] == "overdue" for r in records)
        waiting = [r for r in records if r["status"] == "waiting"]
        suffix = f" · {len(waiting)} waiting" if waiting else ""
        self.scheduled_button.setText(f"Scheduled ({count}){suffix}")
        self.scheduled_button.setToolTip(f"{overdue} overdue · automatic retry when eligible" if overdue else "View saved local schedules")
        message = ""
        if waiting:
            reason = waiting[0].get("note") or "Waiting — checking local ComfyUI availability"
            retry = waiting[0].get("next_retry_at")
            when = f" · Next retry {display_time(retry)}" if retry else ""
            message = f"{len(waiting)} scheduled request(s) waiting. Next: {reason}{when}"
        self.schedule_status.setText(message)
        self.schedule_status.setVisible(bool(message))
        if overdue and overdue != getattr(self, "overdue_count", 0):
            message = f"{overdue} scheduled request(s) overdue. Availability will be retried when eligible."
            self.log.appendPlainText(message)
            if self.worker is None:
                self.status.setText(message)
        self.overdue_count = overdue

    def get_batch_store(self):
        from .batches import BatchStore
        folder = Path(self.output_folder.text().strip()).expanduser().resolve()
        if self.batch_store is None or self.batch_store.root != folder / ".simplegen-batches":
            self.batch_store = BatchStore(folder)
        return self.batch_store

    def refresh_batch_count(self) -> None:
        try:
            from .batches import registered_stores
            self.get_batch_store()
            batches = [batch for store in registered_stores() for batch in store.list()]
        except (OSError, ValueError, RuntimeError):
            self.batches_button.setText("Batches (unavailable)")
            return
        visible = [b for b in batches if b.get("status") not in {"complete", "cancelled"}]
        attention = sum(b.get("status") in {"preparation_failed", "failed"} for b in visible)
        self.batches_button.setText(f"Batches ({len(visible)})" + (f" · {attention} review" if attention else ""))

    def show_batches(self, initial_id=None) -> None:
        from .batch_ui import BatchDialog
        try:
            from .batches import registered_stores
            store = self.get_batch_store()
            if not isinstance(initial_id, str):
                stores = registered_stores()
                if len(stores) > 1:
                    labels = [str(item.root.parent) for item in stores]
                    selected, accepted = QInputDialog.getItem(self, "Choose batch folder",
                        "Saved projects folder:", labels, labels.index(str(store.root.parent)) if str(store.root.parent) in labels else 0, False)
                    if not accepted:
                        return
                    store = stores[labels.index(selected)]
            dialog = BatchDialog(store, self.refresh_batch_count,
                                 initial_id if isinstance(initial_id, str) else None, self, settings=self.collect_settings())
            dialog.exec()
            dialog.deleteLater()
            self.refresh_batch_count()
            QTimer.singleShot(0, self.poll_batches)
        except (OSError, ValueError, RuntimeError) as error:
            QMessageBox.warning(self, "Cannot open batches", str(error))

    def stage_batch_member(self, settings: Settings, concept: str, seed: dict | None, oom_source: Path | None = None) -> None:
        from .batch_ui import BatchWorker, batch_label
        try:
            from .engine import fork_uses_snapshot
            settings.validate(require_folders=not fork_uses_snapshot(settings, seed))
            if not concept:
                raise ValueError("Enter a concept first.")
            if not settings.template_file:
                raise ValueError("Choose a workflow JSON from the workflow folder.")
            store = self.get_batch_store()
            if not settings.global_desktop:
                raise ValueError("Batch mode requires the managed Global RunPod setup. Select One-off for another connection.")
            eligible = [batch for batch in store.list()
                        if batch.get("status") in {"draft", "prepared", "preparation_failed", "failed", "paused"}]
            labels = ["New batch"] + [batch_label(batch) + f" · {batch['id'][:8]}" for batch in eligible]
            selected, accepted = QInputDialog.getItem(self, "Add to RunPod batch",
                "Choose the draft batch for this concept and its current settings:", labels, 0, False)
            if not accepted:
                return
            identity = None if selected == "New batch" else eligible[labels.index(selected) - 1]["id"]
            settings.save()
            def stage(_emit, _cancel, _detach):
                from .engine import create_run
                batch = store.create() if identity is None else store.get(identity)
                if oom_source is not None:
                    from .oom_transfer import create_cloud_copy
                    folder, _state = create_cloud_copy(oom_source, settings, concept)
                else:
                    folder, _state = create_run(settings, concept, seed)
                return store.add(batch["id"], folder)
            self.batch_stage_worker = BatchWorker(stage, self)
            self.batch_stage_worker.succeeded.connect(self.batch_staged)
            self.batch_stage_worker.failed.connect(lambda message: QMessageBox.warning(self, "Cannot add batch job", message))
            self.batch_stage_worker.finished.connect(self.batch_stage_finished)
            self.generate_button.setEnabled(False)
            self.status.setText("Saving concept and settings to batch…")
            self.batch_stage_worker.start()
        except (OSError, ValueError, RuntimeError) as error:
            QMessageBox.warning(self, "Cannot add batch job", str(error))

    def batch_staged(self, batch: dict) -> None:
        self.settings = self.collect_settings()
        self.staged_fork = None
        self.status.setText("Saved to batch. Review jobs and prepare before scheduling compute.")
        self.refresh_batch_count()
        self.show_batches(batch["id"])

    def batch_stage_finished(self) -> None:
        worker, self.batch_stage_worker = self.batch_stage_worker, None
        if worker:
            worker.deleteLater()
        self.generate_button.setEnabled(True)
        if self.closing:
            self.close()

    def poll_batches(self) -> None:
        # The cloud batch slot is independent of the local slot (local runs and schedules).
        if (self.closing or self.batch_worker is not None or self.batch_stage_worker is not None
                or self.cloud_worker is not None or self.override_worker is not None
                or (self.worker is not None and self.display_settings.cloud)
                or self.global_worker is not None or QApplication.activeModalWidget() is not None):
            return
        try:
            from .batches import registered_stores
            for store in registered_stores():
                store.recover_handoffs()
                for row in store.list():
                    if row.get("status") == "complete":
                        store.service_inbox(row["id"])  # Reopen during keep-warm, or reject.
            now_utc = datetime.now(UTC)
            due_batches = [(store, batch) for store in registered_stores()
                           for batch in store.list()
                           if batch.get("status") in {"armed", "waiting", "running"}
                           and (batch.get("status") == "running" or
                                (batch.get("next_check_at") and
                                 datetime.fromisoformat(batch["next_check_at"]) <= now_utc))]
            if not due_batches:
                return
            store, batch = min(due_batches, key=lambda item: item[1].get("next_check_at") or "")
            from .batch_ui import BatchWorker
            self.batch_worker = BatchWorker(lambda emit, cancel, detach: store.run_batch(batch["id"], emit, cancel, detach), self)
            self.batch_worker_folder = Path(batch["folder"])
            self.batch_line = "Dispatching saved RunPod batch…"
            self.batch_worker.event.connect(self.handle_batch_event)
            self.batch_worker.succeeded.connect(self.batch_dispatched)
            self.batch_worker.failed.connect(self.batch_failed)
            self.batch_worker.finished.connect(self.batch_dispatch_finished)
            self.update_jobs_status()
            self.batch_worker.start()
        except (OSError, ValueError, RuntimeError) as error:
            self.log.appendPlainText(f"Batch dispatch: {error}")

    def batch_dispatched(self, result) -> None:
        final_folder = None
        if isinstance(result, dict) and result.get("final_path"):
            self.show_result(result, switch_tab=False)
            final_folder = result.get("folder")
        elif isinstance(result, dict) and result.get("status") == "complete":
            for member in result.get("members", []):
                completed = member.get("result")
                if isinstance(completed, dict) and completed.get("final_path"):
                    self.show_result(completed, switch_tab=False)
                    final_folder = completed.get("folder")
            self.progress.setRange(0, 100)
            self.progress.setValue(100)
            message = f"Batch complete. Final project: {final_folder or result.get('folder', '')}"
            self.status.setText(message)
            self.log.appendPlainText(message)
        elif isinstance(result, dict) and result.get("error"):
            self.batch_failed(result["error"])
        elif isinstance(result, dict) and result.get("status") == "waiting":
            reason = result.get("waiting_reason") or "Waiting for an approved GPU"
            retry = result.get("next_check_at")
            message = f"Batch waiting: {reason}" + (f" · Next check {display_time(retry)}" if retry else "")
            self.status.setText(message)
            self.log.appendPlainText(message)
        self.refresh_batch_count()

    def handle_batch_event(self, event: dict) -> None:
        """Batch progress goes to its own status line and the log, never the local run's progress."""
        message, stage = str(event.get("message", "")), str(event.get("stage", ""))
        text = f"{stage}: {message}" if stage else message
        if text:
            self.batch_line = text[:200]
            self.log.appendPlainText(f"[batch] {text}")
        if isinstance(event.get("completed_result"), dict):
            self.show_result(event["completed_result"], switch_tab=False)
        self.update_jobs_status()

    def batch_failed(self, message: str) -> None:
        self.status.setText(f"Batch needs review: {message}")
        self.log.appendPlainText(f"Batch needs review: {message}")
        self.refresh_batch_count()

    def batch_dispatch_finished(self) -> None:
        worker, self.batch_worker = self.batch_worker, None
        self.batch_worker_folder = None
        self.batch_line = ""
        self.cancel_button.setEnabled(self.worker is not None)
        self.update_summary()
        if worker:
            worker.deleteLater()
        if self.closing:
            self.close()

    def show_schedules(self) -> None:
        dialog = ScheduleDialog(self.scheduler, self.poll_schedules, self, edit=self.edit_schedule,
                                repair=lambda record: self.open_repair(Path(record["folder"]), record["id"]))
        dialog.exec()
        dialog.deleteLater()

    def edit_schedule(self, record: dict) -> None:
        if record["status"] == "attention":
            self.open_saved_run_choice(Path(record["folder"]))
            return
        try:
            with self.scheduler.suspend(record["id"]):
                saved = read_state(Path(record["folder"]) / "state.json")
                expected = fingerprint(record)
                callback = lambda settings, concept, due: self.scheduler.edit(
                    record["id"], expected, settings, concept, due)
                dialog = ScheduleEditDialog(record, Settings(**saved["settings"]), callback, self)
                dialog.exec()
                self.refresh_schedule_count()
        except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
            QMessageBox.warning(self, "Cannot edit schedule", str(error))
        finally:
            QTimer.singleShot(0, self.poll_schedules)

    def poll_schedules(self) -> None:
        if self.closing:
            return
        try:
            due = self.scheduler.tick()
            self.refresh_schedule_count()
            if not due:
                return
            for record in due[1:]:
                self.scheduler.note(record["id"], "Waiting for earlier scheduled requests")
            record = due[0]
            modal = QApplication.activeModalWidget()
            if modal is not None and not isinstance(modal, ScheduleDialog):
                self.scheduler.defer(record["id"], "Waiting for the open dialog to close")
                self.refresh_schedule_count()
                return
            if (self.worker is not None or self.batch_stage_worker is not None
                    or self.cloud_worker is not None or self.override_worker is not None
                    or self.preflight_worker is not None):
                self.scheduler.defer(record["id"], "Waiting for the current simpleGen operation to finish")
                self.refresh_schedule_count()
                return
            if self.schedule_probe is not None:
                return
            if not record.get("note") or record["note"] in (
                "Waiting for earlier scheduled requests", "Waiting for the open dialog to close",
                "Waiting for the current simpleGen operation to finish",
            ):
                self.scheduler.note(record["id"], "Waiting — checking local ComfyUI availability")
            self.refresh_schedule_count()
            state = read_state(Path(record["folder"]) / "state.json")
            settings = Settings(**state["settings"])
            if record["started"]:
                # Explicit Resume must reconnect to its own in-flight prompt.
                self.schedule_ready(record["id"], True)
                return
            folder = Path(record["folder"])

            def probe():
                if not local_queue_empty(settings):
                    return {"idle": False}
                # Deterministic re-check of the saved snapshot before any agent work.
                return {"idle": True, "preflight": run_saved_preflight(folder)}
            worker = OperationWorker(probe, self)
            self.schedule_probe = worker
            expected = fingerprint(self.scheduler.get(record["id"]))
            worker.succeeded.connect(lambda result: self.schedule_ready(
                record["id"], result["idle"], expected, result.get("preflight")))
            worker.failed.connect(lambda message: self.schedule_probe_failed(record["id"], message, expected))
            worker.finished.connect(self.schedule_probe_finished)
            worker.start()
        except (OSError, ValueError, RuntimeError) as error:
            self.log.appendPlainText(f"Scheduler: {error}")

    def schedule_probe_failed(self, identity: str, message: str, expected: str | None = None) -> None:
        try:
            if expected is not None and fingerprint(self.scheduler.get(identity)) != expected:
                return
            self.scheduler.defer(identity, f"Waiting for ComfyUI: {message}")
            self.refresh_schedule_count()
        except (OSError, ValueError, RuntimeError) as error:
            self.log.appendPlainText(f"Scheduler: {error}")

    def schedule_ready(self, identity: str, idle: bool, expected: str | None = None,
                       preflight: dict | None = None) -> None:
        if (self.closing or self.worker is not None or self.cloud_worker is not None
                or self.override_worker is not None or self.preflight_worker is not None):
            return
        modal = QApplication.activeModalWidget()
        if modal is not None and not isinstance(modal, ScheduleDialog):
            return
        try:
            if expected is not None and fingerprint(self.scheduler.get(identity)) != expected:
                return
            # Detect sleep or a reschedule/removal while the availability read was running.
            self.scheduler.tick()
            if expected is not None and fingerprint(self.scheduler.get(identity)) != expected:
                return
            if not idle:
                self.scheduler.defer(identity, "Waiting for local ComfyUI queue to clear")
                self.refresh_schedule_count()
                return
            folder = self.scheduler.claim(identity, preflight)
            if folder is None:
                self.refresh_schedule_count()
                return
            self.refresh_schedule_count()
            self.active_schedule = identity
            self.start_generation(folder)
            if self.worker is None:
                self.finish_schedule(False)
        except (OSError, ValueError, RuntimeError) as error:
            self.log.appendPlainText(f"Scheduler: {error}")

    def schedule_probe_finished(self) -> None:
        worker, self.schedule_probe = self.schedule_probe, None
        if worker:
            worker.deleteLater()
        if self.closing and self.worker is None and self.cloud_worker is None:
            self.close()

    def finish_schedule(self, success: bool) -> None:
        if self.active_schedule:
            try:
                self.scheduler.finish(self.active_schedule, success)
                self.refresh_schedule_count()
            except (OSError, ValueError, RuntimeError) as error:
                self.log.appendPlainText(f"Scheduler: {error}")
            self.active_schedule = None

    def start_generation(self, resume_folder: Path | None = None, *, from_override=False) -> None:
        if (self.batch_stage_worker is not None or self.preflight_worker is not None
                or (self.override_worker is not None and not from_override)):
            return
        settings = self.collect_settings()
        concept = self.concept.toPlainText().strip()
        seed = self.staged_fork if resume_folder is None else None
        if (resume_folder is None and settings.managed_runpod and
                self.batch_mode.currentData() == "batch"):
            self.stage_batch_member(settings, concept, seed)  # Drafting a batch member needs no slot.
            return
        scheduling = resume_folder is None and not settings.cloud and self.schedule_enabled.isChecked()
        if self.worker is not None and not scheduling:
            # One local-slot run at a time; drafts and schedules stay available.
            self.status.setText("A run is already in progress. Schedule this draft, or use Queue locally.")
            return
        if self.batch_worker is not None and (settings.cloud if resume_folder is None else False):
            QMessageBox.warning(self, "Compute is busy", "A RunPod batch is running. Add this concept to it "
                                "(Add to running batch…), or wait before starting a separate cloud run.")
            return
        try:
            if resume_folder is None:
                from .engine import fork_uses_snapshot
                seeded = fork_uses_snapshot(settings, seed)
                settings.validate(require_folders=not seeded)
                if seeded and (not Path(settings.project_folder).expanduser().is_dir()
                               or not settings.output_folder.strip()):
                    raise ValueError("Choose an existing project folder and an output folder for the new run.")
                if not concept:
                    raise ValueError("Enter a concept first.")
                if not settings.template_file:
                    raise ValueError("Choose a workflow JSON from the workflow folder.")
                if self.schedule_enabled.isChecked():
                    selected_time(self.schedule_time)  # Reject an invalid time before checking.
                snapshot = Path(seed["folder"]) / "template.original.json" if seeded else None
                self.preflight_then(settings, snapshot, lambda result, override: self.after_preflight(
                    settings, concept, seed, result, override))
                return
            else:
                state = read_state(resume_folder / "state.json")
                if state.get("sequence_parent") == "..":
                    resume_folder = resume_folder.parent
                    state = read_state(resume_folder / "state.json")
                settings = Settings(**state["settings"])
                settings.validate(require_folders=False)
                if self.active_schedule:
                    concept = state["concept"]
        except (ValueError, OSError, KeyError, TypeError) as exc:
            QMessageBox.warning(self, "Cannot start", str(exc))
            return
        self.launch_generation(settings, concept, resume_folder, seed)

    def preflight_then(self, settings: Settings, template_path: Path | None, proceed, on_fail=None,
                       compiled: bool = True, saved: Path | None = None) -> None:
        """Run deterministic preflight off the UI thread; no agent work starts before it passes.

        ``compiled`` is False only for saved runs from before schema 5 (no compiler check).
        ``saved`` is an accepted run folder: with a complete reference cache it needs no linked mount.
        """
        def check():
            from .refcache import complete
            extra = {} if compiled else {"compiled": False}
            if saved is not None and complete(saved):
                extra["cached_references"] = True
            return run_preflight(settings, template_path=template_path, **extra)
        worker = OperationWorker(check, self)
        self.preflight_worker = worker
        self.generate_button.setEnabled(False)
        self.status.setText("Preflight: checking workflow, models, folders and tools…")
        worker.succeeded.connect(lambda result: self.preflight_done(result, proceed, on_fail))
        worker.failed.connect(lambda message: self.preflight_done(
            {"status": UNVERIFIED, "checks": [{"id": "preflight", "status": UNVERIFIED,
                                                "message": f"Preflight could not run: {message}", "details": {}}]},
            proceed, on_fail))
        worker.finished.connect(self.preflight_finished)
        worker.start()

    def preflight_finished(self) -> None:
        worker, self.preflight_worker = self.preflight_worker, None
        if worker:
            worker.deleteLater()
        self.generate_button.setEnabled(self.worker is None)

    def preflight_done(self, result: dict, proceed, on_fail=None) -> None:
        self.generate_button.setEnabled(self.worker is None)
        status = result.get("status")
        for check in result.get("checks", []):
            if check.get("id") == "h3scene_envelope":
                self.log.appendPlainText("H3 Scene Chain preflight: " + check["message"])
            if check.get("id") == "upscale_service":
                self.log.appendPlainText("4K60: " + check["message"])
        line = summary(result)
        self.status.setText(f"Preflight {status}: {line}" if status != "pass" else "Preflight passed.")
        if self.closing:
            return  # A running local job no longer blocks schedules; launching re-checks the slot.
        if status == FAIL:
            if on_fail is not None:
                on_fail(result)
            else:
                self.preflight_dialog(result, confirm=False)
            return
        override = status == UNVERIFIED
        if override and not self.preflight_dialog(result, confirm=True):
            return
        proceed(result, override)

    def preflight_dialog(self, result: dict, *, confirm: bool) -> bool:
        """Blocking dialog for fail; explicit confirmation for unverified."""
        lines = []
        for check in result.get("checks", []):
            if check["status"] == "pass":
                continue
            lines.append(f"[{check['status'].upper()}] {check['id']}: {check['message']}")
            for model in check.get("details", {}).get("missing_models", []):
                lines.append(f"  {model['class_type']} node {model['node_id']} {model['input']} "
                             f"needs {model['model']}; available:")
                lines.extend(f"    - {choice}" for choice in model["available"])
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Question if confirm else QMessageBox.Icon.Warning)
        box.setWindowTitle("Preflight could not verify" if confirm else "Preflight failed")
        box.setText(summary(result) + ("\n\nContinue without verification?" if confirm else ""))
        box.setDetailedText("\n".join(lines))
        if confirm:
            box.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel)
            box.setDefaultButton(QMessageBox.StandardButton.Cancel)
        return box.exec() == QMessageBox.StandardButton.Yes

    def after_preflight(self, settings: Settings, concept: str, seed: dict | None, result: dict,
                        override: bool) -> None:
        try:
            if self.schedule_enabled.isChecked():
                due = selected_time(self.schedule_time)
                settings.save()
                record = self.scheduler.add(settings, concept, due, fork_seed=seed,
                                            preflight=result, override_unverified=override)
                self.settings = settings
                self.staged_fork = None
                self.schedule_enabled.setChecked(False)
                self.refresh_schedule_count()
                self.status.setText(f"Scheduled locally for {display_time(record['due_at'])}. Keep this computer awake and simpleGen open.")
                self.log.appendPlainText(f"Scheduled: {record['folder']}")
                return
            settings.save()
        except (ValueError, OSError, KeyError, TypeError) as exc:
            QMessageBox.warning(self, "Cannot start", str(exc))
            return
        if self.worker is not None:
            self.status.setText("A run is already in progress. Schedule this draft, or use Queue locally.")
            return
        self.launch_generation(settings, concept, None, seed)

    def launch_generation(self, settings: Settings, concept: str, resume_folder: Path | None,
                          seed: dict | None) -> None:
        if resume_folder is None:
            self.settings = settings
        else:
            self.settings = self.collect_settings()
        self.display_settings = settings
        self.cloud_state = None
        self.cloud_timer.start()
        self.update_summary()
        self.current_folder = resume_folder
        self.repair_button.hide()
        self.salvage_button.hide()
        self.report_button.setEnabled(resume_folder is not None)
        self.log.clear()
        self.status.setText("Resuming saved run…" if resume_folder else "Preparing your concept…")
        self.progress.setRange(0, 0)
        self.set_running(True)
        self.tabs.setCurrentIndex(0)
        self.worker = (GenerationWorker(settings, concept, resume_folder, self, fork_seed=seed)
                       if seed is not None else GenerationWorker(settings, concept, resume_folder, self))
        self.worker.event.connect(self.handle_event)
        self.worker.succeeded.connect(self.generation_succeeded)
        self.worker.failed.connect(self.generation_failed)
        self.worker.resource_choice.connect(self.choose_gpu)
        self.worker.finished.connect(self.worker_finished)
        self.worker.start()
        self.update_jobs_status()
        if seed is not None:
            self.active_fork_seed = seed

    def queue_locally(self) -> None:
        """One-click local schedule due now; it waits for the local ComfyUI queue (Phase 1 preflight applies)."""
        from datetime import timedelta
        settings = self.collect_settings()
        concept = self.concept.toPlainText().strip()
        try:
            if settings.cloud:
                raise ValueError("Queue locally needs the Local destination.")
            if not concept:
                raise ValueError("Enter a concept first.")
            settings.validate()
        except ValueError as error:
            QMessageBox.warning(self, "Cannot queue locally", str(error))
            return

        def queued(result, override):
            try:
                record = self.scheduler.add(settings, concept, datetime.now(UTC) + timedelta(seconds=1),
                                            preflight=result, override_unverified=override)
            except (ValueError, OSError) as error:
                QMessageBox.warning(self, "Cannot queue locally", str(error))
                return
            self.refresh_schedule_count()
            self.status.setText("Queued locally; it starts when the local ComfyUI queue is free.")
            self.log.appendPlainText(f"Queued locally: {record['folder']}")
        self.preflight_then(settings, None, queued)

    def running_batches(self) -> list:
        """Batches that can take an addition: running/armed/waiting, or complete with a warm pod."""
        from .batches import registered_stores, reopen_refusal
        found = []
        for store in registered_stores():
            for row in store.list():
                if row.get("status") in {"running", "armed", "waiting"} or reopen_refusal(row) is None:
                    found.append((store, row))
        return found

    def add_draft_to_running_batch(self, *_args, choose=None, confirm=None) -> None:
        """Prepare the draft off the UI thread (outside the batch lock), show the projected cost, then request it."""
        from .batch_ui import BatchWorker, batch_label
        if self.addition_worker is not None:
            return
        settings, concept = self.collect_settings(), self.concept.toPlainText().strip()
        try:
            if not settings.managed_runpod:
                raise ValueError("Choose the RunPod (managed) destination to add this concept to a batch.")
            if not concept:
                raise ValueError("Enter a concept first.")
            candidates = self.running_batches()
            if not candidates:
                raise ValueError("No running or warm RunPod batch can take an addition. Start a batch instead.")
        except (ValueError, OSError) as error:
            QMessageBox.warning(self, "Cannot add to a running batch", str(error))
            return
        if choose is not None:
            store, batch = choose(candidates)
        elif len(candidates) == 1:
            store, batch = candidates[0]
        else:
            from PySide6.QtWidgets import QInputDialog
            labels = [batch_label(row) for _, row in candidates]
            label, ok = QInputDialog.getItem(self, "Add to running batch", "Batch", labels, 0, False)
            if not ok:
                return
            store, batch = candidates[labels.index(label)]
        from .additions import prepare_addition
        worker = BatchWorker(lambda emit, cancel, detach: prepare_addition(
            store, batch["id"], concept, settings.quantity, emit, cancel, detach, base=settings), self)
        self.addition_worker = worker
        worker.event.connect(lambda event: self.log.appendPlainText(f"[addition] {event.get('message', '')}"))
        worker.succeeded.connect(lambda prepared: self.addition_prepared(store, prepared, confirm))
        worker.failed.connect(lambda message: QMessageBox.warning(self, "Cannot add to a running batch", message))
        worker.finished.connect(self.addition_finished)
        self.status.setText("Preparing the addition; the batch keeps rendering.")
        worker.start()

    def addition_prepared(self, store, prepared: dict, confirm=None) -> None:
        from .additions import submit_addition
        from .cost_safety import confirmation_text
        text = confirmation_text(prepared["projection"])
        ask = confirm or (lambda message: QMessageBox.question(
            self, "Add to running batch?", message + "\n\nQueue it on the batch's GPU?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No) == QMessageBox.StandardButton.Yes)
        if not ask(text):
            self.status.setText("Addition prepared but not queued; its run folder is kept.")
            return
        try:
            submit_addition(store, prepared["batch_id"], Path(prepared["folder"]))
        except (OSError, ValueError, RuntimeError) as error:
            QMessageBox.warning(self, "Cannot add to a running batch", str(error))
            return
        self.status.setText("Addition queued; the batch accepts it at its next step.")
        self.log.appendPlainText(f"Addition requested: {prepared['folder']}")

    def addition_finished(self) -> None:
        worker, self.addition_worker = self.addition_worker, None
        if worker:
            worker.deleteLater()

    def resume_generation(self) -> None:
        # A running cloud batch has its own slot; resuming a saved run uses the local slot.
        folder = QFileDialog.getExistingDirectory(
            self, "Choose saved run", self.output_folder.text()
        )
        if folder:
            self.open_saved_run_choice(Path(folder))

    def open_saved_run_choice(self, folder: Path) -> None:
        if self.worker is not None or self.override_worker is not None:
            return
        try:
            folder = folder.expanduser().resolve()
            state = read_state(folder / "state.json")
            if state.get("sequence_parent") == "..":
                folder = folder.parent
                state = read_state(folder / "state.json")
            dialog = SavedRunDialog(state, folder, self)
            if dialog.exec() != QDialog.DialogCode.Accepted:
                return
            if dialog.choice in ("scene_approve", "scene_deny") or (dialog.choice == "new" and dialog.scene_pending):
                action = {"scene_approve": "approve", "scene_deny": "deny", "new": "edit"}[dialog.choice]
                self.decide_scene_overflow(folder, action,
                                         dialog.concept.toPlainText().strip() if action == "edit" else None)
            elif dialog.choice == "resume":
                self.resume_saved(folder, state)
            elif dialog.choice == "repair":
                self.open_repair(folder)
            elif dialog.choice == "salvage":
                self.salvage_saved(folder)
            elif dialog.choice == "cache":
                self.cache_saved_references(folder)
            elif dialog.choice == "runpod":
                self.add_oom_to_batch(folder, dialog.concept.toPlainText().strip())
            elif dialog.choice == "new":
                self.stage_saved_run(folder, state, dialog.concept.toPlainText().strip())
        except (OSError, ValueError, KeyError, TypeError) as error:
            QMessageBox.warning(self, "Cannot open saved run", str(error))

    def decide_scene_overflow(self, folder: Path, action: str, concept: str | None) -> None:
        """Publish a review decision off the UI thread, then dispatch its linked run."""
        from .h3scene_review import decide
        if self.salvage_worker is not None:
            return

        def operation():
            revision, child = decide(folder, action, concept)
            return {"folder": str(revision), "state": child}

        def done(result):
            revision, child = Path(result["folder"]), result["state"]
            if self.closing:
                return
            if action == "approve":
                self.resume_saved(revision, child)
            elif action == "edit":
                self.stage_saved_run(revision, child, child["concept"])
            else:
                self.status.setText(f"H3 Scene Chain plan denied. Decision saved: {revision}")

        self.status.setText("Saving H3 Scene Chain review decision…")
        worker = OperationWorker(operation, self)
        self.salvage_worker = worker
        worker.succeeded.connect(done)
        worker.failed.connect(lambda message: QMessageBox.warning(self, "Cannot save H3 Scene Chain decision", message))
        worker.finished.connect(self.salvage_finished)
        worker.start()

    def resume_saved(self, folder: Path, state: dict) -> None:
        """Resume unchanged. Work never submitted is preflighted first; reconnecting is not gated."""
        from .repair import submitted
        if state.get("batch_parent") or submitted(folder, state):
            self.start_generation(folder)
            return
        if self.preflight_worker is not None:
            return
        settings = Settings(**state["settings"])
        snapshot = folder / "template.original.json"  # Absent only before the first preparation.
        self.preflight_then(settings, snapshot if snapshot.is_file() else None,
                            lambda _result, _override: self.start_generation(folder),
                            on_fail=lambda result: self.offer_repair(folder, result),
                            compiled=state.get("schema_version", 1) >= 5, saved=folder)

    def offer_repair(self, folder: Path, result: dict) -> None:
        """Resume is blocked by a failing preflight; offer Repair… when the run is eligible."""
        from .repair import refusal
        try:
            state = read_state(folder / "state.json")
            reason = refusal(folder, state)
        except (OSError, ValueError) as error:
            reason = str(error)
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Warning)
        box.setWindowTitle("Resume blocked by preflight")
        box.setText(summary(result) + ("" if reason is None else f"\n\nRepair is not available: {reason}"))
        repair_button = box.addButton("Repair…", QMessageBox.ButtonRole.AcceptRole) if reason is None else None
        box.addButton(QMessageBox.StandardButton.Close)
        box.exec()
        if repair_button is not None and box.clickedButton() is repair_button:
            self.open_repair(folder)

    def salvage_saved(self, folder: Path) -> None:
        """Rebuild a rejected attempt's content into a schema 5 revision off the UI thread."""
        from .repair import salvage
        if self.worker is not None or self.override_worker is not None or self.salvage_worker is not None:
            return
        folder = Path(folder).expanduser().resolve()
        self.status.setText("Rebuilding from saved agent work (no agent pass)…")
        worker = OperationWorker(lambda: salvage(folder), self)
        self.salvage_worker = worker
        worker.succeeded.connect(self.salvage_done)
        worker.failed.connect(self.salvage_failed)
        worker.finished.connect(self.salvage_finished)
        worker.start()

    def cache_saved_references(self, folder: Path) -> None:
        """Snapshot an older accepted run's references off the UI thread (Phase 6)."""
        from .refcache import cache_saved
        if self.salvage_worker is not None:
            return
        self.status.setText("Caching references locally…")
        worker = OperationWorker(lambda: cache_saved(folder), self)
        self.salvage_worker = worker
        worker.succeeded.connect(lambda manifest: self.status.setText(
            f"Cached {len(manifest['references'])} reference(s); this run no longer needs the linked folder."))
        worker.failed.connect(lambda message: (self.status.setText(f"Cannot cache references: {message}"),
                                               QMessageBox.warning(self, "Cannot cache references", message)))
        worker.finished.connect(self.salvage_finished)
        worker.start()

    def salvage_done(self, result: dict) -> None:
        """Preflight the rebuilt revision; a failure opens Repair/Retarget for it at once."""
        revision = Path(result["folder"])
        self.log.appendPlainText(f"Rebuilt revision saved: {revision}")
        self.status.setText(f"Rebuilt from saved agent work: {revision.name}. Checking it with preflight…")
        if self.closing or self.preflight_worker is not None:
            return
        try:
            state = read_state(revision / "state.json")
        except (OSError, ValueError) as error:
            QMessageBox.warning(self, "Cannot open rebuilt run", str(error))
            return
        self.preflight_then(Settings(**state["settings"]), revision / "template.original.json",
                            lambda _result, _override: self.open_saved_run_choice(revision),
                            on_fail=lambda _result: self.open_repair(revision), saved=revision)

    def salvage_failed(self, message: str) -> None:
        self.status.setText(f"Cannot rebuild: {message}")
        QMessageBox.warning(self, "Cannot rebuild from saved agent work", message)

    def salvage_finished(self) -> None:
        worker, self.salvage_worker = self.salvage_worker, None
        if worker:
            worker.deleteLater()

    def _attention_record(self, folder: Path) -> str | None:
        try:
            folder = Path(folder).resolve()
            return next((r["id"] for r in self.scheduler.records()
                         if r["status"] == "attention" and Path(r["folder"]).resolve() == folder), None)
        except (OSError, ValueError):
            return None

    def open_repair(self, folder: Path, schedule_id: str | None = None) -> None:
        """Repair model selections as a linked revision, then Run now or Schedule…."""
        from .repair_ui import RepairDialog
        if self.worker is not None or self.override_worker is not None or self.preflight_worker is not None:
            return
        folder = Path(folder).expanduser().resolve()
        schedule_id = schedule_id or self._attention_record(folder)
        existing = self._unlinked_revision(folder) if schedule_id else None
        if existing is not None:
            # A revision from an earlier repair was never linked: offer it, don't duplicate it.
            result, choice = existing, self.choose_existing_revision(Path(existing["folder"]))
            if choice is None:
                return
        else:
            dialog = RepairDialog(folder, self)
            dialog.exec()
            result, choice = dialog.result, dialog.choice
            dialog.deleteLater()
        if not result:
            return
        revision = Path(result["folder"])
        self.log.appendPlainText(f"Repaired revision {'linked' if existing else 'saved'}: {revision}")
        try:
            if schedule_id:
                self.scheduler.mark_repaired(schedule_id, revision, result["run_id"])
                self.refresh_schedule_count()
            if choice == "run":
                self.start_generation(revision)
            elif choice == "schedule":
                self.schedule_revision(revision, result, schedule_id)
        except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
            QMessageBox.warning(self, "Repaired revision saved", f"{error}\n\nRevision: {revision}")

    def _unlinked_revision(self, folder: Path) -> dict | None:
        from .repair import unlinked_revision
        try:
            linked = {r.get("repaired_to", {}).get("folder") for r in self.scheduler.records()}
            linked |= {r["folder"] for r in self.scheduler.records() if r["status"] != "removed"}
            return unlinked_revision(folder, linked)
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def choose_existing_revision(self, revision: Path) -> str | None:
        """Return "run", "schedule", "link" or None (cancel) for an earlier unlinked revision."""
        box = QMessageBox(self)
        box.setWindowTitle("Repaired revision already exists")
        box.setText(f"This request was already repaired as {revision.name}, but the revision was not linked.\n"
                    "Use that revision instead of repairing again?")
        buttons = {box.addButton("Run now", QMessageBox.ButtonRole.AcceptRole): "run",
                   box.addButton("Schedule…", QMessageBox.ButtonRole.AcceptRole): "schedule",
                   box.addButton("Link only", QMessageBox.ButtonRole.AcceptRole): "link"}
        box.addButton(QMessageBox.StandardButton.Cancel)
        box.exec()
        return buttons.get(box.clickedButton())

    def schedule_revision(self, revision: Path, result: dict, schedule_id: str | None) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle("Schedule repaired run — local time")
        layout = QVBoxLayout(dialog)
        field = date_field()
        layout.addWidget(field)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        accepted = dialog.exec() == QDialog.DialogCode.Accepted
        dialog.deleteLater()
        if not accepted:
            return
        preflight = result["preflight"]
        record = self.scheduler.add_prepared(revision, selected_time(field), preflight=preflight,
                                             override_unverified=preflight.get("status") == UNVERIFIED,
                                             source_record=schedule_id)
        self.refresh_schedule_count()
        self.status.setText(f"Repaired run scheduled locally for {display_time(record['due_at'])}.")

    def add_oom_to_batch(self, folder: Path | None, concept: str | None = None) -> None:
        if folder is None or any((self.worker, self.batch_worker, self.batch_stage_worker, self.override_worker)):
            return
        try:
            from .batch_ui import MemberEditDialog
            from .oom_transfer import cloud_settings, local_oom
            state = read_state(folder / "state.json")
            if not local_oom(folder, state):
                raise ValueError("Choose a failed local run that ran out of memory.")
            config = cloud_settings(state, self.collect_settings())
            dialog = MemberEditDialog({**state, "settings": config.public_dict(),
                                       "concept": concept if concept is not None else state["concept"]}, self)
            dialog.setWindowTitle("Copy local OOM run to RunPod batch")
            note = selectable(f"Cloud model: {config.runpod_default_model}\nCLIP: {config.runpod_default_clip}\nA new cloud workflow and agent preparation will be created. The local run stays saved.")
            dialog.layout().insertWidget(0, note)
            if dialog.exec() != QDialog.DialogCode.Accepted:
                return
            concept, config = dialog.values()
            self.settings = config
            self.display_settings = config
            self.staged_fork = None
            self.current_folder = None
            self.cloud_state = None
            self.project_folder.setText(config.project_folder)
            self.template_folder.setText(config.template_folder)
            self.output_folder.setText(config.output_folder)
            self.refresh_templates()
            self.template.setCurrentIndex(self.template.findData(config.template_file))
            self.steps.setValue(config.steps or 25)
            self.megapixels.setValue(config.megapixels)
            self.duration.setValue(config.duration)
            self.quantity.setValue(config.quantity)
            self.upscale.setChecked(config.upscale)
            self.show_upscale_method(config.upscale_preset)
            self.update_upscale_toggle()
            self.aspect_ratio.setCurrentIndex(self.aspect_ratio.findData(config.aspect_ratio or "16:9"))
            self.concept.setPlainText(concept)
            self.batch_mode.setCurrentIndex(self.batch_mode.findData("batch"))
            self.update_summary()
            self.stage_batch_member(config, concept, None, oom_source=folder)
        except (OSError, ValueError, KeyError, TypeError) as error:
            QMessageBox.warning(self, "Cannot copy OOM run", str(error))

    def stage_saved_run(self, folder: Path, state: dict, concept: str) -> None:
        from .editing import digest, read_json
        from .engine import _template

        snapshot = folder / "template.original.json"
        settings = Settings(**state["settings"])
        # A new project inherits creative settings, not historical deployment
        # configuration. Resuming the original run keeps its saved connection.
        current = self.collect_settings()
        settings = replace(settings, **{
            key: getattr(current, key) for key in (
                "runpod_config_file", "runpod_credentials_file", "runpod_schema_file",
                "runpod_default_workflow", "runpod_default_model", "runpod_default_clip",
            )
        })
        if snapshot.is_file():
            read_json(snapshot)
            sha = digest(snapshot)
            if sha != state.get("template_sha256"):
                raise ValueError("Saved workflow snapshot changed; this run cannot be used as a source.")
        else:
            # Historical schema-1 projects did not retain a native snapshot.
            read_json(_template(settings))
            sha = None
        if settings.managed_runpod:
            settings = replace(settings, comfy_url="")
        seed = {"folder": str(folder), "run_id": state["run_id"],
                "template_sha256": sha, "template_file": settings.template_file,
                "template_folder": settings.template_folder}
        self.settings = settings
        self.display_settings = settings
        self.staged_fork = seed
        self.cloud_state = None
        self.current_folder = None
        self.report_button.setEnabled(False)
        self.project_folder.setText(settings.project_folder)
        self.template_folder.setText(settings.template_folder)
        self.output_folder.setText(settings.output_folder)
        self.refresh_templates()
        index = self.template.findData(settings.template_file)
        if index >= 0:
            self.template.setCurrentIndex(index)
        else:
            self.template.setCurrentIndex(-1)
        self.steps.setValue(settings.steps or 25)
        self.megapixels.setValue(settings.megapixels)
        self.duration.setValue(settings.duration)
        self.aspect_ratio.setCurrentIndex(self.aspect_ratio.findData(settings.aspect_ratio or "16:9"))
        self.quantity.setValue(settings.quantity)
        self.upscale.setChecked(settings.upscale)
        self.show_upscale_method(settings.upscale_preset)
        self.update_upscale_toggle()
        self.endpoint.setText(settings.comfy_url)
        self.concept.setPlainText(concept)
        self.update_summary()
        self.tabs.setCurrentIndex(0)
        suffix = (" The historical run has no saved workflow snapshot; its currently linked workflow will be used."
                  if sha is None else "")
        self.status.setText("New run staged from saved project. Review the concept and settings, then click Generate; "
                            "new agent work will start." + suffix)

    def generate_anyway(self) -> None:
        if self.worker is not None or self.batch_worker is not None or self.override_worker is not None:
            return
        folder = QFileDialog.getExistingDirectory(self, "Choose saved run with a response-format rejection",
                                                  self.output_folder.text())
        if not folder:
            return
        project = Path(folder)
        from .editing import inspect_saved_response
        self._override_operation(lambda: inspect_saved_response(project),
                                 lambda preview: self._confirm_override(project, preview))

    def _override_operation(self, operation, succeeded) -> None:
        worker = OperationWorker(operation, self)
        self.override_worker = worker
        self.anyway_button.setEnabled(False)
        self.generate_button.setEnabled(False)
        self.resume_button.setEnabled(False)
        worker.succeeded.connect(succeeded)
        worker.failed.connect(lambda message: QMessageBox.warning(self, "Cannot generate anyway", message))
        worker.finished.connect(lambda w=worker: self._override_finished(w))
        worker.start()

    def _override_finished(self, worker):
        if self.override_worker is worker:
            self.override_worker = None
            self.anyway_button.setEnabled(self.worker is None)
            self.generate_button.setEnabled(self.worker is None)
            self.resume_button.setEnabled(self.worker is None)
        worker.deleteLater()
        if self.closing and self.override_worker is None:
            QTimer.singleShot(0, self.close)

    def _confirm_override(self, project: Path, preview: dict) -> None:
        if self.closing:
            return
        message = (f"Saved run: {project.name}\n\nRejected response: {preview['reason']}\n"
                   f"Extra fields to ignore: {', '.join(preview['extra_fields'])}\n\n"
                   "The saved workflow, model, inputs, settings and references will be checked again. "
                   "No agent call is repeated. Cloud cost approval still applies. Record this override and resume?")
        if QMessageBox.question(self, "Generate anyway?", message) != QMessageBox.StandardButton.Yes:
            return
        from .editing import request_saved_response_override
        self._override_operation(lambda: request_saved_response_override(project, preview["fingerprint"]),
                                 lambda _: self.start_generation(project, from_override=True))

    def cancel_batch(self) -> None:
        if self.batch_worker is None or not self.confirm_shutdown(batch=True):
            return
        self.batch_worker.cancel_event.set()
        self.batch_line = "Shutting down batch compute. Undownloaded GPU outputs may be lost."
        self.update_jobs_status()

    def cancel_generation(self) -> None:
        """Cancel the local-slot run only. A running batch has its own explicit Shut down batch… control."""
        if self.worker is not None:
            if self.display_settings.managed_runpod and not self.confirm_shutdown():
                return
            self.worker.cancel_event.set()
            self.cancel_button.setEnabled(False)
            self.status.setText(
                "Shutting down managed compute. Undownloaded GPU outputs will be lost."
                if self.display_settings.managed_runpod else "Cancelling… Waiting for the current operation to stop safely."
            )

    def confirm_shutdown(self, batch: bool = False) -> bool:
        from .global_session import is_global
        global_mode = True if batch else is_global(self.display_settings)
        return QMessageBox.question(
            self, "Shut down compute?",
            "This deliberately shuts down this session's compute, even before downloads finish. "
            + ("Undownloaded outputs are on disposable GPU storage and will be lost."
               if global_mode else "Files on the network volume are retained."),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        ) == QMessageBox.StandardButton.Yes

    def choose_gpu(self, request: dict) -> None:
        worker = self.worker
        if worker is None or worker.cancel_event.is_set() or worker.detach_event.is_set():
            request["ready"].set()
            return
        self.choice_dialog = GPUChoiceDialog(request["options"], self)
        timer = QTimer(self.choice_dialog)
        timer.timeout.connect(lambda: self.choice_dialog.reject() if self.choice_dialog is not None and
                              (worker.cancel_event.is_set() or worker.detach_event.is_set()) else None)
        timer.start(100)
        try:
            if self.choice_dialog.exec() == QDialog.DialogCode.Accepted:
                request["choice"] = self.choice_dialog.choice()
        finally:
            timer.stop()
            self.choice_dialog.deleteLater()
            self.choice_dialog = None
            request["ready"].set()

    def cloud_action(self, action: str) -> None:
        if self.cloud_worker is not None:
            return
        if action == "cancel" and self.batch_worker is not None:
            self.cancel_batch()
            return
        if self.batch_worker is not None:
            return
        if action == "cancel" and self.worker is not None:
            self.cancel_generation()
            return
        if self.current_folder is None:
            folder = QFileDialog.getExistingDirectory(self, "Choose saved RunPod project", self.output_folder.text())
            if not folder:
                return
            try:
                state = read_state(Path(folder) / "state.json")
                if state.get("batch_parent"):
                    from .batches import registered_stores
                    owner = next((store.get(state["batch_parent"]) for store in registered_stores()
                                  if (store.root / state["batch_parent"] / "state.json").is_file()), None)
                    if owner is None:
                        raise ValueError("The owning batch could not be found.")
                    folder = owner["folder"]
                    state = read_state(Path(folder) / "state.json")
                config = Settings(**state["settings"])
                if not config.managed_runpod:
                    raise ValueError("This project does not use a managed RunPod session.")
                self.current_folder = Path(folder)
                self.display_settings = config
                self.cloud_state = None
                self.cloud_timer.start()
                self.update_summary()
                self.report_button.setEnabled(True)
            except (OSError, ValueError, TypeError, KeyError) as exc:
                QMessageBox.warning(self, "Cannot open cloud session", str(exc))
                return
        if action == "cancel" and not self.confirm_shutdown():
            return
        from .engine import cloud_action

        folder = self.current_folder
        self.cloud_worker = OperationWorker(lambda: cloud_action(folder, action), self)
        self.cloud_worker.succeeded.connect(
            lambda state: self.show_cloud_status(state) if self.current_folder == folder else None
        )
        self.cloud_worker.failed.connect(
            lambda message: self.cloud_action_failed(message) if self.current_folder == folder else None
        )
        self.cloud_worker.finished.connect(self.cloud_action_finished)
        self.cloud_status_button.setEnabled(False)
        self.cloud_shutdown_button.setEnabled(False)
        self.cloud_worker.start()

    def poll_cloud_status(self) -> None:
        if (not self.closing and self.display_settings.managed_runpod and self.current_folder is not None
                and self.worker is None and self.batch_worker is None and self.cloud_worker is None
                and not (self.cloud_state or {}).get("billing_confirmed_stopped")):
            self.cloud_action("status")

    def poll_registered_global_sessions(self) -> None:
        if self.closing or self.global_worker is not None:
            return
        from .global_registry import poll_registered
        from .global_session import is_global
        skip = [self.current_folder] if (self.current_folder is not None and self.worker is None
                                         and self.display_settings.managed_runpod
                                         and is_global(self.display_settings)) else []
        if self.batch_worker is not None and self.batch_worker_folder is not None:
            # Keep other owned sessions' idle and failure countdowns running;
            # only the running batch's own session is observed by its worker.
            skip.append(self.batch_worker_folder)
        self.global_worker = OperationWorker(lambda: poll_registered(skip), self)
        self.global_worker.succeeded.connect(self.show_registered_global_status)
        self.global_worker.failed.connect(self.global_poll_failed)
        self.global_worker.finished.connect(self.global_poll_finished)
        self.global_worker.start()

    def notify_alerts(self, entries: list[dict], title: str = "simpleGen · RunPod") -> None:
        """Desktop notification (tray where available, else the RunPod banner), once per alert."""
        fresh = [e for e in entries if e.get("key") not in self.shown_alerts]
        if not fresh:
            return
        for entry in fresh:
            self.shown_alerts.add(entry.get("key"))
        from PySide6.QtWidgets import QSystemTrayIcon
        if QSystemTrayIcon.isSystemTrayAvailable():
            if self.tray is None:
                self.tray = QSystemTrayIcon(self.windowIcon(), self)
                self.tray.show()
            for entry in fresh:
                self.tray.showMessage(title, entry["message"],
                                      QSystemTrayIcon.MessageIcon.Warning, 15000)
        self.alert_text = "\n".join(entry["message"].replace("\n", " · ") for entry in fresh[-3:])
        self.global_status.setText((self.global_status.text() + "\n" if self.global_status.text() else "")
                                   + "ALERT — " + self.alert_text)
        self.global_status.setVisible(True)

    def show_registered_global_status(self, summary: dict) -> None:
        count = summary.get("active", 0)
        if not count:
            self.global_status.setVisible(False)
            return
        issues = summary.get("needs_attention", [])
        text = (f"RUNPOD — {count} desktop-owned GPU session(s) are being monitored. "
                "Keep simpleGen running for automatic shutdown.")
        if issues:
            text += "\nNeeds attention: " + ", ".join(Path(path).name for path in issues) + ". Open the saved project to inspect."
        for line in summary.get("notices", []):
            text += "\n" + line
        self.global_status.setText(text)
        self.global_status.setVisible(True)
        self.notify_alerts(summary.get("alerts", []))

    def global_poll_failed(self, _message: str) -> None:
        self.global_status.setText("RUNPOD — Desktop-owned session monitoring needs attention. Open a saved cloud project to inspect.")
        self.global_status.setVisible(True)

    def global_poll_finished(self) -> None:
        worker, self.global_worker = self.global_worker, None
        worker.deleteLater()
        if self.closing and self.worker is None and self.cloud_worker is None:
            self.close()

    def cloud_action_failed(self, message: str) -> None:
        self.cloud_banner.setText(
            "RUNPOD — Billing may continue. Session status could not be confirmed.\n" + message
        )
        self.cloud_banner.setVisible(True)

    def show_cloud_status(self, state: dict) -> None:
        self.cloud_state = state
        stopped = state.get("billing_confirmed_stopped") is True
        phase = str(state.get("phase", "unknown"))
        ack = "verified" if state.get("downloads_verified") else "awaiting verification"
        remaining = state.get("idle_remaining_seconds")
        idle = f" · Idle countdown: {remaining}s" if remaining is not None else ""
        billing = "Compute retirement confirmed" if stopped else "Costs money until compute retirement is confirmed"
        text = f"RUNPOD — {billing}\nSession: {phase} · Downloads: {ack}{idle}"
        if state.get("alerts"):
            text += "\n" + "; ".join(str(alert) for alert in state["alerts"])
        from .cost_safety import session_notices
        for line in session_notices(state):
            text += "\n" + line
        self.cloud_banner.setText(text)
        from .cost_safety import session_alerts
        self.notify_alerts(session_alerts(state))
        self.cloud_banner.setVisible(True)
        if stopped:
            self.cloud_timer.stop()
        self.cloud_shutdown_button.setEnabled(
            not stopped and self.worker is None and self.batch_worker is None
            and self.cloud_worker is None and self.current_folder is not None
        )

    def cloud_action_finished(self) -> None:
        worker, self.cloud_worker = self.cloud_worker, None
        worker.deleteLater()
        self.cloud_status_button.setEnabled(self.batch_worker is None)
        self.cloud_shutdown_button.setEnabled(
            self.worker is None and self.batch_worker is None and self.current_folder is not None
            and not (self.cloud_state or {}).get("billing_confirmed_stopped")
        )
        if self.closing and self.worker is None:
            self.close()

    def handle_event(self, event: dict) -> None:
        message = str(event.get("message", ""))
        stage = str(event.get("stage", ""))
        text = f"{stage}: {message}" if stage else message
        self.status.setText(text)
        if text:
            self.log.appendPlainText(text)
        progress = event.get("progress")
        if progress is None:
            self.progress.setRange(0, 0)
        else:
            self.progress.setRange(0, 100)
            self.progress.setValue(max(0, min(100, int(progress))))
        if event.get("folder"):
            folder = Path(event["folder"])
            if self.batch_worker is not None:
                try:
                    child = read_state(folder / "state.json")
                    if child.get("batch_parent"):
                        from .batches import registered_stores
                        owner = next((store.get(child["batch_parent"]) for store in registered_stores()
                                      if (store.root / child["batch_parent"] / "state.json").is_file()), None)
                        if owner:
                            folder = owner["folder"]
                except (OSError, ValueError, KeyError, TypeError):
                    pass
            self.current_folder = Path(folder)
            self.report_button.setEnabled(True)
            if self.active_fork_seed is not None:
                self.staged_fork = None
                self.active_fork_seed = None
        if isinstance(event.get("cloud_status"), dict):
            self.show_cloud_status(event["cloud_status"])
        if isinstance(event.get("completed_result"), dict):
            self.show_result(event["completed_result"], switch_tab=False)
        self.update_jobs_status()

    def generation_succeeded(self, result: dict) -> None:
        self.finish_schedule(True)
        try:
            if self.scheduler.reconcile_completed(Path(result["folder"])):
                self.refresh_schedule_count()
        except (OSError, ValueError, KeyError, TypeError) as error:
            self.log.appendPlainText(f"Schedule ledger: {error}")
        self.progress.setRange(0, 100)
        self.progress.setValue(100)
        self.current_folder = Path(result["folder"])
        self.report_button.setEnabled(True)
        self.update_summary()
        self.log.appendPlainText(f"Complete. Saved to {self.current_folder}")
        self.status.setText(f"Complete. Saved to {self.current_folder}")
        self.show_result(result)

    def generation_failed(self, message: str) -> None:
        self.finish_schedule(False)
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.status.setText(message)
        self.log.appendPlainText(message)
        from .oom_transfer import is_oom_message
        if not self.display_settings.cloud and is_oom_message(message) and self.current_folder:
            self.oom_source = self.current_folder
            self.oom_button.show()
            self.status.setText("Local generation ran out of memory. Review settings or copy it to a RunPod batch.")
        if self.current_folder:
            self.log.appendPlainText(f"Report and saved run: {self.current_folder}")
            from .repair import refusal
            try:
                state = read_state(self.current_folder / "state.json")
                self.repair_button.setVisible(refusal(self.current_folder, state) is None)
                from .repair import salvage_refusal
                self.salvage_button.setVisible(salvage_refusal(self.current_folder, state) is None)
            except (OSError, ValueError):
                self.repair_button.hide()
                self.salvage_button.hide()

    def worker_finished(self) -> None:
        worker = self.worker
        self.worker = None
        self.set_running(False)
        self.update_summary()
        if not self.closing:
            QTimer.singleShot(0, self.poll_cloud_status)
        if worker:
            worker.deleteLater()
        if self.current_folder and not self.closing:
            try:
                state = read_state(self.current_folder / "state.json")
                if state.get("status") == "needs_review":
                    folder = self.current_folder
                    QTimer.singleShot(0, lambda: self.open_saved_run_choice(folder))
            except (OSError, ValueError):
                pass
        if self.closing and self.cloud_worker is None:
            self.close()

    def show_result(self, result: dict, *, switch_tab: bool = True) -> None:
        variants = result.get("variations", [])
        previous = self.variation_picker.currentIndex()
        self.variation_picker.blockSignals(True)
        self.variation_picker.clear()
        for variant in variants:
            self.variation_picker.addItem(variant["name"], {**variant, "folder": result["folder"]})
        self.variation_picker.setVisible(len(variants) > 1)
        self.variation_picker.setCurrentIndex(max(0, min(previous, len(variants) - 1)))
        self.variation_picker.blockSignals(False)
        self.display_video(self.variation_picker.currentData() if variants else result)
        if switch_tab:
            self.tabs.setCurrentIndex(1)

    def select_variation(self, _index: int) -> None:
        result = self.variation_picker.currentData()
        if result:
            self.display_video(result)

    def display_video(self, result: dict) -> None:
        folder = Path(result["folder"]).expanduser().resolve()
        path = Path(result["final_path"]).expanduser()
        if not path.is_absolute():
            path = folder / path
        path = path.resolve()
        self.result_folder = folder
        self.result_folder_button.setEnabled(True)
        self.video_path.setText(f"Final video: {path}")
        self.folder_path.setText(f"Project folder: {folder}")
        meta = result.get("metadata", {})
        size = int(meta.get("size_bytes", 0)) / (1024 * 1024)
        duration = float(meta.get("duration", 0))
        self.video_info.setText(
            f"{meta.get('width', '?')} × {meta.get('height', '?')} px"
            f"    ·    {duration:.2f} seconds    ·    {size:.2f} MiB"
        )
        self.player.setSource(QUrl.fromLocalFile(str(path)))
        self.play_button.setEnabled(True)
        self.shown_final, self.shown_metadata = path, dict(meta)
        self.upscale_panel.show_final(path)

    def play_version(self, path: str) -> None:
        """Switch Watch between the original final cut and its verified 4K60 version."""
        self.player.stop()
        self.player.setSource(QUrl.fromLocalFile(path))
        is_original = self.shown_final is not None and Path(path) == self.shown_final
        self.video_path.setText(f"{'Final video' if is_original else '4K60 version'}: {path}")
        if not is_original:
            from .upscale import upscaled_metadata
            meta = upscaled_metadata(Path(path)) or {}
            self.video_info.setText(f"{meta.get('width', '?')} × {meta.get('height', '?')} px · 60 fps"
                                    f"    ·    {float(meta.get('duration', 0)):.2f} seconds"
                                    f"    ·    {int(meta.get('size_bytes', 0)) / (1024 * 1024):.2f} MiB")
        else:
            meta = self.shown_metadata
            self.video_info.setText(
                f"{meta.get('width', '?')} × {meta.get('height', '?')} px"
                f"    ·    {float(meta.get('duration', 0)):.2f} seconds"
                f"    ·    {int(meta.get('size_bytes', 0)) / (1024 * 1024):.2f} MiB")

    def upscale_action(self, action: str, job_id: str) -> None:
        from . import upscale
        try:
            if action == "upscale":
                if self.shown_final is None:
                    return
                settings = replace(self.settings, aspect_ratio=self.display_settings.aspect_ratio)
                dialog = UpscaleDialog(self.shown_final, settings, self.shown_metadata, self)
                if dialog.exec() != QDialog.DialogCode.Accepted:
                    return
                upscale.create(self.shown_final, origin="manual", **dialog.choice())
            elif action == "cancel":
                upscale.request_cancel(job_id)
            elif action == "retry":
                upscale.retry(job_id)
            elif action == "dismiss":
                upscale.dismiss(job_id)
        except (OSError, ValueError, KeyError, upscale.UpscaleError) as error:
            QMessageBox.warning(self, "4K60 upscale", str(error))
        self.upscale_panel.refresh()
        QTimer.singleShot(0, self.poll_upscales)

    def poll_upscales(self) -> None:
        if self.closing or self.upscale_worker is not None:
            return
        from . import upscale
        settings = self.settings
        self.upscale_worker = OperationWorker(lambda: upscale.tick(settings), self)
        self.upscale_worker.succeeded.connect(self.upscales_polled)
        self.upscale_worker.failed.connect(lambda message: self.upscale_status_line(message))
        self.upscale_worker.finished.connect(self.upscale_poll_finished)
        self.upscale_worker.start()

    def upscales_polled(self, result: dict) -> None:
        from . import upscale
        self.upscale_note = result.get("note", "")
        records = result.get("records", [])
        self.upscale_panel.refresh(records)
        active = sum(1 for r in records if r["status"] in upscale.ACTIVE)
        flagged = sum(1 for r in records if r["status"] in upscale.FLAGGED)
        parts = [f"{active} in progress"] if active else []
        if flagged:
            parts.append(f"{flagged} need your decision (open the run in Watch)")
        line = ("4K60 upscales: " + ", ".join(parts)) if parts else ""
        if line and self.upscale_note:
            line += f" · {self.upscale_note}"
        self.upscale_status_line(line)
        self.notify_alerts(result.get("alerts", []), title="simpleGen · 4K60")

    def upscale_status_line(self, text: str) -> None:
        self.upscale_status.setText(text)
        self.upscale_status.setVisible(bool(text))

    def upscale_poll_finished(self) -> None:
        worker, self.upscale_worker = self.upscale_worker, None
        worker.deleteLater()
        if self.closing:
            self.close()

    def open_completed_project(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self, "Open completed project", self.output_folder.text()
        )
        if not folder:
            return
        try:
            state = read_state(Path(folder) / "state.json")
            result = dict(state.get("result") or state)
            result["folder"] = folder
            path = Path(result["final_path"])
            if not path.is_absolute():
                path = Path(folder) / path
            if not path.is_file():
                raise ValueError("The final video is missing. Resume the run if it is incomplete.")
            if self.worker is None and isinstance(state.get("settings"), dict):
                self.display_settings = Settings(**state["settings"])
                self.current_folder = Path(folder)
                self.cloud_state = None
                self.cloud_timer.start()
                self.update_summary()
            self.show_result(result)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            QMessageBox.warning(self, "Could not open completed project", str(exc))

    def open_run_folder(self) -> None:
        if self.current_folder:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.current_folder.resolve())))

    def open_result_folder(self) -> None:
        if self.result_folder:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.result_folder)))

    def toggle_playback(self) -> None:
        if self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.player.pause()
        else:
            self.player.play()

    def update_position(self, position: int) -> None:
        if not self.seek.isSliderDown():
            self.seek.setValue(position)

        def clock(milliseconds: int) -> str:
            seconds = milliseconds // 1000
            return f"{seconds // 60}:{seconds % 60:02d}"

        self.time_label.setText(f"{clock(position)} / {clock(self.player.duration())}")

    def playback_error(self, _error, message: str) -> None:
        self.status.setText(
            f"Playback: {message}. The saved video is available through Open folder."
        )

    def closeEvent(self, event: QCloseEvent) -> None:
        self.closing = True
        self.schedule_timer.stop()
        self.batch_timer.stop()
        self.cloud_timer.stop()
        self.global_timer.stop()
        self.workflow_timer.stop()
        self.upscale_timer.stop()
        if self.worker is not None:
            self.closing = True
            self.worker.detach_event.set()
            if self.choice_dialog is not None:
                self.choice_dialog.reject()
            self.status.setText("Detaching… Cloud compute and saved work remain available for reconnect.")
            event.ignore()
            return
        if self.batch_worker is not None:
            self.batch_worker.detach_event.set()
            self.status.setText("Detaching from batch. Saved work and compute remain available to reconnect.")
            event.ignore()
            return
        if self.batch_stage_worker is not None:
            self.batch_stage_worker.detach_event.set()
            event.ignore()
            return
        if self.cloud_worker is not None:
            self.closing = True
            event.ignore()
            return
        if self.global_worker is not None:
            event.ignore()
            return
        if self.schedule_probe is not None:
            event.ignore()
            return
        if self.override_worker is not None:
            event.ignore()
            return
        if self.catalog_worker is not None:
            event.ignore()
            return
        if self.upscale_worker is not None:
            self.status.setText("Finishing a 4K60 upscale transfer before closing…")
            event.ignore()
            return
        self.player.stop()
        try:
            self.collect_settings().save()
        except (OSError, ValueError):
            pass
        self.scheduler.close()
        event.accept()


def main() -> None:
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("simpleGen")
    app.setStyle("Fusion")
    app.setStyleSheet(STYLE)
    try:
        window = MainWindow()
    except (OSError, ValueError, TypeError) as exc:
        QMessageBox.critical(None, "simpleGen could not open", f"Could not load settings: {exc}")
        return
    window.show()
    app.exec()
