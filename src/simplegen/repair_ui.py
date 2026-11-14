"""Repair dialog: pick replacement models from the target's own choices."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

from . import repair
from .preflight import FAIL, PASS, UNVERIFIED, summary
from .qtutil import retire_worker


class RepairDialog(QDialog):
    """One row per model selection; Save revision publishes a linked revision.

    ``inspect``/``apply`` run off the UI thread. After a revision is saved,
    Run now / Schedule… close the dialog with ``choice`` set.
    """

    def __init__(self, folder: Path, parent=None, *, inspect=None, apply=None, confirm=None, retarget=None,
                 choose_file=None):
        super().__init__(parent)
        self.folder = Path(folder)
        self.inspect_call = inspect or repair.inspect
        self.apply_call = apply or repair.apply
        self.retarget_call = retarget or repair.retarget
        self.choose_file = choose_file or self._choose_file
        self.last_action = None
        self.confirm = confirm or self._confirm
        self.choice = None
        self.result: dict | None = None
        self.info: dict | None = None
        self.combos: dict[str, QComboBox] = {}
        self.worker = None
        self.setWindowTitle("Repair model selection")
        self.setMinimumWidth(720)
        layout = QVBoxLayout(self)
        intro = QLabel(f"Saved run: {self.folder.name}\nChoose a replacement for each unavailable model. "
                       "The accepted script, prompts and workflow edits are kept; no agent runs. "
                       "A new linked revision is saved; the original run is not changed.\n"
                       "To use a different workflow file, use Edit as new run (needs a new agent pass).")
        intro.setWordWrap(True)
        layout.addWidget(intro)
        self.grid = QGridLayout()
        for column, text in enumerate(("Node", "Input", "Current model", "Replacement")):
            self.grid.addWidget(QLabel(f"<b>{text}</b>"), 0, column)
        layout.addLayout(self.grid)
        self.status = QLabel("Reading the destination's available models…")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        buttons = QHBoxLayout()
        self.save_button = QPushButton("Save revision")
        self.save_button.setEnabled(False)
        self.save_button.clicked.connect(lambda: self.save())
        self.run_button = QPushButton("Run now")
        self.run_button.setEnabled(False)
        self.run_button.clicked.connect(lambda: self.finish("run"))
        self.schedule_button = QPushButton("Schedule…")
        self.schedule_button.setEnabled(False)
        self.schedule_button.clicked.connect(lambda: self.finish("schedule"))
        close = QPushButton("Close")
        close.clicked.connect(self.reject)
        self.retarget_button = QPushButton("Use another workflow file…")
        self.retarget_button.setToolTip("Recompile the saved plan against another workflow; no agent pass.")
        self.retarget_button.clicked.connect(lambda: self.retarget())
        try:
            import json
            version = json.loads((self.folder / "state.json").read_text(encoding="utf-8")).get("schema_version", 1)
        except (OSError, ValueError):
            version = 1
        self.retarget_button.setVisible(version >= 5)
        for button in (self.save_button, self.retarget_button, self.run_button, self.schedule_button, close):
            buttons.addWidget(button)
        layout.addLayout(buttons)
        self._start(lambda: self.inspect_call(self.folder), self.loaded)

    def _start(self, operation, succeeded):
        from .ui import OperationWorker
        worker = OperationWorker(operation, self)
        self.worker = worker
        worker.succeeded.connect(succeeded)
        worker.failed.connect(self.failed)
        worker.finished.connect(self._finished)
        worker.start()

    def _finished(self):
        worker, self.worker = self.worker, None
        retire_worker(self, worker)
        self.update_actions()  # Buttons were computed while the worker still ran.

    def failed(self, message: str):
        self.status.setText(f"Cannot repair: {message}")
        self.save_button.setEnabled(False)
        self.retarget_button.setEnabled(self.result is None)

    def loaded(self, info: dict):
        self.info = info
        if info.get("error") or info.get("schema_sha256") is None:
            self.failed(f"the destination's models could not be read ({info.get('error') or 'unknown'}). "
                        "Unverified is never compatible; try again when it is reachable.")
            return
        for row, selection in enumerate(info["selections"], 1):
            self.grid.addWidget(QLabel(f"{selection['title']} (node {selection['node_id']})"), row, 0)
            self.grid.addWidget(QLabel(selection["input"]), row, 1)
            current = selection["current"] + ("" if selection["available"] else "  — unavailable")
            self.grid.addWidget(QLabel(current), row, 2)
            combo = QComboBox()
            combo.addItem("", None)
            for choice in selection["choices"]:
                combo.addItem(choice, choice)
            if selection["available"]:
                combo.setCurrentIndex(combo.findData(selection["current"]))
            combo.currentIndexChanged.connect(self.update_actions)
            self.grid.addWidget(combo, row, 3)
            self.combos[selection["key"]] = combo
        missing = sum(not s["available"] for s in info["selections"])
        self.status.setText(f"{missing} model selection(s) unavailable on {info['target'].title()}."
                            if missing else "Every selected model is available on the destination.")
        self.update_actions()

    def replacements(self) -> dict:
        chosen = {}
        for selection in (self.info or {}).get("selections", []):
            value = self.combos[selection["key"]].currentData()
            if value and value != selection["current"]:
                chosen[selection["key"]] = value
        return chosen

    def update_actions(self, *_args):
        self.save_button.setEnabled(self.result is None and self.worker is None and bool(self.replacements()))
        self.retarget_button.setEnabled(self.result is None and self.worker is None)

    def save(self, override: bool = False):
        chosen = self.replacements()
        if not chosen or self.result is not None:
            return
        self._run(lambda flag: self.apply_call(self.folder, chosen, override_unverified=flag), override)

    def _run(self, action, override: bool = False):
        self.last_action = action
        self.save_button.setEnabled(False)
        self.retarget_button.setEnabled(False)
        self.status.setText("Saving revision and running preflight…")
        self._start(lambda: action(override), lambda result: self.saved(result, override))

    def _choose_file(self):
        from PySide6.QtWidgets import QFileDialog
        path, _ = QFileDialog.getOpenFileName(self, "Choose workflow JSON", str(self.folder), "Workflow JSON (*.json)")
        return path or None

    def retarget(self):
        """Recompile the saved plan against another workflow file (schema 5 runs)."""
        if self.result is not None or self.worker is not None:
            return
        path = self.choose_file()
        if not path:
            return
        self._run(lambda flag: self.retarget_call(self.folder, Path(path), override_unverified=flag))

    def saved(self, result: dict, override: bool):
        status = result.get("status")
        preflight = result.get("preflight") or {}
        if status == "published":
            self.result = result
            for combo in self.combos.values():
                combo.setEnabled(False)
            note = " (unverified, override recorded)" if preflight.get("status") == UNVERIFIED else ""
            self.status.setText(f"Preflight {preflight.get('status', PASS)}{note}. "
                                f"Revision saved: {Path(result['folder']).name}")
            self.run_button.setEnabled(True)
            self.schedule_button.setEnabled(True)
            self.save_button.setEnabled(False)
            return
        if status == UNVERIFIED and not override:
            self.status.setText(f"Preflight unverified: {summary(preflight)}")
            if self.confirm(preflight) and self.last_action is not None:
                self._run(self.last_action, True)
                return
        elif status == FAIL:
            self.status.setText(f"Preflight failed; revision not saved: {summary(preflight)}")
        self.update_actions()

    def _confirm(self, preflight: dict) -> bool:
        answer = QMessageBox.question(
            self, "Preflight could not verify",
            summary(preflight) + "\n\nSave the revision without verification? The override is recorded.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel, QMessageBox.StandardButton.Cancel)
        return answer == QMessageBox.StandardButton.Yes

    def finish(self, choice: str):
        if self.result is None:
            return
        self.choice = choice
        self.accept()

    def reject(self):
        if self.worker is not None:
            return  # A running check owns this dialog until it returns.
        super().reject()

    def done(self, code):
        if self.worker is not None and code == QDialog.DialogCode.Rejected:
            return
        super().done(code)
