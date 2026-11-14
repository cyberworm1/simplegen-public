"""Agent settings can never reach launch in a state the CLI would refuse (2026-10-01).

Reported: Grok + "Balanced" + no model passed scheduling and failed only at dispatch
("Select an available model in Settings; this provider has no cached tier mapping").
One rule (agents.resolve_agent) now drives preflight, every settings dialog and
build_command; these tests check every provider/tier/model/reasoning combination.
"""

import itertools
import json
import os
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

from simplegen import agents, preflight
from simplegen.settings import Settings

TIERS = ("Economy", "Balanced", "Strongest", "Custom")
LEVELS = ("auto", "none", "minimal", "low", "medium", "high", "xhigh", "max", "ultra")
CODEX_MODELS = [
    {"slug": "gpt-6-luna", "visibility": "list", "supported_reasoning_levels": [{"effort": "low"}, {"effort": "medium"}]},
    {"slug": "gpt-6-sol", "visibility": "list",
     "supported_reasoning_levels": [{"effort": "low"}, {"effort": "medium"}, {"effort": "high"}]},
    {"slug": "gpt-6-astra", "visibility": "list",
     "supported_reasoning_levels": [{"effort": "medium"}, {"effort": "high"}, {"effort": "xhigh"}]},
]
MODELS = {
    "codex": ["", "gpt-6-luna", "gpt-6-sol", "gpt-6-astra", "not-a-codex-model"],
    "claude": ["", "haiku", "sonnet", "opus", "claude-explicit-id"],
    "grok": ["", "grok-explicit-id"],
}


@pytest.fixture
def codex_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("CODEX_HOME", str(tmp_path))
    (tmp_path / "models_cache.json").write_text(json.dumps({"models": CODEX_MODELS}))
    monkeypatch.setattr(agents, "executable", lambda name: f"/usr/bin/{name}")
    return tmp_path


def every_combination():
    for provider in agents.PROVIDERS:
        for tier, model, reasoning in itertools.product(TIERS, MODELS[provider], LEVELS):
            yield Settings(provider=provider, model_tier=tier, model=model, reasoning=reasoning)


def test_rule_matches_launch_for_every_combination(codex_cache, tmp_path):
    """agent_problem is None exactly when build_command launches, and a launch is always sound."""
    listed = {m["slug"] for m in CODEX_MODELS}
    for settings in every_combination():
        problem = agents.agent_problem(settings)
        try:
            agents.build_command(settings, tmp_path, [tmp_path], True)
            launched = True
        except ValueError as error:
            launched = False
            assert str(error) == problem, settings
        assert launched == (problem is None), settings
        if launched:
            model, reasoning = agents.resolve_agent(settings)
            assert reasoning in agents.reasoning_options(settings.provider, model), settings
            if settings.provider == "codex":
                assert model in listed or (model == "" and reasoning == "auto"), settings
            if settings.provider == "grok":
                assert model or settings.model_tier == "Custom", settings


def test_reported_case_grok_balanced_without_model(codex_cache):
    settings = Settings(provider="grok", model_tier="Balanced", model="", reasoning="high")
    problem = agents.agent_problem(settings)
    assert problem and "Custom" in problem and "grok models" in problem
    assert agents.agent_problem(replace(settings, model_tier="Custom")) is None
    assert agents.agent_problem(replace(settings, model="grok-explicit-id")) is None


@pytest.mark.parametrize("settings, fragment", [
    (Settings(provider="codex", model_tier="Custom", model="", reasoning="ultra"), "Reasoning 'ultra'"),
    (Settings(provider="codex", model_tier="Custom", model="not-a-codex-model", reasoning="auto"),
     "isn't in your Codex model list"),
    (Settings(provider="codex", model_tier="Custom", model="gpt-6-luna", reasoning="xhigh"), "Reasoning 'xhigh'"),
    (Settings(provider="claude", model_tier="Economy", model="", reasoning="high"), "Reasoning 'high'"),
    (Settings(provider="claude", model_tier="Custom", model="sonnet", reasoning="ultra"), "Reasoning 'ultra'"),
    (Settings(provider="local-thing", model_tier="Custom", model="", reasoning="auto"), "Unknown agent"),
])
def test_previously_silent_or_late_failures_are_caught(codex_cache, settings, fragment):
    assert fragment in (agents.agent_problem(settings) or "")


def test_codex_without_a_model_list_allows_only_the_default(tmp_path, monkeypatch):
    monkeypatch.setenv("CODEX_HOME", str(tmp_path))  # no models_cache.json
    assert agents.agent_problem(Settings(provider="codex", model_tier="Custom", reasoning="auto")) is None
    assert agents.agent_problem(Settings(provider="codex", model_tier="Balanced", reasoning="auto"))
    assert agents.agent_problem(Settings(provider="codex", model_tier="Custom", reasoning="high"))


def test_switching_agents_always_lands_on_a_valid_configuration(codex_cache):
    from simplegen.ui import change_agent
    for start in every_combination():
        for provider in agents.PROVIDERS:
            assert agents.agent_problem(change_agent(start, provider)) is None, (start, provider)


# Preflight gates Generate, Schedule, dispatch, batch add/prepare and Resume


def test_preflight_fails_on_unrunnable_agent_settings(codex_cache):
    bad = Settings(provider="grok", model_tier="Balanced", model="", reasoning="high")
    result = preflight.run_preflight(bad, object_info={}, resolve=lambda name: name)
    check = next(c for c in result["checks"] if c["id"] == "agent")
    assert result["status"] == preflight.FAIL and check["status"] == preflight.FAIL
    assert "Custom" in check["message"]


def test_preflight_skips_agent_check_when_accepted_work_is_reused(codex_cache):
    bad = Settings(provider="grok", model_tier="Balanced", model="", reasoning="high")
    result = preflight.run_preflight(bad, object_info={}, resolve=lambda name: name, agent_needed=False)
    check = next(c for c in result["checks"] if c["id"] == "agent")
    assert check["status"] == preflight.PASS and "reused" in check["message"]


def test_saved_preflight_needs_the_agent_only_without_accepted_work(codex_cache, tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(preflight, "run_preflight", lambda settings, **kw: calls.append(kw["agent_needed"]) or {})
    bad = Settings(provider="grok", model_tier="Balanced", reasoning="high").public_dict()
    for name, extra in (("fresh", {}), ("accepted", {"workflow_acceptance": {"valid": True}})):
        folder = tmp_path / name
        folder.mkdir()
        (folder / "state.json").write_text(json.dumps({"schema_version": 5, "settings": bad, **extra}))
        preflight.run_saved_preflight(folder, cached_references=False)
    assert calls == [True, False]


# Dialogs refuse to save a combination that cannot launch


@pytest.fixture
def app():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


@pytest.fixture
def base(tmp_path, monkeypatch, codex_cache):
    monkeypatch.setenv("SIMPLEGEN_HOME", str(tmp_path / "config"))
    for folder in ("references", "templates"):
        (tmp_path / folder).mkdir()
    (tmp_path / "templates" / "workflow.json").write_text("{}")
    return Settings(project_folder=str(tmp_path / "references"), template_folder=str(tmp_path / "templates"),
                    template_file=str(tmp_path / "templates" / "workflow.json"),
                    output_folder=str(tmp_path / "outputs"), provider="grok", model_tier="Custom",
                    model="", reasoning="medium")


def test_settings_dialog_turns_modelless_grok_tiers_into_custom(app, base):
    from simplegen import ui
    dialog = ui.SettingsDialog(base)
    for tier in ("Economy", "Balanced", "Strongest"):
        dialog.tier.setCurrentText(tier)
        assert dialog.tier.currentText() == "Custom"
        assert agents.agent_problem(dialog.collect_settings()) is None
    dialog.deleteLater()


def test_settings_dialog_refuses_unrunnable_settings(app, base, monkeypatch):
    from simplegen import ui
    warnings = []
    monkeypatch.setattr(ui.QMessageBox, "warning", lambda *args: warnings.append(args[2]))
    dialog = ui.SettingsDialog(base)
    monkeypatch.setattr(dialog, "collect_settings",
                        lambda: replace(base, model_tier="Balanced", model="", reasoning="high"))
    dialog.accept()
    assert warnings and "Custom" in warnings[0]
    assert dialog.result() != ui.QDialog.DialogCode.Accepted
    assert not (Path(os.environ["SIMPLEGEN_HOME"]) / "settings.json").exists()
    dialog.deleteLater()


def test_schedule_edit_dialog_refuses_unrunnable_settings(app, base):
    from datetime import UTC, datetime, timedelta

    from simplegen import ui
    due = datetime.now(UTC) + timedelta(hours=1)
    record = {"id": "x", "concept": "c", "due_at": due.isoformat(), "selected_local": due.astimezone().isoformat(),
              "status": "scheduled"}
    dialog = ui.ScheduleEditDialog(record, base, lambda *a: {}, None)
    dialog.tier.setCurrentText("Balanced")
    assert dialog.tier.currentText() == "Custom"  # Grok has no Balanced model
    dialog.settings = replace(base, provider="claude")
    dialog.provider.setCurrentIndex(dialog.provider.findData("claude"))
    dialog.tier.setCurrentText("Economy")  # haiku
    dialog.reasoning.clear()
    dialog.reasoning.addItem("high")
    with pytest.raises(ValueError, match="Reasoning 'high'"):
        dialog.values()
    dialog.deleteLater()


def test_batch_member_dialog_refuses_free_text_reasoning_typos(app, base):
    from simplegen.batch_ui import MemberEditDialog
    state = {"concept": "c", "settings": replace(base, provider="claude", model_tier="Balanced",
                                                  model="sonnet", reasoning="high").public_dict()}
    dialog = MemberEditDialog(state, None)
    dialog.reasoning.setText("hgih")
    with pytest.raises(ValueError, match="Reasoning 'hgih'"):
        dialog.values()
    dialog.deleteLater()
