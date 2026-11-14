"""Shared test helpers."""

import pytest

# A passing preflight result for tests whose subject is not preflight itself.
PASSED = {"status": "pass", "target": "local", "checked_at": "2026-09-28T00:00:00+00:00",
          "schema_sha256": None, "template_sha256": None, "checks": []}


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path_factory, monkeypatch):
    """Default SIMPLEGEN_HOME (reference cache, stores) to a temporary folder; tests may override it."""
    monkeypatch.setenv("SIMPLEGEN_HOME", str(tmp_path_factory.mktemp("home")))


@pytest.fixture(autouse=True)
def _codex_model_cache(tmp_path_factory, monkeypatch):
    """A realistic Codex model list, so default Settings (Codex, Economy, low) are a valid
    agent configuration, as on a machine where Codex is installed. Tests may override CODEX_HOME."""
    import json
    home = tmp_path_factory.mktemp("codex")
    levels = lambda *names: [{"effort": n} for n in names]
    (home / "models_cache.json").write_text(json.dumps({"models": [
        {"slug": "gpt-6-luna", "visibility": "list", "supported_reasoning_levels": levels("low", "medium")},
        {"slug": "gpt-6-sol", "visibility": "list", "supported_reasoning_levels": levels("low", "medium", "high")},
        {"slug": "gpt-6-astra", "visibility": "list",
         "supported_reasoning_levels": levels("medium", "high", "xhigh")},
    ]}))
    monkeypatch.setenv("CODEX_HOME", str(home))


@pytest.fixture(autouse=True)
def _batch_preflight_passes(monkeypatch):
    """Batch tests use synthetic templates; preflight has its own tests."""
    from simplegen import batches
    monkeypatch.setattr(batches, "member_preflight", lambda target: dict(PASSED, target="cloud"))


@pytest.fixture(autouse=True)
def _no_real_webhook(monkeypatch):
    """Tests never contact real services: alert webhooks fail offline unless a test stubs them."""
    from simplegen import cost_safety

    def offline(url, text):
        raise OSError("tests never POST to a real webhook")
    monkeypatch.setattr(cost_safety, "_post", offline)


@pytest.fixture(autouse=True)
def _no_unexpected_modal_warnings(monkeypatch):
    """An unexpected warning box would block an offscreen test forever; fail fast instead.

    Tests that expect a warning replace QMessageBox.warning themselves (their monkeypatch wins).
    """
    try:
        from PySide6.QtWidgets import QMessageBox
    except ImportError:  # pragma: no cover -- Qt not installed
        return

    def unexpected(*args, **_kwargs):
        text = " | ".join(str(a) for a in args[1:3])
        raise AssertionError(f"Unexpected modal warning in test: {text}")
    for name in ("warning", "critical"):
        monkeypatch.setattr(QMessageBox, name, unexpected)


@pytest.fixture
def _no_desktop_audio(monkeypatch):
    """UI tests inspect controls/paths; never initialize host audio services.

    Qt audio discovery blocks in restricted headless sessions. FFmpeg media
    verification remains real and is exercised separately by the engine tests.
    """
    from simplegen import ui
    monkeypatch.setenv("QT_MEDIA_BACKEND", "none")
    monkeypatch.setattr(ui, "QAudioOutput", lambda parent=None: None)
