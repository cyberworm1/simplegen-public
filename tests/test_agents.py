import json
import sys
import threading
from threading import Event

import pytest

from simplegen import agents
from simplegen.settings import Settings


def test_environment_removes_billable_keys(monkeypatch):
    for key in (
        "CODEX_API_KEY",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "XAI_API_KEY",
        "CLAUDE_CODE_USE_BEDROCK",
    ):
        monkeypatch.setenv(key, "test-secret")
    env = agents.clean_environment()
    assert env["GROK_DISABLE_API_KEY_AUTH"] == "true"
    assert not any(
        key in env
        for key in (
            "CODEX_API_KEY",
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "XAI_API_KEY",
            "CLAUDE_CODE_USE_BEDROCK",
        )
    )


def test_frozen_bundle_restores_loader_paths_for_external_cli(monkeypatch):
    monkeypatch.setattr(agents.sys, "frozen", True, raising=False)
    monkeypatch.setenv("LD_LIBRARY_PATH", "/bundle/internal")
    monkeypatch.setenv("LD_LIBRARY_PATH_ORIG", "/user/libs")
    assert agents.clean_environment()["LD_LIBRARY_PATH"] == "/user/libs"


def test_subscription_auth_rejects_api_session(monkeypatch):
    class Result:
        returncode = 0
        stdout = "Logged in using API key"
        stderr = ""

    monkeypatch.setattr(agents.subprocess, "run", lambda *a, **k: Result())
    with pytest.raises(ValueError, match="subscription"):
        agents._subscription("codex", "codex", {})


def test_claude_rejects_stored_api_key_precedence(monkeypatch):
    class Result:
        returncode = 0
        stdout = json.dumps(
            {"loggedIn": True, "authMethod": "claude.ai", "apiKeySource": "/login managed key"}
        )
        stderr = ""

    monkeypatch.setattr(agents.subprocess, "run", lambda *a, **k: Result())
    with pytest.raises(ValueError, match="subscription"):
        agents._subscription("claude", "claude", {})


def test_grok_camel_case_result_envelope():
    result = agents._extract(json.dumps({"structuredOutput": {"script": "A frame", "prompts": []}}))
    assert result["script"] == "A frame"


def test_grok_final_json_after_progress_preambles():
    envelope = {
        "text": 'Reading files.Editing workflow.{"script":"A blue sea", "used_references":[]}'
    }
    assert agents._extract(json.dumps(envelope))["script"] == "A blue sea"
    with pytest.raises(agents.InvalidAgentResponse):
        agents._extract(json.dumps({"text": 'Reading.{"script":"incomplete"} Still working.'}))


def test_provider_commands_use_requested_effort_without_api_or_bypass(tmp_path, monkeypatch):
    monkeypatch.setattr(agents, "executable", lambda name: name)
    models = {"codex": "gpt-6-sol", "claude": "sonnet", "grok": "test-model"}  # Codex: a listed model
    for provider in ("codex", "claude", "grok"):
        config = Settings(
            provider=provider, model=models[provider], model_tier="Custom", reasoning="low"
        )
        args = agents.build_command(config, tmp_path, [tmp_path])
        assert models[provider] in args
        assert not any("bypass" in a or "api-key" in a for a in args)
        assert any("low" in arg for arg in args)
        assert "--bare" not in args
        assert "--max-turns" not in args


def test_direct_edit_commands_enable_scoped_writes_and_keep_subscription_mode(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(agents, "executable", lambda name: name)
    for provider in ("codex", "claude", "grok"):
        model = {"codex": "gpt-6-sol"}.get(provider, "test")  # Codex: a listed model
        settings = Settings(provider=provider, model=model, model_tier="Custom")
        args = agents.build_command(settings, tmp_path, [tmp_path / "references"], True)
        assert "--max-turns" not in args
        assert not any("bypass" in arg or arg == "--bare" for arg in args)
        if provider == "codex":
            assert args[args.index("--sandbox") + 1] == "workspace-write"
            assert "--add-dir" not in args
        elif provider == "claude":
            assert "--restricted" in args
            assert f"Edit(/{tmp_path / 'workflow.json'})" in args[args.index("--allowedTools") + 1]
        else:
            assert args[args.index("--sandbox") + 1] == "workspace"
            assert "--json-schema" not in args
            assert f"Edit({tmp_path / 'workflow.json'})" in args
            assert args[args.index("--permission-mode") + 1] == "dontAsk"
    assert set(agents.plan_schema(True)["properties"]) == {"script", "used_references"}


def test_codex_model_choices_from_installed_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("CODEX_HOME", str(tmp_path))
    (tmp_path / "models_cache.json").write_text(
        json.dumps(
            {
                "models": [
                    {
                        "slug": "future-mini",
                        "visibility": "list",
                        "supported_reasoning_levels": [{"effort": "low"}, {"effort": "high"}],
                    },
                    {"slug": "hidden", "visibility": "hide"},
                ]
            }
        )
    )
    assert agents.model_options("codex") == ["future-mini"]
    assert agents.suggest_model("codex", "Economy") == "future-mini"
    assert agents.reasoning_options("codex", "future-mini") == ["auto", "low", "high"]


def test_redact_environment_credentials(monkeypatch):
    monkeypatch.setenv("SIMPLEGEN_COMFY_TOKEN", "a-private-cloud-secret")
    assert "a-private-cloud-secret" not in agents.redact("token a-private-cloud-secret")


@pytest.fixture
def fake_cli(tmp_path, monkeypatch):
    script = tmp_path / "fake_cli.py"
    script.write_text(
        "import json,sys,time\nrequest=sys.stdin.read()\n"
        "print(json.dumps({'structured_output': {'script':'A scene', 'prompts':[], 'references':[], 'used_references':[]}}))\n"
    )
    monkeypatch.setattr(agents, "build_command", lambda *a: [sys.executable, str(script)])
    monkeypatch.setattr(agents, "_subscription", lambda *a: None)
    config = Settings(project_folder=str(tmp_path), template_folder=str(tmp_path))
    return script, config


def test_cli_process_result_and_saved_logs(tmp_path, fake_cli):
    _, settings = fake_cli
    result = agents.generate_plan(
        settings, tmp_path, {"concept": "synthetic"}, Event(), lambda _: None
    )
    assert result["script"] == "A scene"
    assert (tmp_path / "plan.json").is_file()
    assert (tmp_path / "agent.log").is_file()


def test_agent_cancellation_terminates_process_and_preserves_log(tmp_path, fake_cli):
    script, settings = fake_cli
    script.write_text("import time\nprint('started',flush=True)\ntime.sleep(30)\n")
    cancel = Event()
    timer = threading.Timer(0.2, cancel.set)
    timer.start()
    try:
        with pytest.raises(agents.Cancelled):
            agents.generate_plan(settings, tmp_path, {}, cancel, lambda _: None)
    finally:
        timer.cancel()
    assert "started" in (tmp_path / "agent.log").read_text()


def test_agent_malformed_output_has_no_repair_loop(tmp_path, fake_cli):
    script, settings = fake_cli
    script.write_text("print('not a plan')\n")
    with pytest.raises(ValueError, match="did not return"):
        agents.generate_plan(settings, tmp_path, {}, Event(), lambda _: None)
    assert not (tmp_path / "plan.json").exists()


@pytest.mark.parametrize("stage", ["creative", "resources"])
@pytest.mark.parametrize("cancelled", [False, True])
def test_agents_can_exceed_legacy_timeout_and_still_cancel(tmp_path, monkeypatch, stage, cancelled):
    from simplegen import resources

    clock = {"now": 0}
    cancel = Event()
    response = {"script": "Completed after an hour", "used_references": [], "options": []}

    class Process:
        returncode = 0
        calls = 0

        def communicate(self, input=None, timeout=None):
            self.calls += 1
            if self.calls == 1:
                clock["now"] = 3600
                if cancelled:
                    cancel.set()
                raise agents.subprocess.TimeoutExpired("fake-agent", timeout)
            return json.dumps(response), ""

        def poll(self):
            return self.returncode

    monkeypatch.setattr(agents.time, "monotonic", lambda: clock["now"])
    monkeypatch.setattr(agents.subprocess, "Popen", lambda *a, **k: Process())
    monkeypatch.setattr(agents, "build_command", lambda *a: ["fake-agent"])
    monkeypatch.setattr(agents, "_subscription", lambda *a: None)
    settings = Settings(project_folder=str(tmp_path), template_folder=str(tmp_path), agent_timeout=900)
    run = agents.generate_plan if stage == "creative" else resources.assess
    if cancelled:
        with pytest.raises(agents.Cancelled):
            run(settings, tmp_path, {}, cancel, lambda _: None)
    else:
        assert run(settings, tmp_path, {}, cancel, lambda _: None) == response
    assert (tmp_path / "agent.log").is_file()


def test_claude_prefers_user_install_over_desktop_path(tmp_path, monkeypatch):
    monkeypatch.setattr(agents.Path, 'home', lambda: tmp_path)
    binary = tmp_path / '.npm-global/bin/claude'
    binary.parent.mkdir(parents=True)
    binary.write_text('test')
    binary.chmod(0o755)
    monkeypatch.setattr(agents.shutil, 'which', lambda name: '/usr/local/bin/claude')
    assert agents.executable('claude') == str(binary)


def test_claude_startup_failure_is_not_reported_as_logout(monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr(agents.subprocess, 'run', lambda *a, **k: SimpleNamespace(
        returncode=1, stdout='', stderr='SyntaxError: private diagnostic'))
    with pytest.raises(agents.AgentPreflightError, match='CLI installation') as error:
        agents._subscription('claude', '/old/claude', {})
    assert 'sign in using' not in str(error.value)
    assert 'private diagnostic' not in str(error.value)


def _assert_strict(node, path="root"):
    """OpenAI strict structured output (Codex) and Claude's --json-schema tool input."""
    if isinstance(node, dict):
        if node.get("type") == "object":
            assert "properties" in node, f"{path}: free-form object"
            assert node.get("additionalProperties") is False, f"{path}: additionalProperties must be false"
            assert set(node["required"]) == set(node["properties"]), f"{path}: every property must be required"
        for key, value in node.items():
            _assert_strict(value, f"{path}.{key}")
    elif isinstance(node, list):
        for index, value in enumerate(node):
            _assert_strict(value, f"{path}[{index}]")


@pytest.mark.parametrize("family", ["edit", "edit_variations", "compiler", "compiler_variations", "single_shot",
                                    "scene_chain"])
def test_every_agent_response_format_is_strict(family):
    brief = {"prompt_sections": ["summary", "detail"], "timing_options": [{"id": "a"}],
             "reference_slots": [{"node_id": "12"}]}
    schema = {
        "edit": lambda: agents.plan_schema(True),
        "edit_variations": lambda: agents.plan_schema(True, 3),
        "compiler": lambda: agents.plan_schema(False, 1, brief),
        "compiler_variations": lambda: agents.plan_schema(False, 3, brief),
        "single_shot": lambda: agents.plan_schema(creative={"single_shot": True}),
        "scene_chain": lambda: agents.plan_schema(creative={"scene_chain": True, "workflow_brief": True}),
    }[family]()
    _assert_strict(schema)
