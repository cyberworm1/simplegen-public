"""Managed settings store links only; credentials are owned by the setup adapter."""
import json
from dataclasses import replace

import pytest

from simplegen.settings import Settings


def test_managed_defaults_use_user_directory_and_keep_independent_tiers(tmp_path, monkeypatch):
    monkeypatch.setenv("SIMPLEGEN_HOME", str(tmp_path))
    one, two = Settings.defaults(), Settings.defaults()
    assert one.runpod_config_file == str(tmp_path / "runpod/global-profile.json")
    assert one.runpod_credentials_file == str(tmp_path / "runpod/credentials.json")
    assert one.runpod_schema_file == str(tmp_path / "runpod/object_info.json")
    assert not one.managed_runpod
    one.runpod_gpu_tiers.append("GPU A")
    assert two.runpod_gpu_tiers == []


def test_managed_mode_defers_endpoint_auth_validation_until_provisioned(tmp_path):
    settings = Settings(managed_runpod=True, comfy_url="", browser_login_file="linked/login.json")
    assert settings.cloud
    settings.save(tmp_path / "settings.json")
    saved = Settings.load(tmp_path / "settings.json")
    assert saved.managed_runpod and saved.cloud
    assert saved.comfy_url == ""
    with pytest.raises(ValueError, match="ComfyUI address"):
        replace(settings, managed_runpod=False).validate(require_folders=False)


def test_managed_validation_requires_cloud_and_path_strings():
    settings = Settings(managed_runpod=True)
    settings.cloud = False
    with pytest.raises(ValueError, match="requires cloud"):
        settings.validate(require_folders=False)
    settings.cloud = True
    settings.runpod_credentials_file = {"api_key": "must-not-be-saved"}
    with pytest.raises(ValueError, match="file paths"):
        settings.validate(require_folders=False)


def test_settings_never_read_or_copy_linked_credentials(tmp_path):
    credentials = tmp_path / "credentials.json"
    credentials.write_text('{"api_key":"private-value"}')
    settings = Settings(managed_runpod=True, runpod_credentials_file=str(credentials))
    settings.save(tmp_path / "settings.json")
    assert "private-value" not in (tmp_path / "settings.json").read_text()
    assert json.loads((tmp_path / "settings.json").read_text())["runpod_credentials_file"] == str(credentials)


def test_legacy_settings_load_with_managed_disabled_and_paths_added(tmp_path, monkeypatch):
    monkeypatch.setenv("SIMPLEGEN_HOME", str(tmp_path))
    path = tmp_path / "settings.json"
    path.write_text('{"cloud":true,"comfy_url":"https://existing.example"}')
    settings = Settings.load(path)
    assert settings.cloud and not settings.managed_runpod
    assert settings.runpod_config_file == str(tmp_path / "runpod/global-profile.json")
