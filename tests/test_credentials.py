import json
import os
import stat
from dataclasses import replace
from pathlib import Path

import pytest

from simplegen.credentials import REQUIRED, check_setup, load_credentials, scaffold_runpod
from simplegen.settings import Settings


def test_scaffold_is_private_exclusive_and_credential_free_config(tmp_path):
    paths = scaffold_runpod(tmp_path / "private")
    assert stat.S_IMODE((tmp_path / "private").stat().st_mode) == 0o700
    for filename in paths.values():
        assert stat.S_IMODE(Path(filename).stat().st_mode) == 0o600
    first = {key: Path(path).read_bytes() for key, path in paths.items()}
    assert scaffold_runpod(tmp_path / "private") == paths
    assert {key: Path(path).read_bytes() for key, path in paths.items()} == first
    creds = json.loads(first["credentials_file"])
    assert creds == dict.fromkeys(("RUNPOD_API_KEY", "SG_COMFY_USERNAME", "SG_COMFY_PASSWORD"), "")
    assert not (tmp_path / "private/controller.json").exists()


def test_credentials_environment_override_and_file_reference(tmp_path, monkeypatch):
    paths = scaffold_runpod(tmp_path)
    settings = replace(Settings(), runpod_credentials_file=paths["credentials_file"])
    source = tmp_path / "linked.json"
    source.write_text(json.dumps({"password": "private-password"}))
    source.chmod(0o600)
    values = dict.fromkeys(REQUIRED, "private-value")
    values["SG_ADMIN_TOKEN"] = "private-master-token" * 3
    values["SG_COMFY_PASSWORD"] = {"file": str(source), "key": "password"}
    Path(paths["credentials_file"]).write_text(json.dumps(values))
    monkeypatch.setenv("RUNPOD_API_KEY", "environment-value")
    before = dict(os.environ)
    result = load_credentials(settings)
    assert result["RUNPOD_API_KEY"] == "environment-value"
    assert result["SG_COMFY_PASSWORD"] == "private-password"
    assert dict(os.environ) == before


def test_credentials_reject_public_permissions_without_exposing_contents(tmp_path):
    path = tmp_path / "public.json"
    path.write_text('{"RUNPOD_API_KEY":"do-not-print-me"}')
    path.chmod(0o644)
    with pytest.raises(ValueError, match="mode 600") as error:
        load_credentials(replace(Settings(), runpod_credentials_file=str(path)))
    assert "do-not-print-me" not in str(error.value)


def test_scaffold_symlink_rejected_without_overwriting(tmp_path):
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "linked"
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match="symbolic"):
        scaffold_runpod(link)
    assert list(target.iterdir()) == []


def test_setup_is_local_and_reports_missing_values(tmp_path, monkeypatch):
    paths = scaffold_runpod(tmp_path)
    for key in REQUIRED:
        monkeypatch.delenv(key, raising=False)
    settings = replace(Settings(), runpod_config_file=paths["config_file"],
                       runpod_credentials_file=paths["credentials_file"],
                       runpod_schema_file=paths["schema_file"])
    result = check_setup(settings)
    assert result["ready"] is False
    assert len(result["issues"]) >= 2
    assert "SG_ADMIN_TOKEN" not in json.dumps(result)
