"""Offline checks for the app-owned template and its folder exemption."""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from simplegen import builtin_workflows as builtin
from simplegen.engine import _template
from simplegen.h3scene.__main__ import template
from simplegen.settings import Settings

LEGACY = Path(__file__).parent / "fixtures/real/h3scene/legacy/template.json"


@pytest.mark.parametrize("modified", [False, True])
def test_old_builtin_removed_only_when_identical(modified):
    from simplegen.h3scene.__main__ import template_bytes
    from simplegen.h3scene_compat import LEGACY_FILENAME, legacy_template_bytes

    payload = LEGACY.read_bytes()
    assert legacy_template_bytes(template_bytes()) == payload
    stale = builtin.scene_template_path().with_name(LEGACY_FILENAME)
    stale.parent.mkdir(parents=True)
    if modified:
        payload += b"\n"
    stale.write_bytes(payload)
    assert builtin.ensure_scene_template().is_file()
    assert stale.exists() == modified
    if modified:
        assert stale.read_bytes() == payload


def test_old_builtin_symlink_is_preserved(tmp_path):
    from simplegen.h3scene_compat import LEGACY_FILENAME

    target = tmp_path / "user.json"
    target.write_bytes(LEGACY.read_bytes())
    stale = builtin.scene_template_path().with_name(LEGACY_FILENAME)
    stale.parent.mkdir(parents=True)
    stale.symlink_to(target)
    builtin.ensure_scene_template()
    assert stale.is_symlink() and target.read_bytes() == LEGACY.read_bytes()


@pytest.mark.parametrize("filename_only", [False, True])
def test_old_builtin_settings_and_destination_profiles_resolve(tmp_path, filename_only):
    from simplegen.h3scene_compat import LEGACY_FILENAME

    current = builtin.ensure_scene_template()
    old = LEGACY_FILENAME if filename_only else str(current.with_name(LEGACY_FILENAME))
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({"template_file": old, "template_folder": "", "connection_profiles": {
        "local": {"template_file": old, "template_folder": ""}}}))
    settings = Settings.load(path)
    assert settings.template_file == str(current)
    assert settings.connection_profiles["local"]["template_file"] == str(current)
    assert _template(settings) == current
    settings.save(path)
    assert LEGACY_FILENAME not in path.read_text()


def test_user_export_with_old_filename_is_not_redirected(tmp_path):
    from simplegen.h3scene_compat import LEGACY_FILENAME

    exported = tmp_path / LEGACY_FILENAME
    exported.write_bytes(LEGACY.read_bytes())
    settings = Settings(template_folder=str(tmp_path), template_file=LEGACY_FILENAME)
    assert _template(settings) == exported
    assert not builtin.is_builtin_template(settings.template_file, settings.template_folder)


def test_builtin_matches_command_and_refreshes_only_when_changed(tmp_path, monkeypatch):
    from simplegen import scheduling
    command_path = template(tmp_path / "export")
    path = builtin.ensure_scene_template()
    assert path == builtin.scene_template_path()
    assert path.read_bytes() == command_path.read_bytes()
    original = path.stat()
    write = scheduling.atomic_bytes
    writes = []

    def record_write(target, payload):
        # The published template remains intact until atomic replacement.
        assert target.read_bytes() == b"stale"
        writes.append(target)
        write(target, payload)

    monkeypatch.setattr(scheduling, "atomic_bytes", record_write)
    assert builtin.ensure_scene_template() == path
    assert not writes
    assert path.stat().st_mtime_ns == original.st_mtime_ns
    path.write_bytes(b"stale")
    assert builtin.ensure_scene_template() == path
    assert writes == [path]
    assert path.read_bytes() == command_path.read_bytes()
    assert not path.with_suffix(".json.tmp").exists()


def test_unwritable_home_is_logged_and_not_offered(tmp_path, monkeypatch, caplog):
    home = tmp_path / "unwritable"
    home.write_text("Cannot create a workflow directory under a file.")
    monkeypatch.setenv("SIMPLEGEN_HOME", str(home))
    assert builtin.ensure_scene_template() is None
    assert "Built-in H3 Scene Chain workflow unavailable" in caplog.text
    assert str(home) in caplog.text


def test_only_app_owned_path_bypasses_folder_requirement(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    path = builtin.ensure_scene_template()
    settings = Settings(project_folder=str(project), template_file=str(path), output_folder=str(tmp_path / "out"))
    settings.validate()
    assert _template(settings) == path
    exported = template(tmp_path / "export")
    with pytest.raises(ValueError, match="linked template folder"):
        _template(replace(settings, template_file=str(exported)))
    with pytest.raises(ValueError, match="template folder"):
        replace(settings, template_file=str(exported)).validate()
    # Previously generated user copies keep ordinary folder semantics.
    user_settings = replace(settings, template_folder=str(exported.parent), template_file=exported.name)
    user_settings.validate()
    assert _template(user_settings) == exported
    path.unlink()
    with pytest.raises(ValueError, match="built-in H3 Scene Chain workflow is unavailable"):
        _template(settings)


def test_bundled_templates_install_unchanged_and_need_no_folder(tmp_path):
    installed = builtin.ensure_bundled_templates()
    assert [label for label, _ in installed] == [label for _, label, _ in builtin.BUNDLED]
    project = tmp_path / "project"
    project.mkdir()
    for (filename, _, title), (_, path) in zip(builtin.BUNDLED, installed):
        assert path == builtin.bundled_template_path(filename)
        assert path.read_bytes() == (builtin.BUNDLED_FOLDER / filename).read_bytes()
        assert builtin.builtin_name(path) == title
        settings = Settings(project_folder=str(project), template_file=str(path), output_folder=str(tmp_path / "out"))
        assert builtin.is_builtin_template(settings.template_file, settings.template_folder)
        settings.validate()
        assert _template(settings) == path
    # A stale copy is replaced; a missing copy reports by name.
    path = installed[0][1]
    path.write_bytes(b"stale")
    builtin.ensure_bundled_templates()
    assert path.read_bytes() == (builtin.BUNDLED_FOLDER / builtin.BUNDLED[0][0]).read_bytes()
    path.unlink()
    settings = Settings(project_folder=str(project), template_file=str(path), output_folder=str(tmp_path / "out"))
    with pytest.raises(ValueError, match="built-in H3 Persistent References Chain workflow is unavailable"):
        _template(settings)


def test_bundled_templates_carry_no_example_prompts():
    for filename, _, _ in builtin.BUNDLED:
        text = (builtin.BUNDLED_FOLDER / filename).read_text()
        assert "<Subject 1> is" not in text and "<d>[English]" not in text, filename


def test_unlinked_workflow_folder_adds_no_reference_root(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    settings = Settings(project_folder=str(tmp_path / "project"), template_folder="")
    assert settings.reference_roots() == [(tmp_path / "project").resolve()]
    linked = replace(settings, template_folder=str(tmp_path / "workflows"))
    assert linked.reference_roots() == [(tmp_path / "project").resolve(), (tmp_path / "workflows").resolve()]
