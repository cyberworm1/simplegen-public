"""Local OOM escalation creates a fresh cloud draft, without running an agent/pod."""
import json
from pathlib import Path

import pytest

from simplegen.oom_transfer import cloud_settings, create_cloud_copy, is_oom_message, local_oom
from simplegen.settings import Settings, atomic_json


@pytest.fixture
def source(tmp_path):
    assets = tmp_path / 'assets'
    assets.mkdir()
    templates = tmp_path / 'templates'
    templates.mkdir()
    template = templates / 'cloud.json'
    atomic_json(template, {'nodes': [
        {'id': 1, 'type': 'UNETLoader', 'widgets_values': ['cloud-old.safetensors', 'default']},
        {'id': 2, 'type': 'CLIPLoader', 'widgets_values': ['clip-old.safetensors', 'minimax', 'default']},
        {'id': 3, 'type': 'VAELoader', 'widgets_values': ['video-vae.safetensors']}]})
    settings = Settings(project_folder=str(assets), template_folder=str(templates),
                        template_file='local.json', output_folder=str(tmp_path / 'runs'),
                        duration=30, steps=20, quantity=3, aspect_ratio='9:16',
                        provider='grok', model_tier='Custom', model='', reasoning='medium',
                        runpod_default_workflow=str(template))
    folder = tmp_path / 'failed-local'
    state = {'run_id': 'original', 'client_id': 'old-client', 'prompt_id': 'failed-prompt',
             'status': 'failed', 'settings': settings.public_dict(), 'concept': 'Goat carries hay',
             'error': 'CUDA out of memory', 'downloads': {'old': {}}}
    atomic_json(folder / 'state.json', state)
    return folder, state, settings, template


def test_cloud_copy_preserves_source_and_replaces_only_model_selectors(source):
    folder, state, settings, template = source
    original = (folder / 'state.json').read_bytes()
    original_template = template.read_bytes()
    target_settings = cloud_settings(state, settings)
    assert target_settings.quantity == 3 and target_settings.steps == 20
    assert target_settings.aspect_ratio == '9:16' and target_settings.global_desktop
    target, draft = create_cloud_copy(folder, target_settings, 'Revised concept')
    assert (folder / 'state.json').read_bytes() == original
    assert template.read_bytes() == original_template
    assert draft['run_id'] != state['run_id'] and 'prompt_id' not in draft
    assert draft['downloads'] == {} and draft['status'] == 'created'
    assert draft['forked_from']['run_id'] == 'original'
    nodes = json.loads((target / 'template.original.json').read_text())['nodes']
    assert nodes[0]['widgets_values'][0] == settings.runpod_default_model
    assert nodes[1]['widgets_values'][0] == settings.runpod_default_clip
    assert nodes[2]['widgets_values'][0] == 'video-vae.safetensors'
    assert 'workflow.json' not in {p.name for p in target.iterdir()}


def test_only_failed_local_oom_is_eligible(source):
    folder, state, settings, _ = source
    assert local_oom(folder, state)
    assert is_oom_message('torch.OutOfMemoryError')
    assert not is_oom_message('model file missing')
    state['settings']['cloud'] = True
    assert not local_oom(folder, state)
    atomic_json(folder / 'state.json', state)
    with pytest.raises(ValueError, match='failed local'):
        create_cloud_copy(folder, cloud_settings(state, settings), 'retry')


def test_missing_cloud_loader_does_not_create_project(source):
    folder, state, settings, template = source
    atomic_json(template, {'nodes': []})
    with pytest.raises(ValueError, match='one UNETLoader'):
        create_cloud_copy(folder, cloud_settings(state, settings), 'retry')
    assert not Path(settings.output_folder).exists()


def test_saved_run_dialog_offers_oom_transfer(source, monkeypatch):
    monkeypatch.setenv('QT_QPA_PLATFORM', 'offscreen')
    from PySide6.QtWidgets import QApplication

    from simplegen.ui import SavedRunDialog
    app = QApplication.instance() or QApplication([])
    folder, state, _, _ = source
    dialog = SavedRunDialog(state, folder)
    assert not dialog.rp_button.isHidden()
    dialog.rp_button.click()
    assert dialog.choice == 'runpod'
    dialog.close()
    assert app is not None


@pytest.mark.usefixtures("_no_desktop_audio")
def test_oom_action_stages_cloud_settings_for_batch_review(source, monkeypatch, tmp_path):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.setenv("SIMPLEGEN_HOME", str(tmp_path / "home"))
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication, QDialog

    from simplegen.batch_ui import MemberEditDialog
    from simplegen.ui import MainWindow
    folder, _, settings, _ = source
    monkeypatch.setattr(Settings, "load", lambda: settings)
    monkeypatch.setattr(MemberEditDialog, "exec", lambda self: QDialog.DialogCode.Accepted)
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    for timer in window.findChildren(QTimer):
        timer.stop()
    captured = []
    monkeypatch.setattr(window, "stage_batch_member", lambda *args, **kwargs: captured.append((args, kwargs)))
    window.add_oom_to_batch(folder)
    assert len(captured) == 1
    args, kwargs = captured[0]
    assert args[0].managed_runpod and args[0].global_desktop
    assert args[0].runpod_default_clip == settings.runpod_default_clip
    assert args[1] == "Goat carries hay" and kwargs["oom_source"] == folder
    assert window.current_folder is None
    window.close()
    app.processEvents()
