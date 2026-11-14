"""Real engine and Qty execution share a single batch allocation and acknowledgement."""
import copy
import json
from importlib.resources import files
from pathlib import Path
from threading import Event

import pytest
from test_variations import sequence  # noqa: F401
from test_workflows import info, workflow  # noqa: F401

from simplegen import batches, engine, global_session, refcache, resources
from simplegen.settings import Settings
from simplegen_runpod.global_gpu import GlobalGPUProfile


@pytest.mark.parametrize('fault_mode', ['', 'uncertain', 'detach', 'monitor', 'missing_reference', 'tamper', 'render_failed'])
def test_real_batch_qty_execution_and_recovery(sequence, info, monkeypatch, fault_mode):  # noqa: F811
    _, agent_calls, jobs, trace, events, fault, detach, _, root, _ = sequence
    profile = root/'global-profile.json'
    profile.write_text(json.dumps({**json.loads(files('simplegen').joinpath('data/global-runpod-profile.json').read_text()),
                                  'template_id': 'tmpl-test', 'volume_id': 'vol-test'}))
    settings = Settings(project_folder=str(root/'assets'), template_folder=str(root/'templates'),
        template_file='template.json', output_folder=str(root/'runs'), quantity=3, duration=4,
        managed_runpod=True, global_desktop=True, runpod_config_file=str(profile))
    store = batches.BatchStore(settings.output_folder)
    batch = store.create()
    for concept in ['First theme', 'Second theme']:
        folder, _ = engine.create_run(settings, concept)
        store.add(batch['id'], folder)
    catalog = [{'gpu_id': identity, 'hourly_usd': rate, 'gpu_count': 1, 'vram_gb': 96,
                'availability': 'HIGH', 'data_center': 'US-TEST',
                'data_centers': [{'id': 'US-TEST', 'availability': 'HIGH'}]}
               for identity, rate in [('a', 1), ('b', 2)]]
    monkeypatch.setattr(batches, '_catalog', lambda *args, **kwargs: copy.deepcopy(catalog))
    monkeypatch.setattr(resources, 'assess_batch', lambda *args: pytest.fail('Unexpected GPU agent'))
    sealed = []
    allocations = []
    class Session:
        retired = False
        def __init__(self, settings, folder, state, persist, *args):
            self.settings, self.state, self.persist = settings, state, persist
        def schema(self): return copy.deepcopy(info)
        def config(self):
            return global_session.GlobalConfig(GlobalGPUProfile(template_id="tmpl-test", volume_id="vol-test"), ['a', 'b'], 3)
        def connect(self, choice=None):
            saved = self.state.setdefault('managed_session', {})
            if not saved.get('sid'):
                assert choice and choice['gpu_id'] == 'a'
                saved.update(backend='global_desktop', sid='shared', name='sg-shared-gpu', pod_id='pod',
                             create_intent=True, jobs={})
                allocations.append('pod')
                self.persist()
            return self.settings
        def before_submit(self, job_id=None):
            assert job_id
            registered = self.state['managed_session']['jobs']
            if job_id in registered: return False
            registered[job_id] = None
            self.persist()
            return True
        def bind(self, prompt_id, job_id=None):
            self.state['managed_session']['jobs'][job_id] = prompt_id
            self.persist()
        def complete(self, receipts):
            assert len(jobs) == 6
            assert len(receipts) == 12
            assert len(self.state['managed_session']['jobs']) == 6
            assert all(Path(receipt['path']).is_file() for receipt in receipts)
            assert all(m['status'] == 'complete' for m in self.state['members'])
            self.state['managed_session']['downloads_verified'] = True
            sealed.append(receipts)
            self.persist()
        def close(self): pass
    monkeypatch.setattr(global_session, 'session_for', Session)
    ready = store.prepare(batch['id'], events.append, Event())
    assert ready['status'] == 'prepared'
    assert len(agent_calls) == 2
    assert not allocations and not jobs and not sealed
    store.configure(batch['id'], catalog_options=catalog, gpu_choices=[{'gpu_id': 'a', 'max_hourly': 1},
                                              {'gpu_id': 'b', 'max_hourly': 2}])
    store.run_now(batch['id'])
    fault['mode'] = 'uncertain' if fault_mode == 'missing_reference' else fault_mode
    if fault_mode == 'render_failed':
        original_monitor = engine.Comfy.monitor
        def failed_render(self, prompt_id, *args, **kwargs):
            if prompt_id == 'job-2':
                raise RuntimeError('ComfyUI execution failed: out of memory')
            return original_monitor(self, prompt_id, *args, **kwargs)
        monkeypatch.setattr(engine.Comfy, 'monitor', failed_render)
    def on_event(event):
        events.append(event)
        if fault_mode == 'tamper' and event.get('stage') == 'Queueing':
            first = Path(store.get(batch['id'])['members'][0]['folder']) / 'variation-001/workflow.api.json'
            first.write_text(first.read_text() + ' ')
    if fault_mode:
        with pytest.raises(RuntimeError):
            store.run_batch(batch['id'], on_event, Event(), detach)
        assert not sealed
        owner = store.get(batch['id'])['managed_session']
        if fault_mode == 'detach':
            assert 'owner_failure' not in owner  # Detach is not an owner failure.
        else:
            assert owner['owner_failure']['kind'] == 'batch'
            assert owner['recovery_completed_at']  # Every submitted job was tried.
        if fault_mode == 'tamper':
            assert not jobs and store.get(batch['id'])['status'] == 'failed'
            return
        assert len(jobs) == (2 if fault_mode in ('uncertain', 'missing_reference') else 6)
        assert store.dispatch_due(events.append, Event()) == []
        assert len(jobs) == (2 if fault_mode in ('uncertain', 'missing_reference') else 6)
        detach.clear()
        store.resume(batch['id'])
        resumed = store.get(batch['id'])['managed_session']
        assert 'owner_failure' not in resumed and 'recovery_completed_at' not in resumed
        if fault_mode == 'render_failed':
            # A permanently failed prompt must not strand later queued outputs,
            # trigger a re-render, or allow acknowledgement on explicit resume.
            assert sum(name == 'download' for name, _ in trace) == 10
            with pytest.raises(RuntimeError, match='out of memory'):
                store.run_batch(batch['id'], events.append, Event(), detach)
            assert len(jobs) == 6 and not sealed
            assert sum(name == 'download' for name, _ in trace) == 10
            assert store.get(batch['id'])['status'] == 'failed'
            assert len(agent_calls) == 2
            return
        if fault_mode == 'missing_reference':
            # Phase 6: the verified cached copy stands in for the original (mount gone after acceptance).
            (root/'assets'/'reference.png').unlink()
            assert refcache.usage()['bytes'] > 0
    result = store.run_batch(batch['id'], on_event, Event(), detach)
    assert refcache.usage()['bytes'] == 0  # Batch complete: pins released and blobs collected.
    assert result['status'] == 'complete'
    assert allocations == ['pod']
    assert len(jobs) == 6 and len(sealed) == 1
    assert len(agent_calls) == 2
    if fault_mode in ('', 'detach', 'monitor'):
        submissions = [i for i, (name, _) in enumerate(trace) if name == 'submit']
        downloads = [i for i, (name, _) in enumerate(trace) if name == 'download']
        assert max(submissions) < min(downloads)  # Comfy queue is filled before transfer work.
    assert len([name for name, _ in trace if name == 'submit']) == 6
    assert len([name for name, _ in trace if name == 'upload']) == 1
    assert len([r for r in result['members'] if r.get('result')]) == 2


def test_execution_time_excludes_queue_wait():
    history = {'status': {'messages': [
        ['execution_start', {'timestamp': 1_800_000_000_000}],
        ['execution_success', {'timestamp': 1_800_000_013_250}],
    ]}}
    assert engine._execution_seconds(history) == 13.25
    assert engine._execution_seconds({'status': {'completed': True}}) is None
