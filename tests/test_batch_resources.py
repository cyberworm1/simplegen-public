"""Aggregate GPU estimates preserve peak VRAM and charge startup only once."""
import copy
import json
from threading import Event

import pytest

from simplegen import resources
from simplegen.settings import Settings


def graph(width):
    return {'1': {'class_type': 'H3RCProject', 'inputs': {'width': width, 'height': 1120, 'default_steps': 40}},
            '2': {'class_type': 'UNETLoader', 'inputs': {'unet_name': 'h3_bf16.safetensors'}},
            '3': {'class_type': 'PrimitiveStringMultiline', 'inputs': {'value': 'private concept'}}}


def candidates():
    return [{'gpu_id': 'A', 'hourly_usd': 1, 'vram_gb': 96, 'availability': 'UNAVAILABLE'},
            {'gpu_id': 'B', 'hourly_usd': 2, 'vram_gb': 141, 'availability': 'HIGH'}]


def response():
    return {'options': [{'gpu_id': k, 'suitable': True, 'recommended': k == 'A',
                         'runtime_seconds_low': 600, 'runtime_seconds_high': 1200,
                         'reason': 'Suitable for all sequential workflows'} for k in ['A', 'B']]}


def test_batch_assessment_has_all_workloads_and_single_startup(tmp_path, monkeypatch):
    monkeypatch.setenv('SIMPLEGEN_HOME', str(tmp_path/'home'))
    captured = []
    def runner(settings, folder, request, cancel, emit):
        captured.append(request)
        return response()
    options = resources.assess_batch(Settings(quantity=3), tmp_path/'assessment',
        [graph(640), graph(640), graph(1024)], candidates(), Event(), lambda _: None,
        agent_runner=runner)
    request = captured[0]
    assert request['render_count'] == 3
    assert len(request['workloads']) == 2
    assert request['workloads'][0]['video_count'] == 2
    assert request['execution'] == 'sequential_batch'
    assert request['workloads'][1]['nodes'][0]['width'] == 1024
    assert request['workloads'][0]['nodes'][1]['unet_name'] == 'h3_bf16.safetensors'
    assert 'private concept' not in json.dumps(request)
    assert len(options) == 2
    assert options[0]['estimated_session_seconds_low'] == 600 + 1500 + 300
    assert options[0]['estimated_cost_low'] == pytest.approx(2400/3600)
    assert options[1]['estimated_cost_high'] == pytest.approx(2*3000/3600)
    assert (tmp_path/'assessment'/'receipt.json').exists()


@pytest.mark.parametrize('bad', ['duplicate', 'missing', 'negative', 'reversed', 'boolean', 'unsuitable_recommendation'])
def test_batch_assessment_rejects_malformed_response(tmp_path, monkeypatch, bad):
    monkeypatch.setenv('SIMPLEGEN_HOME', str(tmp_path/'home'))
    result = response()
    first = result['options'][0]
    if bad == 'duplicate': result['options'].append(copy.deepcopy(first))
    elif bad == 'missing': result['options'].pop()
    elif bad == 'negative': first['runtime_seconds_low'] = -1
    elif bad == 'reversed': first['runtime_seconds_low'] = 2000
    elif bad == 'boolean': first['suitable'] = 1
    elif bad == 'unsuitable_recommendation': first['suitable'] = False
    with pytest.raises(ValueError):
        resources.assess_batch(Settings(), tmp_path, [graph(640)], candidates(), Event(), lambda _: None,
                               agent_runner=lambda *args: result)
    assert not (tmp_path/'options.json').exists()


def test_cancelled_assessment_does_not_invoke_agent(tmp_path):
    cancel = Event()
    cancel.set()
    with pytest.raises(resources.agents.Cancelled):
        resources.assess_batch(Settings(), tmp_path, [graph(640)], candidates(), cancel, lambda _: None,
                              agent_runner=lambda *args: pytest.fail('Agent called'))


def test_batch_cost_history_preserves_total_work(tmp_path):
    import csv

    from simplegen.resources import record_run
    path = tmp_path / "costs.csv"
    state = {"run_id": "batch", "members": [{"id": "a"}, {"id": "b"}],
             "status": "complete", "render_count": 6, "generation_seconds": 600,
             "settings": {"quantity": 3, "duration": 30, "megapixels": 1},
             "resource_choice": {"gpu_id": "gpu", "cpu_hourly_usd": 0},
             "managed_session": {"quoted_gpu_hourly_usd": 2,
                 "session_started_at": "2026-09-25T12:00:00+00:00",
                 "session_completed_at": "2026-09-25T13:00:00+00:00"}}
    record_run(tmp_path, state, history_path=path)
    with path.open() as stream:
        row = next(csv.DictReader(stream))
    assert row["quantity"] == "6"
    assert row["generation_seconds"] == "600"
    assert row["megapixels"] == row["requested_duration_seconds"] == ""
    assert float(row["observed_compute_cost_estimate_usd"]) == 2
