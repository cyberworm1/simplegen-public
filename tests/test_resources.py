"""Resource selection/cost evidence, with no paid resources or real agent calls."""

import csv
import json
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit

import pytest

from simplegen import resources
from simplegen.agents import Cancelled
from simplegen.settings import Settings


@pytest.fixture
def config():
    return SimpleNamespace(
        volume_id="volume",
        region="EU-RO-1",
        gpu_payload={
            "gpuTypeIds": ["GPU A"],
            "gpuCount": 2,
            "allowedCudaVersions": ["13.1", "13.0"],
        },
        cpu_payload={"computeType": "CPU"},
        cpu_max_hourly=0.3,
        gpu_max_hourly=12,
    )


def gpu(name="GPU A", region="EU-RO-1", price=2, available="HIGH"):
    return {
        "id": name,
        "memory": 96,
        "secure": True,
        "price": {"secure": price},
        "dataCenters": [{"id": region, "availability": available}],
    }


class Provider:
    def __init__(self, rows=None):
        self.rows = rows if rows is not None else [gpu()]
        self.calls = []
        self.region = "EU-RO-1"
        self.cpu_rate = 0.1

    def volume(self, volume_id):
        self.calls.append(("volume", volume_id))
        return {"id": volume_id, "dataCenterId": self.region}

    def quote(self, payload):
        assert payload["computeType"] == "CPU"
        return self.cpu_rate

    def _request(self, method, path, *, catalog):
        assert method == "GET" and catalog
        self.calls.append((method, path))
        return {"gpus": self.rows}


def estimate(settings, folder, request, cancel, emit):
    return {
        "options": [
            {
                "gpu_id": item["gpu_id"],
                "suitable": True,
                "runtime_seconds_low": 60,
                "runtime_seconds_high": 180,
                "reason": "96 GB available; runtime uncalibrated.",
                "recommended": i == 0,
            }
            for i, item in enumerate(request["available"])
        ]
    }


def options(config, tmp_path, *, provider=None, runner=estimate, tiers=None, cancel=None):
    return resources.gpu_options(
        config,
        {},
        tiers or [],
        {"1": {"class_type": "Sampler", "inputs": {"steps": 20}}},
        tmp_path,
        Settings(),
        cancel or Event(),
        lambda _: None,
        provider=provider or Provider(),
        agent_runner=runner,
    )


def test_region_tier_filtering_and_cost_math(config, tmp_path):
    provider = Provider(
        [
            gpu(),
            gpu("GPU B"),
            gpu("GPU C"),
            gpu("GPU D", "US-TX-1"),
            gpu("GPU E", available="NONE"),
            gpu("GPU F", price=10),
        ]
    )
    choices = options(
        config, tmp_path, provider=provider, tiers=["GPU B", "GPU D", "GPU E", "GPU F"]
    )
    assert {x["gpu_id"] for x in choices} == {"GPU B", "GPU A"}
    choice = choices[0]
    assert choice["gpu_count"] == 2 and choice["hourly_usd"] == 4
    assert choice["selection_source"] == "manual_catalog"
    assert "estimated_cost_low" not in choice
    query = parse_qs(urlsplit(provider.calls[1][1]).query)
    assert query["count"] == ["2"] and query["minCudaVersion"] == ["13.0"]
    assert choice["data_center"] == config.region and choice["volume_id"] == config.volume_id


def test_global_options_include_all_available_locations(tmp_path):
    config = SimpleNamespace(
        gpu_tiers=["GPU A"], gpu_payload={"gpuTypeIds": ["GPU A"], "gpuCount": 1},
        global_profile=SimpleNamespace(min_cuda_version="13.0", volume_id="global-volume"),
        gpu_max_hourly=5.0,
    )
    row = gpu(region="AP-IN-1", available="HIGH")
    row["dataCenters"] += [{"id": "US-TX-1", "availability": "LOW"},
                           {"id": "AP-IN-1", "availability": "MEDIUM"}]
    provider = Provider([row, gpu("GPU NOT IN TIERS", region="AP-IN-1", price=3)])
    choice = options(config, tmp_path, provider=provider)[0]
    assert choice["availability"] == "HIGH"
    assert choice["data_center"] == "US-TX-1"
    assert choice["available_data_centers"] == ["US-TX-1", "AP-IN-1"]
    assert choice["available_location_count"] == 2
    assert choice["placement_scope"] == "global"
    assert choice["global_volume_id"] == "global-volume"
    query = parse_qs(urlsplit(provider.calls[0][1]).query)
    assert query["minCudaVersion"] == ["13.0"] and "countryCodes" not in query




def test_quantity_scales_runtime_and_cost_but_not_vram(config, tmp_path):
    choice = resources.gpu_options(config, {}, [], {}, tmp_path, Settings(quantity=3),
        Event(), lambda _: None, provider=Provider(), agent_runner=estimate)[0]
    assert choice["quantity"] == 3 and choice["vram_gb"] == 96
    assert "runtime_seconds_low" not in choice


def test_existing_history_adds_quantity_without_losing_rows(tmp_path):
    path = tmp_path / "history.csv"
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=resources.HISTORY_FIELDS[:-1])
        writer.writeheader()
        writer.writerow({"run_id": "older", "generation_seconds": 25})
    resources.record_run(tmp_path, {"run_id": "new", "settings": {"quantity": 3}}, history_path=path)
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]["run_id"] == "older" and rows[0]["generation_seconds"] == "25"
    assert [row["quantity"] for row in rows] == ["1", "3"]


@pytest.mark.parametrize("rows", [[], [gpu(available="NONE")], [gpu(region="US-TX-1")]])
def test_no_capacity_never_calls_agent(config, tmp_path, rows):
    with pytest.raises(ValueError, match="No configured GPU"):
        options(
            config,
            tmp_path,
            provider=Provider(rows),
            runner=lambda *args: pytest.fail("agent called"),
        )


def test_wrong_volume_region_refuses(config, tmp_path):
    provider = Provider()
    provider.region = "US-TX-1"
    with pytest.raises(ValueError, match="volume"):
        options(config, tmp_path, provider=provider)
    assert len(provider.calls) == 1


def test_cpu_over_ceiling_refuses_before_agent(config, tmp_path):
    provider = Provider()
    provider.cpu_rate = 1
    with pytest.raises(ValueError, match="CPU quote exceeds"):
        options(
            config, tmp_path, provider=provider, runner=lambda *args: pytest.fail("agent called")
        )






def test_cancel_before_network(config, tmp_path):
    cancel = Event()
    cancel.set()
    provider = Provider()
    with pytest.raises(Cancelled):
        options(config, tmp_path, provider=provider, cancel=cancel)
    assert not provider.calls


def test_workload_excludes_prompts_and_reference_paths():
    summary = resources.workload(
        {
            "1": {
                "class_type": "Sampler",
                "inputs": {
                    "steps": 30,
                    "width": 768,
                    "length": 121,
                    "prompt": "private plot",
                    "image": "/private/actor.jpg",
                    "unet_name": "/models/H3-int8.safetensors",
                },
            }
        }
    )
    assert summary["nodes"][0]["steps"] == 30
    assert summary["nodes"][0]["unet_name"] == "H3-int8.safetensors"
    assert "private" not in json.dumps(summary)


@pytest.mark.parametrize("envelope", ["plain", "claude", "grok", "stream"])
def test_resource_parser_accepts_provider_final_envelopes(envelope):
    result = {"options": []}
    data = json.dumps(result)
    if envelope == "claude":
        data = json.dumps({"type": "result", "structured_output": result})
    if envelope == "grok":
        data = json.dumps({"text": "Assessment complete. " + data})
    if envelope == "stream":
        data = "{}\n" + json.dumps({"type": "result", "result": data})
    assert resources._extract(data) == result


def test_invalid_parser_never_accepts_progress():
    with pytest.raises(ValueError, match="did not return"):
        resources._extract('{"text":"I will inspect the GPU catalog"}')


def state(run_id="one"):
    return {
        "run_id": run_id,
        "status": "complete",
        "settings": {"megapixels": 1, "duration": 10},
        "resource_choice": {
            "gpu_id": "GPU A",
            "hourly_usd": 2,
            "cpu_hourly_usd": 0.1,
            "estimated_cost_low": 1,
            "estimated_cost_high": 2,
        },
        "generation_started_at": "2026-09-21T00:00:00+00:00",
        "generation_completed_at": "2026-09-21T00:10:00+00:00",
        "managed_session": {
            "session_started_at": "2026-09-21T00:00:00+00:00",
            "session_completed_at": "2026-09-21T00:20:00+00:00",
        },
    }


def test_history_idempotent_measured_and_estimated_separate(tmp_path):
    path = tmp_path / "history.csv"
    resources.record_run(tmp_path, state(), history_path=path)
    resources.record_run(tmp_path, state(), history_path=path)
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1
    assert float(rows[0]["generation_seconds"]) == 600
    assert float(rows[0]["observed_compute_cost_estimate_usd"]) == pytest.approx(0.7)
    assert rows[0]["actual_billed_usd"] == ""


def test_history_concurrent_jobs_are_not_lost(tmp_path):
    path = tmp_path / "history.csv"
    with ThreadPoolExecutor(max_workers=4) as executor:
        list(
            executor.map(
                lambda i: resources.record_run(tmp_path, state(str(i)), history_path=path),
                range(20),
            )
        )
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    assert {row["run_id"] for row in rows} == {str(i) for i in range(20)}


def test_incomplete_history_does_not_invent_bill(tmp_path):
    path = tmp_path / "history.csv"
    data = state()
    data["managed_session"] = {}
    data.pop("generation_completed_at")
    resources.record_run(tmp_path, data, history_path=path)
    with path.open() as stream:
        row = next(csv.DictReader(stream))
    assert row["observed_compute_cost_estimate_usd"] == row["generation_seconds"] == ""






def test_manual_catalog_never_calls_resource_agent(config, tmp_path, monkeypatch):
    def forbidden(*args):
        pytest.fail("GPU selection must not invoke an agent")
    monkeypatch.setattr(resources, "assess", forbidden)
    choices = options(config, tmp_path, runner=forbidden)
    saved = json.loads((tmp_path / "gpu-catalog.json").read_text())
    assert saved["options"] == choices
    assert choices and all("recommended" not in c for c in choices)
