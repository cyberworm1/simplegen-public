"""RunPod v2 fixture checks. These tests never contact the real provider."""

import copy
import json
from types import SimpleNamespace

import httpx
import pytest

from simplegen.runpod import RequestUncertain, RunPod, RunPodAuthentication, RunPodUnavailable


@pytest.fixture
def config(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_RUNPOD_KEY", "synthetic-private-key")
    payload = {
        "imageName": "python@sha256:test", "dockerStartCmd": ["python", "/workspace/start.py"],
        "networkVolumeId": "volume1", "volumeMountPath": "/workspace", "gpuTypeIds": ["GPU A"],
        "gpuCount": 1, "containerDiskInGb": 20, "minRAMPerGPU": 128,
        "dataCenterIds": ["EU-RO-1"], "cloudType": "SECURE", "ports": ["8189/http"],
        "env": {"CONFIG_TOKEN": "synthetic-config-secret"},
    }
    (tmp_path / "gpu-session.json").write_text(json.dumps({"payload": payload, "pod_id": "manual1"}))
    # configuration only needs the path; a malformed file proves it does not read credentials.
    (tmp_path / "browser-login.json").write_text("credentials-not-read-by-adapter")
    return SimpleNamespace(runpod_config_folder=str(tmp_path), runpod_credentials_file="",
                           runpod_token_env="TEST_RUNPOD_KEY", runpod_gpu_tiers=["GPU A", "GPU B"],
                           runpod_managed=True)


def pod(**changes):
    value = {
        "id": "owned1", "name": "simplegen-run1", "status": "RUNNING", "cost": 2.0,
        "gpu": {"id": "GPU A", "count": 1, "memory": 128}, "cloud": "SECURE",
        "dataCenterId": "EU-RO-1", "mounts": {"network": [{"volumeId": "volume1", "path": "/workspace"}]},
        "env": {"SIMPLEGEN_RUN_ID": "run1", "SIMPLEGEN_OWNER_TOKEN": "owner1", "PASSWORD": "private"},
    }
    return {**value, **changes}


def gpu(gpu_id="GPU A", region="EU-RO-1", rate=2.0, availability="HIGH"):
    return {"id": gpu_id, "memory": 96, "secure": True, "price": {"secure": rate},
            "dataCenters": [{"id": region, "availability": availability}]}


def choice(**changes):
    return {"gpu_id": "GPU A", "gpu_count": 1, "volume_id": "volume1",
            "data_center": "EU-RO-1", "hourly_usd": 2.0, **changes}


def mock_adapter(config, handler):
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return RunPod(config, client)


def catalog_response(request, *, rate=2.0):
    if request.url.path == "/v2/network-volumes/volume1":
        return httpx.Response(200, json={"id": "volume1", "dataCenter": "EU-RO-1"})
    if request.url.path == "/v2/catalog/gpus":
        return httpx.Response(200, json={"gpus": [gpu(rate=rate)]})
    return None


def test_configuration_uses_linked_paths_without_importing_manual_pod(config):
    adapter = mock_adapter(config, lambda _: pytest.fail("configuration must be offline"))
    data = adapter.configuration()
    assert data["volume_id"] == "volume1"
    assert data["cmd"] == ["python", "/workspace/start.py"]
    assert data["browser_login_file"].endswith("browser-login.json")
    assert "manual1" not in str(data)
    assert "synthetic-config-secret" not in str(data)


def test_catalog_filters_volume_region_stock_and_tier_and_uses_count(config):
    def handler(request):
        assert request.headers["Authorization"] == "Bearer synthetic-private-key"
        if "/network-volumes/" in request.url.path:
            return catalog_response(request)
        assert request.url.params["count"] == "2"
        assert "gpuCount" not in request.url.params
        assert request.url.params["minCudaVersion"] == "13.0"
        return httpx.Response(200, json={"gpus": [gpu(), gpu("GPU B", "US-TX-1"),
                                                 gpu("GPU C"), gpu("GPU B", availability="NONE")]})
    options = mock_adapter(config, handler).options(2)
    assert len(options) == 1
    assert options[0]["hourly_usd"] == 4.0
    assert options[0]["vram_gb"] == 96


def test_create_translates_v1_template_to_v2_and_keeps_auth_gateway(config):
    posts = []
    def handler(request):
        response = catalog_response(request)
        if response:
            return response
        payload = json.loads(request.content)
        posts.append(payload)
        assert payload["gpu"] == {"id": "GPU A", "count": 1, "minRamPerGpu": 128, "minCudaVersion": "13.0"}
        assert payload["mounts"]["network"] == [{"volumeId": "volume1", "path": "/workspace"}]
        assert payload["cmd"] == ["python", "/workspace/start.py"]
        assert payload["ports"] == ["8189/http"]
        assert payload["env"]["CONFIG_TOKEN"] == "synthetic-config-secret"
        assert "imageName" not in payload and "networkVolumeId" not in payload
        return httpx.Response(201, json=pod())
    result = mock_adapter(config, handler).create(choice(), "run1", "owner1")
    assert result["allocation_valid"]
    assert len(posts) == 1
    assert "PASSWORD" not in result["env"]


def test_price_change_stops_before_creation(config):
    def handler(request):
        assert request.method == "GET"
        return catalog_response(request, rate=2.1)
    with pytest.raises(RunPodUnavailable, match="price changed"):
        mock_adapter(config, handler).create(choice(), "run1", "owner1")


@pytest.mark.parametrize("failure", ["timeout", "server_error", "bad_response", "empty_receipt"])
def test_uncertain_create_is_never_retried(config, failure):
    calls = []
    def handler(request):
        if request.method == "GET":
            return catalog_response(request)
        calls.append(request)
        if failure == "timeout":
            raise httpx.ReadTimeout("request-secret-must-not-leak", request=request)
        if failure == "empty_receipt":
            return httpx.Response(204)
        return httpx.Response(500 if failure == "server_error" else 201, text="broken")
    with pytest.raises(RequestUncertain):
        mock_adapter(config, handler).create(choice(), "run1", "owner1")
    assert len(calls) == 1


def test_allocated_over_budget_pod_is_returned_for_durable_cleanup(config):
    def handler(request):
        return catalog_response(request) or httpx.Response(201, json=pod(cost=3.0))
    result = mock_adapter(config, handler).create(choice(), "run1", "owner1")
    assert result["id"] == "owned1"
    assert result["allocation_valid"] is False
    assert "ceiling" in result["allocation_error"]


def test_find_owned_follows_pagination_and_ignores_manual_pods(config):
    def handler(request):
        if "cursor" not in request.url.params:
            return httpx.Response(200, json={"pods": [pod(id="manual1", name="manual")],
                "pagination": {"hasNextPage": True, "nextCursor": "second"}})
        assert request.url.params["cursor"] == "second"
        return httpx.Response(200, json={"pods": [pod()], "pagination": {"hasNextPage": False, "nextCursor": None}})
    assert mock_adapter(config, handler).find_owned("run1", "owner1")["id"] == "owned1"


@pytest.mark.parametrize("changed", [{"name": "manual"}, {"env": {}},
                                      {"mounts": {"network": [{"volumeId": "other", "path": "/workspace"}]}}])
def test_termination_requires_all_ownership_checks(config, changed):
    def handler(request):
        assert request.method == "GET"
        return httpx.Response(200, json=pod(**changed))
    with pytest.raises(ValueError, match="ownership"):
        mock_adapter(config, handler).terminate("owned1", "run1", "owner1")


def test_termination_verifies_absence_and_is_idempotent(config):
    deleted = []
    def handler(request):
        if request.method == "DELETE":
            deleted.append(True)
            return httpx.Response(204)
        return httpx.Response(404) if deleted else httpx.Response(200, json=pod())
    adapter = mock_adapter(config, handler)
    assert adapter.terminate("owned1", "run1", "owner1")
    assert adapter.terminate("owned1", "run1", "owner1")
    assert len(deleted) == 1


def test_delete_ack_without_absence_is_not_reported_complete(config):
    def handler(request):
        return httpx.Response(204) if request.method == "DELETE" else httpx.Response(200, json=pod())
    assert mock_adapter(config, handler).terminate("owned1", "run1", "owner1") is False


def test_ambiguous_owned_pods_block_reconciliation(config):
    adapter = mock_adapter(config, lambda _: httpx.Response(200, json={
        "pods": [pod(), pod(id="owned2")], "pagination": {"hasNextPage": False}}))
    with pytest.raises(RequestUncertain, match="Multiple"):
        adapter.find_owned("run1", "owner1")


def test_incomplete_listing_cannot_prove_no_owned_pod(config):
    adapter = mock_adapter(config, lambda _: httpx.Response(200, json={"pods": []}))
    with pytest.raises(RunPodUnavailable, match="ownership remains uncertain"):
        adapter.find_owned("run1", "owner1")


def test_malformed_allocation_retains_pod_id_for_durable_cleanup(config):
    def handler(request):
        return catalog_response(request) or httpx.Response(201, json=pod(gpu=None, env=[], mounts=None))
    result = mock_adapter(config, handler).create(choice(), "run1", "owner1")
    assert result["id"] == "owned1"
    assert result["allocation_valid"] is False


def test_provider_error_never_echoes_credentials(config):
    adapter = mock_adapter(config, lambda _: httpx.Response(401, text="synthetic-private-key"))
    with pytest.raises(RunPodAuthentication) as caught:
        adapter.get("owned1")
    assert "synthetic-private-key" not in str(caught.value)


def test_manual_mode_does_not_create(config):
    config.runpod_managed = False
    adapter = mock_adapter(config, lambda _: pytest.fail("must not contact provider"))
    with pytest.raises(ValueError, match="managed"):
        adapter.create(choice(), "run1", "owner1")


def test_v2_config_is_supported(config):
    from pathlib import Path

    path = Path(config.runpod_config_folder) / "gpu-session.json"
    payload = {"image": "image:tag", "cmd": ["python", "/workspace/start.py"], "disk": 30,
               "gpu": {"id": "GPU A", "count": 1, "minRamPerGpu": 96}, "cloud": "SECURE",
               "dataCenterIds": ["EU-RO-1"], "ports": ["8189/http"], "mounts": copy.deepcopy(pod()["mounts"])}
    path.write_text(json.dumps({"payload": payload}))
    assert mock_adapter(config, lambda _: None).configuration()["min_ram_per_gpu"] == 96
