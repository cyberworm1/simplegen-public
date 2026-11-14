import json
from importlib.resources import files

import httpx
import pytest

from simplegen_runpod.global_gpu import (
    CreateUncertain,
    GlobalGPUProfile,
    GlobalGPUProvider,
    load_global_profile,
)
from simplegen_runpod.provider import ProviderError


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv("SIMPLEGEN_HOME", str(tmp_path / "simplegen-home"))


def test_graphql_payload_attaches_global_volume_and_keeps_template_startup():
    profile = GlobalGPUProfile(template_id="tmpl-test", volume_id="vol-test")
    payload = profile.deployment_input(name="sg-owned-gpu", gpu_type_id="NVIDIA RTX PRO 6000 Blackwell Server Edition")
    assert payload == {
        "name": "sg-owned-gpu", "templateId": "tmpl-test",
        "gpuTypeId": "NVIDIA RTX PRO 6000 Blackwell Server Edition", "gpuCount": 1,
        "minCudaVersion": "13.0",
        "volumeMounts": [{"volumeId": "vol-test",
                          "volumeType": "OBJECT_STORE_VOLUME", "mountPath": "/workspace-global"}],
        "containerDiskInGb": 200, "volumeInGb": 0, "cloudType": "SECURE",
    }
    assert "dockerStartCmd" not in payload and "networkVolumeId" not in payload
    assert profile.deployment_input(name="sg-owned-gpu", gpu_type_id="GPU A",
                                    data_center_id="US-TX-1")["dataCenterId"] == "US-TX-1"


def test_packaged_global_profile_uses_200_gb_for_new_sessions(tmp_path):
    source = files("simplegen").joinpath("data/global-runpod-profile.json")
    profile_path = tmp_path / "global-profile.json"
    packaged = json.loads(source.read_text(encoding="utf-8"))
    # The packaged profile names no account resources: the user enters their own.
    assert packaged["template_id"] == "" and packaged["volume_id"] == ""
    profile_path.write_text(json.dumps(packaged), encoding="utf-8")
    with pytest.raises(ValueError, match="template ID and Global Volume ID"):
        load_global_profile(profile_path)
    profile_path.write_text(json.dumps({**packaged, "template_id": "tmpl-test", "volume_id": "vol-test"}),
                            encoding="utf-8")
    assert load_global_profile(profile_path).container_disk_gb == 200


@pytest.mark.parametrize("field", ["template_id", "volume_id"])
@pytest.mark.parametrize("value", ["", "has space", "x" * 101, "../etc"])
def test_profile_ids_must_be_plain_identifiers(field, value):
    with pytest.raises(ValueError, match="template ID and Global Volume ID"):
        GlobalGPUProfile(**{"template_id": "tmpl-test", "volume_id": "vol-test", field: value}).deployment_input(
            name="sg-owned-gpu", gpu_type_id="GPU A")


def test_deploy_one_post_and_sanitized_uncertain_response():
    requests = []
    def handler(request):
        requests.append(request)
        assert request.headers["User-Agent"] == "Mozilla/5.0"
        assert request.url == "https://api.runpod.io/graphql"
        assert json.loads(request.content)["variables"]["input"]["templateId"] == "tmpl-test"
        return httpx.Response(503, text="private provider detail")

    provider = GlobalGPUProvider("private-api", httpx.Client(transport=httpx.MockTransport(handler)))
    with pytest.raises(CreateUncertain, match="reconcile") as exc:
        provider.deploy(GlobalGPUProfile(template_id="tmpl-test", volume_id="vol-test"), name="sg-owned-gpu", gpu_type_id="GPU A")
    assert len(requests) == 1
    assert "private" not in str(exc.value)


def test_reconcile_exact_name_never_adopts_unrelated_pod():
    def handler(request):
        return httpx.Response(200, json={"pods": [
            {"id": "unrelated", "name": "someone-else", "status": "RUNNING", "cost": 1.0},
            {"id": "owned", "name": "sg-owned-gpu", "status": "RUNNING", "cost": 2.0},
        ], "pagination": {"hasNextPage": False, "nextCursor": None}})
    provider = GlobalGPUProvider("private-api", httpx.Client(transport=httpx.MockTransport(handler)))
    assert provider.reconcile_exact_name("sg-owned-gpu") == {
        "id": "owned", "name": "sg-owned-gpu", "costPerHr": 2.0,
    }


def test_graphql_schema_rejection_does_not_expose_provider_error():
    provider = GlobalGPUProvider("private-api", httpx.Client(transport=httpx.MockTransport(
        lambda _: httpx.Response(200, json={"errors": [{"message": "private provider detail"}]})
    )))
    with pytest.raises(ProviderError, match="rejected") as exc:
        provider.deploy(GlobalGPUProfile(template_id="tmpl-test", volume_id="vol-test"), name="sg-owned-gpu", gpu_type_id="GPU A")
    assert not isinstance(exc.value, CreateUncertain)
    assert "private" not in str(exc.value)


def test_global_profile_is_separate_from_legacy_schema(tmp_path):
    path = tmp_path / "global.json"
    path.write_text(json.dumps({
        "mode": "global_desktop", "schema_version": 1,
        "template_id": "tmpl-test", "volume_id": "vol-test",
        "mount_path": "/workspace-global", "container_disk_gb": 200,
        "min_cuda_version": "13.0", "cloud_type": "SECURE", "port": "8189/http",
        "gpu_tiers": ["GPU A"], "gpu_max_hourly": 3.0,
    }))
    assert load_global_profile(path).volume_id == "vol-test"
    stale = json.loads(path.read_text())
    stale["container_disk_gb"] = 150
    path.write_text(json.dumps(stale))
    with pytest.raises(ValueError, match="200 GB"):
        load_global_profile(path)
    assert GlobalGPUProfile(template_id="tmpl-test", volume_id="vol-test", container_disk_gb=150).deployment_input(
        name="sg-owned-gpu", gpu_type_id="GPU A"
    )["containerDiskInGb"] == 150
    path.write_text(json.dumps({"schema_version": 2}))
    with pytest.raises(ValueError, match="global desktop"):
        load_global_profile(path)
