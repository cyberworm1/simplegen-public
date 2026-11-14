"""Phase 7a: every pod and volume call uses RunPod REST v2; callers keep v1 names."""

import copy
import json
from importlib.resources import files
from threading import Event

import httpx
import pytest

from simplegen_runpod import provider as provider_module
from simplegen_runpod.global_gpu import GlobalGPUProvider
from simplegen_runpod.provider import (
    ProviderError,
    RunPod,
    create_body_from_v1,
    gpu_is_stopped,
    pod_from_v2,
)

V2 = "https://api.runpod.io/v2"


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv("SIMPLEGEN_HOME", str(tmp_path / "simplegen-home"))
    monkeypatch.setattr(provider_module.time, "sleep", lambda _seconds: None)


def client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def v2_pod(pod_id, name, status="RUNNING", cost=1.5, **extra):
    return {"id": pod_id, "name": name, "status": status, "cost": cost,
            "dataCenterId": "EU-RO-1",
            "mounts": {"network": [{"volumeId": "YOUR_NETWORK_VOLUME_ID", "path": "/workspace"}]}, **extra}


def paged(pages):
    """Serve GET /v2/pods pages keyed by cursor; record every request."""
    seen = []

    def handler(request):
        seen.append(request)
        assert str(request.url).startswith(V2 + "/pods?")
        cursor = request.url.params.get("cursor")
        index = 0 if cursor is None else int(cursor)
        last = index == len(pages) - 1
        return httpx.Response(200, json={"pods": pages[index], "pagination": {
            "hasNextPage": not last, "nextCursor": None if last else str(index + 1)}})
    return handler, seen


@pytest.mark.parametrize(("status", "desired", "stopped"), [
    ("PROVISIONING", "RUNNING", False),
    ("STARTING", "RUNNING", False),
    ("RUNNING", "RUNNING", False),
    ("EXITED", "EXITED", True),
    ("TERMINATED", "TERMINATED", False),
    ("ERROR", "ERROR", False),
    ("SOMETHING_NEW", "SOMETHING_NEW", False),
    (None, "UNKNOWN", False),
])
def test_every_v2_status_has_a_deliberate_v1_meaning(status, desired, stopped):
    pod = pod_from_v2(v2_pod("p1", "sg-x", status=status))
    assert pod["desiredStatus"] == desired
    assert gpu_is_stopped(pod) is stopped
    assert pod["status"] == status  # v2 fields are kept as they came.


@pytest.mark.parametrize("status", ["EXITED", "TERMINATED", "RUNNING"])
def test_zero_or_missing_cost_is_unknown_never_free(status):
    assert "costPerHr" not in pod_from_v2(v2_pod("p1", "sg-x", status=status, cost=0.0))
    missing = v2_pod("p1", "sg-x", status=status)
    del missing["cost"]
    assert "costPerHr" not in pod_from_v2(missing)
    assert pod_from_v2(v2_pod("p1", "sg-x", status=status, cost=2.25))["costPerHr"] == 2.25


def test_volume_and_region_use_the_names_the_manager_reads():
    pod = pod_from_v2(v2_pod("p1", "sg-x"))
    assert pod["networkVolumeId"] == "YOUR_NETWORK_VOLUME_ID"
    assert pod["machine"] == {"dataCenterId": "EU-RO-1"}
    bare = pod_from_v2({"id": "p2", "name": "sg-y", "status": "RUNNING"})
    assert "networkVolumeId" not in bare and "machine" not in bare


def test_pod_list_reads_every_page_and_finds_a_pod_on_page_two():
    handler, seen = paged([
        [v2_pod("a", "someone-else")],
        [v2_pod("b", "sg-session-gpu", status="EXITED", cost=0.0)],
    ])
    pods = RunPod("private-api", client(handler)).list()
    assert [pod["id"] for pod in pods] == ["a", "b"]
    assert pods[1]["desiredStatus"] == "EXITED" and "costPerHr" not in pods[1]
    assert [r.url.params.get("cursor") for r in seen] == [None, "1"]
    assert all(r.url.params["limit"] == "100" for r in seen)
    assert all(r.headers["Authorization"] == "Bearer private-api" for r in seen)


def test_repeated_or_invalid_cursor_fails_closed():
    def handler(_request):
        return httpx.Response(200, json={"pods": [], "pagination": {
            "hasNextPage": True, "nextCursor": "same"}})
    with pytest.raises(ProviderError, match="pagination"):
        RunPod("k", client(handler)).list()
    with pytest.raises(ProviderError, match="pod-list"):
        RunPod("k", client(lambda _r: httpx.Response(200, json=[]))).list()


def test_get_uses_v2_detail_and_404_is_absence():
    urls = []

    def handler(request):
        urls.append(str(request.url))
        if request.url.path.endswith("/gone"):
            return httpx.Response(404, json={"title": "Not Found", "status": 404, "detail": "pod not found"})
        return httpx.Response(200, json=v2_pod("p1", "sg-x"))
    api = RunPod("k", client(handler))
    assert api.get("p1")["desiredStatus"] == "RUNNING"
    assert api.get("gone") is None
    assert urls[0] == V2 + "/pods/p1"


@pytest.mark.parametrize("code", [400, 401, 403, 404, 422])
def test_v2_error_format_maps_to_status_without_leaking_detail(code):
    calls = []

    def handler(_request):
        calls.append(1)
        return httpx.Response(code, json={"title": "x", "status": code, "detail": "private provider detail"})
    with pytest.raises(ProviderError) as exc:
        RunPod("k", client(handler)).volume("vol1")
    assert exc.value.status == code and "private" not in str(exc.value)
    assert len(calls) == 1  # Client errors are not retried.


def test_read_429_waits_for_retry_after_capped(monkeypatch):
    waits, responses = [], [
        httpx.Response(429, headers={"Retry-After": "7"}),
        httpx.Response(429, headers={"Retry-After": "900"}),
        httpx.Response(200, json={"id": "vol1", "dataCenter": "EU-RO-1", "name": "v", "size": 1, "type": "STANDARD"}),
    ]
    monkeypatch.setattr(provider_module.time, "sleep", waits.append)
    volume = RunPod("k", client(lambda _r: responses.pop(0))).volume("vol1")
    assert waits == [7.0, provider_module.MAX_RETRY_AFTER_SECONDS]
    assert volume["dataCenterId"] == "EU-RO-1"


def bundled_config():
    return json.loads(files("simplegen").joinpath("data/runpod-config.json").read_text())


def test_saved_v1_gpu_payload_becomes_a_v2_create_body():
    payload = copy.deepcopy(bundled_config()["gpu_payload"])
    payload.update(name="sg-s1-gpu", dockerStartCmd=["python", "/workspace/comfy-h3/gpu_boot.py"])
    assert create_body_from_v1(payload) == {
        "name": "sg-s1-gpu", "image": payload["imageName"],
        "cmd": ["python", "/workspace/comfy-h3/gpu_boot.py"], "disk": 20,
        "ports": ["8189/http"], "cloud": "SECURE", "dataCenterIds": ["EU-RO-1"],
        "mounts": {"network": [{"volumeId": "YOUR_NETWORK_VOLUME_ID", "path": "/workspace"}]},
        "gpu": {"id": "NVIDIA RTX PRO 6000 Blackwell Server Edition", "count": 1,
                "minRamPerGpu": 128, "allowedCudaVersions": ["13.0"]},
    }


def test_saved_v1_cpu_payload_becomes_a_v2_create_body():
    payload = copy.deepcopy(bundled_config()["cpu_payload"])
    payload.update(name="sg-s1-cpu", dockerStartCmd=["simplegen-runpod", "manager"], env={"A": "b"})
    body = create_body_from_v1(payload)
    assert body["cpu"] == {"id": "cpu3c", "vcpuCount": 2}
    assert "gpu" not in body and "mounts" not in body and body["env"] == {"A": "b"}


@pytest.mark.parametrize("change", [
    {"interruptible": True}, {"supportPublicIp": True}, {"volumeInGb": 10},
    {"somethingNew": 1}, {"computeType": "TPU"}, {"gpuTypeIds": ["A", "B"]},
])
def test_unmappable_create_is_refused_before_any_request(change):
    payload = {**bundled_config()["gpu_payload"], "name": "sg-s1-gpu", **change}
    calls = []
    api = RunPod("k", client(lambda r: calls.append(r) or httpx.Response(201, json={})))
    with pytest.raises(ProviderError) as exc:
        api.create(payload)
    assert calls == [] and exc.value.status == 422  # Certain: nothing was created.


def test_create_is_one_post_even_on_5xx_or_429():
    for code in (503, 429):
        calls = []

        def handler(request, calls=calls, code=code):
            calls.append(request)
            return httpx.Response(code)
        payload = {**bundled_config()["gpu_payload"], "name": "sg-s1-gpu"}
        with pytest.raises(ProviderError) as exc:
            RunPod("k", client(handler)).create(payload)
        assert len(calls) == 1 and exc.value.status == code
        assert calls[0].method == "POST" and str(calls[0].url) == V2 + "/pods"


def test_create_returns_the_v1_view_of_the_new_pod():
    payload = {**bundled_config()["gpu_payload"], "name": "sg-s1-gpu"}
    pod = RunPod("k", client(lambda _r: httpx.Response(201, json=v2_pod("new1", "sg-s1-gpu")))).create(payload)
    assert pod["id"] == "new1" and pod["costPerHr"] == 1.5 and pod["networkVolumeId"] == "YOUR_NETWORK_VOLUME_ID"


def test_delete_uses_v2_and_treats_404_as_already_gone():
    seen = []

    def handler(request):
        seen.append((request.method, str(request.url)))
        return httpx.Response(204 if request.url.path.endswith("/p1") else 404)
    api = RunPod("k", client(handler))
    api.delete("p1")
    api.delete("p2")
    assert seen == [("DELETE", V2 + "/pods/p1"), ("DELETE", V2 + "/pods/p2")]


# Global Volume sessions: create stays GraphQL; find, get and delete are v2.

def test_global_reconcile_reads_every_page():
    handler, seen = paged([[v2_pod("a", "other")], [v2_pod("owned", "sg-owned-gpu", cost=2.0)]])
    api = GlobalGPUProvider("private-api", client(handler))
    assert api.reconcile_exact_name("sg-owned-gpu") == {"id": "owned", "name": "sg-owned-gpu", "costPerHr": 2.0}
    assert len(seen) == 2


def test_global_reconcile_keeps_identity_when_cost_is_unknown():
    handler, _ = paged([[v2_pod("owned", "sg-owned-gpu", status="EXITED", cost=0.0)]])
    assert GlobalGPUProvider("k", client(handler)).reconcile_exact_name("sg-owned-gpu") == {
        "id": "owned", "name": "sg-owned-gpu"}


def test_global_reconcile_refuses_duplicates_across_pages():
    handler, _ = paged([[v2_pod("a", "sg-owned-gpu")], [v2_pod("b", "sg-owned-gpu")]])
    with pytest.raises(ProviderError, match="Multiple pods"):
        GlobalGPUProvider("k", client(handler)).reconcile_exact_name("sg-owned-gpu")


def test_global_reconcile_fails_closed_on_bad_pages():
    with pytest.raises(ProviderError, match="reconciliation failed"):
        GlobalGPUProvider("k", client(lambda _r: httpx.Response(500))).reconcile_exact_name("sg-owned-gpu")


def test_global_get_and_delete_use_v2():
    seen = []

    def handler(request):
        seen.append((request.method, str(request.url)))
        if request.method == "DELETE":
            return httpx.Response(204)
        return httpx.Response(200, json=v2_pod("p1", "sg-owned-gpu", status="EXITED", cost=0.0))
    api = GlobalGPUProvider("k", client(handler))
    pod = api.get("p1")
    assert gpu_is_stopped(pod) and "costPerHr" not in pod
    api.delete("p1")
    assert seen == [("GET", V2 + "/pods/p1"), ("DELETE", V2 + "/pods/p1")]


def test_global_session_status_keeps_accepted_quote_when_reconciled_cost_unknown(tmp_path, monkeypatch):
    from simplegen import global_session as module
    from simplegen.global_session import GlobalSession
    from simplegen.settings import Settings

    profile = tmp_path / "profile.json"
    profile.write_text(json.dumps({**json.loads(files("simplegen").joinpath("data/global-runpod-profile.json").read_text()),
                     "template_id": "tmpl-test", "volume_id": "vol-test"}))
    settings = Settings(cloud=True, managed_runpod=True, global_desktop=True,
                        runpod_config_file=str(profile))
    monkeypatch.setattr(module, "_secret", lambda *_: "private-api")
    state = {"run_id": "job", "managed_session": {
        "backend": "global_desktop", "name": "sg-owned-gpu", "create_attempted": True,
        "quoted_hourly": 1.25}}

    class Provider:
        def reconcile_exact_name(self, name):
            return {"id": "ownedpod", "name": name}
        def get(self, pod_id):
            return {"id": pod_id, "name": "sg-owned-gpu", "desiredStatus": "EXITED"}

    session = GlobalSession(settings, tmp_path, state, lambda: None, lambda _: None, Event(), Event())
    session._provider = Provider()
    session.status()
    saved = state["managed_session"]
    assert saved["pod_id"] == "ownedpod" and "actual_hourly" not in saved
    assert saved["quoted_gpu_hourly_usd"] == 1.25 and saved["phase"] == "stopped"
