import copy
import hashlib
import io
import json
from dataclasses import replace
from threading import Event

import pytest

from simplegen.lifecycle import Detached
from simplegen.managed import ManagedSession
from simplegen.settings import Settings
from simplegen_runpod.control import new_session
from simplegen_runpod.outputs import digest


class Provider:
    def __init__(self):
        self.calls = []
        self.pods = {}
        self.rate = .2

    def quote(self, payload):
        self.calls.append(("quote", copy.deepcopy(payload)))
        return self.rate

    def get(self, pod_id):
        self.calls.append(("get", pod_id))
        return self.pods.get(pod_id)


class Ledger:
    def __init__(self, remote):
        self.remote = remote
        self.reloads = 0

    def reload(self):
        self.reloads += 1

    def get(self, sid):
        assert self.remote["id"] == sid
        return copy.deepcopy(self.remote)


class API:
    def __init__(self, remote):
        self.remote = remote
        self.calls = []
        self.failed = False

    def status(self, sid):
        self.calls.append(("status", sid))
        if self.failed:
            raise RuntimeError("unavailable")
        return copy.deepcopy(self.remote)

    def begin_job(self, sid, job_id):
        self.calls.append(("begin", job_id))
        if self.failed:
            raise RuntimeError("uncertain")
        return {**self.remote, "submission_allowed": True}

    def bind_prompt(self, sid, job_id, prompt_id):
        self.calls.append(("bind", prompt_id))
        return self.remote

    def complete(self, sid, **receipt):
        self.calls.append(("complete", receipt))
        self.remote.update(downloads_verified=True, download_receipt={
            "manifest_sha256": receipt["manifest_sha256"],
            "files": sorted(receipt["receipts"], key=lambda r: r["id"]),
        })
        return self.remote

    def cancel(self, sid, request_id):
        self.calls.append(("cancel", request_id))
        return self.remote


@pytest.fixture
def session(tmp_path, monkeypatch):
    monkeypatch.setenv("SIMPLEGEN_HOME", str(tmp_path / "home"))
    shared = {"cloudType": "SECURE", "volumeInGb": 0, "containerDiskInGb": 10,
                  "imageName": "test@sha256:" + "a" * 64, "supportPublicIp": False,
                  "dataCenterIds": ["EU-RO-1"]}
    config = {"volume_id": "volume1", "region": "EU-RO-1", "cpu_max_hourly": .2,
                  "gpu_max_hourly": 1, "allow_provisioning": True,
                  "cpu_payload": {**shared, "computeType": "CPU", "cpuFlavorIds": ["cpu3c"],
                               "vcpuCount": 2, "ports": ["8097/http"]},
                  "gpu_payload": {**shared, "computeType": "GPU", "gpuTypeIds": ["GPU"],
                               "gpuCount": 1, "ports": ["8189/http"],
                               "networkVolumeId": "volume1", "volumeMountPath": "/workspace"}}
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    schema = tmp_path / "schema.json"
    schema.write_text(json.dumps({"TestNode": {"input": {"required": {}}}}))
    settings = replace(Settings(), managed_runpod=True, cloud=True,
                       runpod_config_file=str(path), runpod_schema_file=str(schema))
    state, writes = {"run_id": "a" * 32}, []
    session = ManagedSession(settings, tmp_path / "project", state,
                             lambda: writes.append(copy.deepcopy(state)), lambda event: None,
                             Event(), Event())
    session._credentials = {"RUNPOD_API_KEY": "private-api", "SG_ADMIN_TOKEN": "private-admin" * 4,
                            "SG_COMFY_USERNAME": "private-user", "SG_COMFY_PASSWORD": "private-password",
                            "SG_S3_ACCESS_KEY": "private-s3", "SG_S3_SECRET_KEY": "private-s3-secret"}
    session._provider = Provider()
    sid = "b" * 32
    remote = new_session(sid, session.config(), 1)
    remote.update(phase="ready", activation_requested=True,
                  control_url="https://cpu1-8097.proxy.runpod.net")
    remote["resources"]["cpu"].update(pod_id="cpu1", state="active", observed_present=True)
    remote["resources"]["gpu"].update(pod_id="gpu1", state="active", observed_present=True)
    session._ledger = Ledger(remote)
    session._api = API(remote)
    session._s3 = object()
    session.test_writes = writes
    return session


def existing(session):
    remote = session._ledger.remote
    session.saved.update(sid=remote["id"], config=remote["config"],
                         control_url=remote["control_url"], output_prefix="comfy-h3/ComfyUI/output")
    return remote


def test_schema_frozen_before_cloud_and_tamper_rejected(session):
    result = session.schema()
    assert "TestNode" in result
    assert session._provider.calls == []
    assert session.saved["schema_sha256"]
    (session.folder / "managed-object-info.json").write_text("{}")
    with pytest.raises(ValueError, match="changed"):
        session.schema()


def test_new_regional_cpu_session_is_deprecated_before_any_paid_action(session):
    with pytest.raises(ValueError, match="deprecated"):
        session.connect()
    assert "sid" not in session.saved
    assert session._provider.calls == []


def test_choice_does_not_reenable_deprecated_cpu_bootstrap(session):
    session._provider.rate = .9
    with pytest.raises(ValueError, match="deprecated"):
        session.connect({"gpu_id": "GPU", "gpu_count": 1, "hourly_usd": .5})
    assert "sid" not in session.saved


def test_saved_regional_session_can_reconnect_without_bootstrap(session):
    existing(session)
    endpoint = session.connect()
    assert endpoint.comfy_url == "https://gpu1-8189.proxy.runpod.net"
    assert endpoint.browser_login_file.startswith(str(session.folder.parent / "home"))
    serialized = json.dumps(session.test_writes)
    assert all(value not in serialized for value in session._credentials.values())


def test_saved_regional_session_without_controller_url_never_bootstraps(session):
    existing(session)
    session.saved.pop("control_url")
    with pytest.raises(RuntimeError, match="deprecated"):
        session.connect()


def test_submission_uncertainty_never_repeats_permission_or_prompt_post(session):
    existing(session)
    session._api.failed = True
    with pytest.raises(RuntimeError):
        session.before_submit()
    assert session.saved["submission_intent"] is True
    assert session.before_submit() is False
    assert [op for op, _ in session._api.calls] == ["begin"]


def test_sequence_job_intents_are_independent_and_durable(session):
    existing(session)
    assert session.before_submit("variation1") is True
    session.bind("prompt1", "variation1")
    assert session.before_submit("variation1") is False
    assert session.before_submit("variation2") is True
    session.bind("prompt2", "variation2")
    assert session.saved["job_intents"]["variation1"]["bind_prompt_id"] == "prompt1"
    assert session.saved["job_intents"]["variation2"]["bind_prompt_id"] == "prompt2"
    with pytest.raises(ValueError, match="different ComfyUI job"):
        session.bind("wrong", "variation1")
    assert not session.saved.get("completion_intent")


def test_detach_during_connection_does_not_cancel_or_start(session):
    existing(session)
    session.detach.set()
    with pytest.raises(Detached):
        session.connect()
    assert session._api.calls == []
    assert session._provider.calls == []


def output(session, tmp_path):
    remote = existing(session)
    item = {"prompt_id": "prompt1", "filename": "final.mp4", "subfolder": "simplegen/run1", "type": "output"}
    item["id"] = digest(item)
    remote["manifest"] = [item]
    remote["manifest_sha256"] = digest([item])
    remote["jobs"][session.state["run_id"]] = {"prompt_id": "prompt1", "status": "success"}
    target = tmp_path / "final.mp4"
    target.write_bytes(b"finished-video")
    receipt = {**item, "path": str(target), "size_bytes": target.stat().st_size,
               "sha256": hashlib.sha256(target.read_bytes()).hexdigest()}
    return remote, receipt


def test_completion_receipt_maps_manifest_and_retries_without_changed_data(session, tmp_path):
    remote, receipt = output(session, tmp_path)
    session.complete([receipt])
    session.complete([receipt])
    assert sum(op == "complete" for op, _ in session._api.calls) == 1
    assert session.saved["completion_acknowledged"] is True
    assert session.saved["completion_intent"]["receipts"][0]["id"] == remote["manifest"][0]["id"]


def test_completion_rejects_changed_file_and_unrelated_video(session, tmp_path):
    _, receipt = output(session, tmp_path)
    with pytest.raises(ValueError, match="exactly once"):
        session.complete([{**receipt, "filename": "other.mp4"}])
    with pytest.raises(ValueError, match="changed"):
        session.complete([{**receipt, "sha256": "0" * 64}])
    assert not any(op == "complete" for op, _ in session._api.calls)


def test_retired_cpu_status_and_bind_use_read_only_provider_and_journal(session, tmp_path):
    _, receipt = output(session, tmp_path)
    session._api.failed = True
    endpoint = session.connect()
    assert endpoint.comfy_url.endswith("gpu1-8189.proxy.runpod.net")
    assert session.retired is True
    session.bind("prompt1")
    session.complete([receipt])
    assert session.saved["completion_acknowledged"] is False
    assert session.saved["completion_disposition"] == "saved_locally_after_compute_retired"
    assert all(op == "status" for op, _ in session._api.calls)
    assert all(op == "get" for op, _ in session._provider.calls)
    assert session._ledger.reloads > 0


def test_cancel_is_explicit_idempotent_and_not_automatic_on_close(session):
    existing(session)
    session.shutdown()
    session.shutdown()
    requests = [value for op, value in session._api.calls if op == "cancel"]
    assert len(requests) == 2 and requests[0] == requests[1]


def test_s3_download_checks_manifest_and_complete_length(session, tmp_path):
    _, receipt = output(session, tmp_path)
    class S3:
        def get_object(self, **kwargs):
            assert kwargs == {"Bucket": "volume1", "Key": "comfy-h3/ComfyUI/output/simplegen/run1/final.mp4"}
            return {"Body": io.BytesIO(b"finished-video"), "ContentLength": 14}
    session._s3 = S3()
    destination = tmp_path / "downloaded.mp4"
    result = session.download(receipt, destination, Event(), lambda got, total: None)
    assert result["sha256"] == receipt["sha256"]
    assert destination.read_bytes() == b"finished-video"
    with pytest.raises(ValueError, match="safe"):
        session.download({**receipt, "subfolder": "../outside"}, destination, Event(), lambda *a: None)


def test_partial_s3_download_keeps_previous_local_file(session, tmp_path):
    _, receipt = output(session, tmp_path)
    class S3:
        def get_object(self, **kwargs):
            return {"Body": io.BytesIO(b"partial"), "ContentLength": 100}
    session._s3 = S3()
    destination = tmp_path / "existing.mp4"
    destination.write_bytes(b"prior")
    with pytest.raises(ValueError, match="incomplete"):
        session.download(receipt, destination, Event(), lambda *a: None)
    assert destination.read_bytes() == b"prior"
    assert not destination.with_name("existing.mp4.part").exists()
