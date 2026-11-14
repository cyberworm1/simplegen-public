import base64
import hashlib
import json
from threading import Event

import httpx
import pytest
import websocket

from simplegen.agents import Cancelled
from simplegen.comfy import Comfy, ComfyRequestError, output_files
from simplegen.settings import Settings, with_runpod_config


def test_browser_login_authenticates_http_and_websocket(monkeypatch, tmp_path):
    login = tmp_path / "browser-login.json"
    login.write_text(json.dumps({"username": "viewer", "password": "test-secret"}))
    settings = Settings(comfy_url="https://pod.example.test", browser_login_file=str(login))
    expected = "Basic " + base64.b64encode(b"viewer:test-secret").decode()
    monkeypatch.setenv(settings.token_env, "ignored-bearer-token")
    polls = 0

    def handler(request):
        nonlocal polls
        assert request.headers["Authorization"] == expected
        polls += 1
        return httpx.Response(
            200, json={} if polls == 1 else {"mine": {"status": {"completed": True}}}
        )

    connected = []

    class Socket:
        def recv(self):
            return json.dumps(
                {"type": "progress", "data": {"prompt_id": "mine", "value": 1, "max": 2}}
            )

        def close(self):
            pass

    def connect(url, **kwargs):
        assert url.startswith("wss://pod.example.test/ws?")
        assert kwargs["header"]["Authorization"] == expected
        connected.append(url)
        return Socket()

    ticks = iter([0, 3, 6])
    monkeypatch.setattr("simplegen.comfy.time.monotonic", lambda: next(ticks))
    monkeypatch.setattr("simplegen.comfy.websocket.create_connection", connect)
    comfy = Comfy(settings, httpx.Client(transport=httpx.MockTransport(handler)))
    comfy.monitor("mine", "client", {"1": {}}, Event(), lambda _: None)
    assert connected
    assert "test-secret" not in json.dumps(settings.public_dict())
    comfy.close()


@pytest.mark.parametrize("contents", ['{"username":"viewer"}', "not-json"])
def test_invalid_browser_login_fails_without_exposing_contents(tmp_path, contents):
    login = tmp_path / "login.json"
    login.write_text(contents)
    with pytest.raises(ValueError, match="Cannot read browser login") as exc:
        Comfy(Settings(comfy_url="https://pod.example.test", browser_login_file=str(login)))
    assert contents not in str(exc.value)


def test_browser_credentials_require_https():
    with pytest.raises(ValueError, match="HTTPS"):
        Comfy(Settings(browser_login_file="login.json"))


def test_runpod_import_rejects_unrelated_url(tmp_path):
    (tmp_path / "gpu-session.json").write_text(
        json.dumps({"pod_id": "example", "url": "https://unrelated.example.test"})
    )
    with pytest.raises(ValueError, match="valid gpu-session"):
        with_runpod_config(Settings(), tmp_path)


def client(handler):
    return Comfy(Settings(), httpx.Client(transport=httpx.MockTransport(handler)))


def test_download_is_atomic_and_checksummed(tmp_path):
    payload = b"synthetic media"
    comfy = client(lambda request: httpx.Response(200, content=payload))
    target = tmp_path / "final.mp4"
    receipt = comfy.download({"filename": "clip.mp4"}, target, Event())
    assert target.read_bytes() == payload
    assert receipt["sha256"] == hashlib.sha256(payload).hexdigest()
    assert not target.with_suffix(".mp4.part").exists()


def test_cancelled_download_does_not_replace_final(tmp_path):
    target = tmp_path / "clip.mp4"
    target.write_bytes(b"existing")
    cancel = Event()
    cancel.set()
    with pytest.raises(Cancelled):
        client(lambda _: httpx.Response(200, content=b"other")).download(
            {"filename": "x.mp4"}, target, cancel
        )
    assert target.read_bytes() == b"existing"


def test_short_download_is_not_final(tmp_path):
    comfy = client(
        lambda _: httpx.Response(200, content=b"short", headers={"content-length": "100"})
    )
    target = tmp_path / "final.mp4"
    with pytest.raises(RuntimeError, match="Incomplete"):
        comfy.download({"filename": "x.mp4"}, target, Event())
    assert not target.exists()


def test_output_files_support_native_and_vhs_and_ignore_remote_urls():
    entry = {
        "outputs": {
            "7": {"images": [{"filename": "final.mp4", "subfolder": "out", "type": "output"}]},
            "2": {
                "gifs": [{"filename": "shot.mp4", "type": "output", "url": "http://evil.invalid"}]
            },
        }
    }
    result = output_files(entry)
    assert [r["filename"] for r in result] == ["final.mp4", "shot.mp4"]
    assert all("url" not in r for r in result)


@pytest.mark.parametrize(
    "filename,subfolder",
    [
        ("../escape.mp4", ""),
        ("clip.mp4", "../outside"),
        ("clip.mp4", "/outside"),
        ("x\\escape.mp4", ""),
    ],
)
def test_unsafe_output_paths_rejected(filename, subfolder):
    with pytest.raises(ValueError, match="unsafe"):
        output_files(
            {"outputs": {"1": {"videos": [{"filename": filename, "subfolder": subfolder}]}}}
        )


def test_submission_stamps_reconciliation_id():
    def handler(request):
        data = json.loads(request.content)
        assert data["extra_data"]["simplegen_run_id"] == "run"
        assert data["client_id"] == "client"
        return httpx.Response(200, json={"prompt_id": "job"})

    assert client(handler).submit({"1": {}}, "run", "client") == "job"


def test_uncertain_submission_found_in_history():
    def handler(request):
        if request.url.path == "/queue":
            return httpx.Response(200, json={"queue_running": [], "queue_pending": []})
        return httpx.Response(
            200, json={"job": {"prompt": [0, "job", {}, {"simplegen_run_id": "mine"}]}}
        )

    assert client(handler).find_submission("mine") == "job"


def test_targeted_cancel_never_sends_global_interrupt():
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json={"system": {"comfyui_version": "0.36.0"}})

    client(handler).cancel_job("mine")
    assert json.loads(requests[-1].content) == {"prompt_id": "mine"}
    assert json.loads(requests[0].content) == {"delete": ["mine"]}


def test_older_comfy_does_not_receive_unsafe_interrupt():
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json={"system": {"comfyui_version": "0.3.0"}})

    message = client(handler).cancel_job("mine")
    assert not any(request.url.path == "/interrupt" for request in requests)
    assert "may continue" in message


def test_monitor_saved_success_needs_no_websocket():
    def handler(request):
        return httpx.Response(
            200,
            json={"mine": {"status": {"status_str": "success", "completed": True}, "outputs": {}}},
        )

    assert client(handler).monitor("mine", "client", {}, Event(), lambda _: None)["status"][
        "completed"
    ]


def test_monitor_failure_reports_server_message():
    def handler(request):
        return httpx.Response(
            200,
            json={
                "mine": {
                    "status": {
                        "status_str": "error",
                        "messages": [["execution_error", "out of memory"]],
                    }
                }
            },
        )

    with pytest.raises(RuntimeError, match="out of memory"):
        client(handler).monitor("mine", "client", {}, Event(), lambda _: None)


def test_real_progress_filters_other_jobs(monkeypatch):
    polls = 0

    def handler(request):
        nonlocal polls
        polls += 1
        if polls < 5:
            return httpx.Response(200, json={})
        return httpx.Response(200, json={"mine": {"status": {"completed": True}}})

    events = iter(
        [
            {"type": "progress", "data": {"prompt_id": "other", "value": 99, "max": 100}},
            {"type": "progress", "data": {"prompt_id": "mine", "value": 5, "max": 20}},
            {"type": "executed", "data": {"prompt_id": "mine", "node": "1"}},
        ]
    )

    class Socket:
        def recv(self):
            return json.dumps(next(events))

        def close(self):
            pass

    clock = iter(range(0, 100, 3))
    monkeypatch.setattr("simplegen.comfy.time.monotonic", lambda: next(clock))
    monkeypatch.setattr("simplegen.comfy.websocket.create_connection", lambda *a, **k: Socket())
    output = []
    # First last_poll=0 tick does not read history; arrange success before events exhaust.
    polls = 1
    client(handler).monitor("mine", "client", {"1": {}, "2": {}}, Event(), output.append)
    assert all("99" not in e["message"] for e in output)
    assert any("5/20" in e["message"] for e in output)


def test_missing_job_after_restart_fails_without_submission(monkeypatch):
    routes = []

    def handler(request):
        routes.append(request.url.path)
        return httpx.Response(200, json={})

    clock = iter([0, 25])
    monkeypatch.setattr("simplegen.comfy.time.monotonic", lambda: next(clock))
    with pytest.raises(RuntimeError, match="absent"):
        client(handler).monitor("lost", "client", {}, Event(), lambda _: None)
    assert "/prompt" not in routes


@pytest.fixture
def monitoring_clock(monkeypatch):
    """Make status reconciliation immediately due; avoid real sleeps or sockets."""
    ticks = iter([0, *range(25, 200, 3)])
    monkeypatch.setattr("simplegen.comfy.time.monotonic", lambda: next(ticks))

    class Socket:
        def recv(self):
            raise websocket.WebSocketTimeoutException()

        def close(self):
            pass

    monkeypatch.setattr("simplegen.comfy.websocket.create_connection", lambda *a, **k: Socket())


def test_monitor_recovers_queue_502_without_resubmitting(monitoring_clock):
    requests = []
    history_reads = 0
    queue_reads = 0
    success = {"status": {"completed": True}, "outputs": {}}

    def handler(request):
        nonlocal history_reads, queue_reads
        requests.append((request.method, request.url.path))
        if request.url.path == "/queue":
            queue_reads += 1
            if queue_reads == 1:
                return httpx.Response(502, text="Temporary gateway failure")
            return httpx.Response(200, json={"queue_running": [[0, "mine"]]})
        assert request.url.path == "/history/mine"
        history_reads += 1
        return httpx.Response(200, json={"mine": success} if history_reads == 3 else {})

    events = []
    assert client(handler).monitor("mine", "client", {}, Event(), events.append) == success
    assert queue_reads == 2
    assert history_reads == 3
    assert any("retrying (1/5)" in event["message"] for event in events)
    assert all(method == "GET" for method, _ in requests)


def test_monitor_recovers_second_history_timeout_and_returns_success_immediately(monitoring_clock):
    requests = []
    history_reads = 0
    success = {"status": {"completed": True}, "outputs": {}}

    def handler(request):
        nonlocal history_reads
        requests.append((request.method, request.url.path))
        if request.url.path == "/queue":
            return httpx.Response(200, json={"queue_running": [], "queue_pending": []})
        assert request.url.path == "/history/mine"
        history_reads += 1
        if history_reads == 2:
            raise httpx.ReadTimeout("read timed out", request=request)
        if history_reads == 4:
            return httpx.Response(200, json={"mine": success})
        assert history_reads < 4, "Completion from reconciliation must return immediately"
        return httpx.Response(200, json={})

    assert client(handler).monitor("mine", "client", {}, Event(), lambda _: None) == success
    assert history_reads == 4
    assert sum(path == "/queue" for _, path in requests) == 2
    assert all(method == "GET" for method, _ in requests)


@pytest.mark.parametrize("phase", ["history", "queue", "second_history"])
def test_monitor_status_failure_budget_covers_entire_reconciliation(monitoring_clock, phase):
    failures = 0
    history_reads = 0
    requests = []

    def handler(request):
        nonlocal failures, history_reads
        requests.append((request.method, request.url.path))
        if request.url.path == "/history/mine":
            history_reads += 1
        fail = (
            phase == "history"
            or (phase == "queue" and request.url.path == "/queue")
            or (
                phase == "second_history"
                and request.url.path == "/history/mine"
                and history_reads % 2 == 0
            )
        )
        if fail:
            failures += 1
            return httpx.Response(503, text="Temporarily unavailable")
        return httpx.Response(200, json={})

    events = []
    with pytest.raises(RuntimeError, match="Connection lost.*Job ID is saved"):
        client(handler).monitor("mine", "client", {}, Event(), events.append)
    assert failures == 5
    assert len([event for event in events if "retrying" in event["message"]]) == 4
    assert all(method == "GET" for method, _ in requests)
    assert not any(path == "/prompt" for _, path in requests)


@pytest.mark.parametrize("status", [400, 401, 403, 404])
@pytest.mark.parametrize("phase", ["history", "queue", "second_history"])
def test_monitor_permanent_http_failure_is_not_retried(monitoring_clock, phase, status):
    failures = 0
    history_reads = 0

    def handler(request):
        nonlocal failures, history_reads
        assert request.method == "GET"
        if request.url.path == "/history/mine":
            history_reads += 1
        fail = (
            phase == "history"
            or (phase == "queue" and request.url.path == "/queue")
            or (
                phase == "second_history"
                and request.url.path == "/history/mine"
                and history_reads == 2
            )
        )
        if fail:
            failures += 1
            return httpx.Response(status, text="Request denied")
        return httpx.Response(200, json={})

    with pytest.raises(ComfyRequestError, match=f"HTTP {status}") as error:
        client(handler).monitor("mine", "client", {}, Event(), lambda _: None)
    assert not error.value.retryable
    assert failures == 1


@pytest.mark.parametrize("status", [408, 429, 500, 502, 503, 504])
def test_monitor_transient_http_failure_is_retried(monitoring_clock, status):
    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        assert request.method == "GET"
        if calls == 1:
            return httpx.Response(status)
        return httpx.Response(200, json={"mine": {"status": {"completed": True}}})

    assert client(handler).monitor("mine", "client", {}, Event(), lambda _: None)["status"][
        "completed"
    ]
    assert calls == 2


@pytest.mark.parametrize("cancel_after", [1, 5])
def test_monitor_cancellation_during_reconciliation_recovery(monitoring_clock, cancel_after):
    cancel = Event()
    requests = []

    def handler(request):
        requests.append((request.method, request.url.path))
        if request.method == "GET" and request.url.path == "/queue":
            if requests.count(("GET", "/queue")) == cancel_after:
                cancel.set()
            return httpx.Response(502)
        if request.url.path == "/system_stats":
            return httpx.Response(200, json={"system": {"comfyui_version": "0.3.0"}})
        if request.method == "POST":
            assert request.url.path == "/queue"
            assert json.loads(request.content) == {"delete": ["mine"]}
        return httpx.Response(200, json={})

    with pytest.raises(Cancelled, match="Removed this job"):
        client(handler).monitor("mine", "client", {}, cancel, lambda _: None)
    assert requests.count(("GET", "/history/mine")) == cancel_after
    assert requests.count(("GET", "/queue")) == cancel_after
    assert ("POST", "/prompt") not in requests


def test_monitor_second_history_generation_failure_is_immediate(monitoring_clock):
    history_reads = 0

    def handler(request):
        nonlocal history_reads
        assert request.method == "GET"
        if request.url.path == "/history/mine":
            history_reads += 1
            if history_reads == 2:
                return httpx.Response(
                    200,
                    json={
                        "mine": {"status": {"status_str": "error", "messages": ["out of memory"]}}
                    },
                )
        return httpx.Response(200, json={})

    with pytest.raises(RuntimeError, match="Generation failed.*out of memory"):
        client(handler).monitor("mine", "client", {}, Event(), lambda _: None)
    assert history_reads == 2


def test_transient_submission_failure_is_not_retried():
    requests = []

    def handler(request):
        requests.append((request.method, request.url.path))
        return httpx.Response(502)

    with pytest.raises(ComfyRequestError, match="HTTP 502"):
        client(handler).submit({}, "run", "client")
    assert requests == [("POST", "/prompt")]


def test_monitor_failure_budget_resets_after_successful_reconciliation(
    monitoring_clock, monkeypatch
):
    ticks = iter([0, 25, 28, 31, 34, 37, 60, 63, 66, 69, 72])
    monkeypatch.setattr("simplegen.comfy.time.monotonic", lambda: next(ticks))
    queue_reads = 0

    def handler(request):
        nonlocal queue_reads
        assert request.method == "GET"
        if request.url.path == "/queue":
            queue_reads += 1
            if queue_reads in (5, 10):
                return httpx.Response(
                    200, json={"queue_running": [[0, "mine"]]} if queue_reads == 5 else {}
                )
            return httpx.Response(502)
        return httpx.Response(
            200, json={"mine": {"status": {"completed": True}}} if queue_reads == 10 else {}
        )

    assert client(handler).monitor("mine", "client", {}, Event(), lambda _: None)["status"][
        "completed"
    ]
    assert queue_reads == 10
