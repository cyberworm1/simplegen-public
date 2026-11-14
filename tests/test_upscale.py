"""4K60 upscale client: framing, durable records and the host protocol (fake service, no network)."""

import hashlib
import json
import shutil
import subprocess
import uuid
from dataclasses import asdict, replace
from pathlib import Path

import httpx
import pytest

from simplegen import upscale
from simplegen.preflight import run_preflight, summary
from simplegen.settings import Settings

SOURCE_META = {"width": 672, "height": 384, "fps": 24.0, "codec": "h264", "duration": 15.0,
               "has_audio": True, "size_bytes": 10}
OUTPUT_META = {"width": 3840, "height": 2194, "fps": 60.0, "codec": "hevc", "duration": 15.0,
               "has_audio": True, "size_bytes": 20}


class FakeHost:
    """In-memory implementation of the upscale service API."""

    def __init__(self, token="secret"):
        self.token = token
        self.jobs: dict[str, dict] = {}
        self.outputs: dict[str, bytes] = {}
        self.calls: list[tuple[str, str]] = []
        self.down = False
        self.default_preset = "seedvr2-3b+rife"
        self.corrupt_download = False

    def health(self):
        return {"service": "simplegen-upscale", "api": 1, "version": "0.1.0",
                "gpu": {"name": "RTX 3080", "vram_total_mb": 10240, "vram_free_mb": 9000},
                "models": [], "default_preset": self.default_preset,
                "presets": [{"id": "seedvr2-3b+rife", "label": "SeedVR2 3B + RIFE", "ready": True,
                             "seconds_per_output_frame": 2.0},
                            {"id": "ultrasharp+rife", "label": "UltraSharp + RIFE", "ready": True,
                             "seconds_per_output_frame": None}],
                "framings": [k for k, _ in upscale.FRAMINGS], "disk_free_bytes": 10**11,
                "queue": {"running": None, "queued": 0},
                "jobs": [{"id": j["id"], "state": j["state"], "bytes": 5} for j in self.jobs.values()]}

    def finish(self, job_id, output=b"4k60-bytes"):
        job = self.jobs[job_id]
        job.update(state="done", progress=1.0, output={"sha256": hashlib.sha256(output).hexdigest(),
                                                       "bytes": len(output)})
        self.outputs[job_id] = output

    def handler(self, request: httpx.Request) -> httpx.Response:
        if self.down:
            raise httpx.ConnectError("down", request=request)
        path = request.url.path.removeprefix("/v1")
        self.calls.append((request.method, path))
        if request.headers.get("Authorization") != f"Bearer {self.token}":
            return httpx.Response(401)
        if path == "/health":
            return httpx.Response(200, json=self.health())
        parts = path.strip("/").split("/")
        job_id = parts[1]
        job = self.jobs.get(job_id)
        if request.method == "PUT":
            body = request.read()
            params = dict(request.url.params)
            if hashlib.sha256(body).hexdigest() != params["sha256"]:
                return httpx.Response(400, json={"detail": "sha256 mismatch"})
            if job:
                same = (job["source"]["sha256"], job["options"]["preset"], job["options"]["framing"]) == (
                    params["sha256"], params["preset"], params["framing"])
                return httpx.Response(200 if same else 409, json=job)
            job = {"id": job_id, "state": "queued", "stage": None, "progress": 0.0, "eta_seconds": None,
                   "error": None, "options": {"preset": params["preset"], "framing": params["framing"],
                                              "name": params["name"]},
                   "source": {"sha256": params["sha256"], "bytes": len(body)}, "output": None}
            self.jobs[job_id] = job
            return httpx.Response(201, json=job)
        if job is None:
            return httpx.Response(404)
        if request.method == "GET" and len(parts) == 2:
            return httpx.Response(200, json=job)
        if request.method == "GET" and parts[2] == "result":
            if job["state"] != "done":
                return httpx.Response(409)
            data = self.outputs[job_id]
            sha = job["output"]["sha256"]
            if self.corrupt_download:
                data = b"x" + data
            return httpx.Response(200, content=data, headers={"X-Output-SHA256": sha})
        if request.method == "POST" and parts[2] == "cancel":
            if job["state"] in ("queued", "running", "waiting_gpu"):
                job["state"] = "cancelled"
            return httpx.Response(200, json=job)
        if request.method == "DELETE":
            if job["state"] in ("running", "waiting_gpu"):
                return httpx.Response(409)
            del self.jobs[job_id]
            self.outputs.pop(job_id, None)
            return httpx.Response(204)
        return httpx.Response(405)


@pytest.fixture
def host(monkeypatch):
    fake = FakeHost()
    monkeypatch.setenv("SIMPLEGEN_UPSCALE_TOKEN", fake.token)
    return fake


@pytest.fixture
def client(host):
    instance = upscale.Client("http://upscaler:8190", host.token, transport=httpx.MockTransport(host.handler))
    yield instance
    instance.close()


@pytest.fixture
def settings():
    return Settings(upscale=True, upscale_url="http://upscaler:8190")


@pytest.fixture
def probed(monkeypatch):
    """ffprobe stand-in: the source and output metadata the verification sees."""
    seen = {"output": dict(OUTPUT_META)}

    def probe(path):
        return dict(SOURCE_META) if "4k60" not in str(path) else dict(seen["output"])
    monkeypatch.setattr(upscale, "probe", probe)
    return seen


def make_run(root: Path, *, upscale_on=True, aspect="16:9", name="run") -> Path:
    folder = root / name
    (folder / "final").mkdir(parents=True)
    final = folder / "final" / "abcd1234_final.mp4"
    final.write_bytes(b"source-video-" + name.encode())
    state = {"schema_version": 5, "run_id": str(uuid.uuid4()), "concept": "A fox in the woods",
             "status": "complete", "settings": asdict(Settings(upscale=upscale_on, aspect_ratio=aspect)),
             "result": {"folder": ".", "final_path": "final/abcd1234_final.mp4"}}
    (folder / "state.json").write_text(json.dumps(state))
    return folder


def final_of(folder: Path) -> Path:
    return folder / "final" / "abcd1234_final.mp4"


# Framing


@pytest.mark.parametrize(("size", "framing", "expected"), [
    ((672, 384), "long_side", (3840, 2194)),
    ((384, 672), "long_side", (2194, 3840)),
    ((544, 960), "long_side", (2176, 3840)),
    ((544, 960), "fit_uhd", (1224, 2160)),
    ((672, 384), "fit_uhd", (3780, 2160)),
    ((800, 800), "long_side", (3840, 3840)),
    ((800, 800), "fit_uhd", (2160, 2160)),
    ((384, 672), "fill_16x9", (3840, 2160)),
    ((384, 672), "pad_16x9", (3840, 2160)),
])
def test_target_size(size, framing, expected):
    assert upscale.target_size(*size, framing) == expected
    assert all(value % 2 == 0 for value in expected)


def test_framing_follows_aspect_override():
    base = Settings(upscale_framing="long_side", upscale_framing_by_aspect={"9:16": "fill_16x9"})
    assert upscale.framing_for(replace(base, aspect_ratio="9:16")) == "fill_16x9"
    assert upscale.framing_for(replace(base, aspect_ratio="16:9")) == "long_side"


# Settings


def test_older_settings_load_with_upscale_off(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({"quantity": 2, "duration": 10.0}))
    loaded = Settings.load(path)
    assert loaded.upscale is False and loaded.upscale_url == "" and loaded.upscale_framing == "long_side"
    old_state_settings = {k: v for k, v in asdict(Settings()).items() if not k.startswith("upscale")}
    assert Settings(**old_state_settings).upscale is False


@pytest.mark.parametrize("change", [
    {"upscale_url": "ftp://host"}, {"upscale_url": "http://user:pw@host:8190"},
    {"upscale_framing": "stretch"}, {"upscale_framing_by_aspect": {"5:4": "fit_uhd"}},
    {"upscale_framing_by_aspect": {"9:16": "stretch"}}, {"upscale": "yes"},
])
def test_invalid_upscale_settings_are_rejected(change):
    with pytest.raises(ValueError):
        replace(Settings(), **change).validate(require_folders=False)


def test_token_never_lands_in_settings(tmp_path, monkeypatch):
    monkeypatch.delenv("SIMPLEGEN_UPSCALE_TOKEN", raising=False)
    upscale.save_token("abc123")
    assert upscale.load_token() == "abc123"
    assert oct(upscale.token_path().stat().st_mode & 0o777) == "0o600"
    assert "abc123" not in json.dumps(Settings().public_dict())
    upscale.token_path().chmod(0o644)
    with pytest.raises(upscale.UpscaleError):
        upscale.load_token()


# Records and the protocol


def test_intent_is_saved_before_any_upload(tmp_path, host, client, probed):
    record = upscale.create(final_of(make_run(tmp_path)), framing="long_side")
    assert record["status"] == "pending" and host.calls == []
    saved = upscale.load(record["id"])
    assert saved["source_sha256"] == hashlib.sha256(final_of(tmp_path / "run").read_bytes()).hexdigest()
    uuid.UUID(saved["id"], version=4)


def test_pending_record_after_crash_checks_host_before_upload(tmp_path, host, client, probed):
    record = upscale.create(final_of(make_run(tmp_path)), framing="long_side")
    upscale.advance(dict(record), client)  # Host has it; local file still says pending.
    host.calls.clear()
    stale = upscale.load(record["id"])
    stale["status"] = "pending"
    upscale.save(stale)
    advanced = upscale.advance(upscale.load(record["id"]), client)
    assert advanced["status"] == "remote"
    assert [c for c in host.calls if c[0] == "PUT"] == []
    assert len(host.jobs) == 1


def test_full_round_trip_downloads_verifies_then_clears_host(tmp_path, host, client, probed):
    folder = make_run(tmp_path)
    record = upscale.create(final_of(folder), framing="long_side")
    record = upscale.advance(record, client)
    assert record["status"] == "remote" and record["preset"] == "seedvr2-3b+rife"
    assert upscale.advance(record, client)["status"] == "remote"  # Still queued on the host.
    host.finish(record["id"])
    record = upscale.advance(record, client)
    assert record["status"] == "complete" and record["host_cleared"] is True
    assert host.jobs == {}
    output = folder / record["output"]["path"]
    assert output.read_bytes() == b"4k60-bytes"
    assert output.name == "abcd1234_final_4k60.mp4"
    assert upscale.upscaled_version(final_of(folder)) == output.resolve()
    manifest = json.loads((folder / "final" / "4k60" / "manifest.json").read_text())
    assert manifest["outputs"][0]["job_id"] == record["id"]
    assert upscale.upscaled_metadata(output)["width"] == 3840
    # The run's own state is untouched.
    assert json.loads((folder / "state.json").read_text())["result"]["final_path"] == "final/abcd1234_final.mp4"


def test_failed_verification_keeps_host_copy_and_is_not_retried(tmp_path, host, client, probed):
    folder = make_run(tmp_path)
    probed["output"] = {**OUTPUT_META, "fps": 30.0}
    record = upscale.advance(upscale.create(final_of(folder), framing="long_side"), client)
    host.finish(record["id"])
    record = upscale.advance(record, client)
    assert record["status"] == "failed" and "frame rate" in record["error"]
    assert record["id"] in host.jobs  # Kept until the owner retries or dismisses.
    assert not (folder / "final" / "4k60" / "abcd1234_final_4k60.mp4").exists()
    host.calls.clear()
    result = upscale.tick(Settings(upscale_url="http://upscaler:8190"), client=client)
    assert host.calls == [] and result["alerts"] == []


def test_checksum_mismatch_fails_without_keeping_a_partial_file(tmp_path, host, client, probed):
    folder = make_run(tmp_path)
    record = upscale.advance(upscale.create(final_of(folder), framing="long_side"), client)
    host.finish(record["id"])
    host.corrupt_download = True
    record = upscale.advance(record, client)
    assert record["status"] == "failed" and "checksum" in record["error"]
    assert list((folder / "final" / "4k60").glob("*")) == []


def test_host_failure_alerts_once_and_waits_for_the_owner(tmp_path, host, client, probed, monkeypatch):
    posted = []
    monkeypatch.setattr("simplegen.cost_safety.webhook_url", lambda: "https://ntfy.example/topic")
    monkeypatch.setattr("simplegen.cost_safety._post", lambda url, text: posted.append(text))
    record = upscale.advance(upscale.create(final_of(make_run(tmp_path)), framing="long_side"), client)
    host.jobs[record["id"]].update(state="failed", error="CUDA out of memory")
    settings = Settings(upscale_url="http://upscaler:8190")
    first = upscale.tick(settings, client=client)
    assert [a["key"] for a in first["alerts"]] == [f"upscale:{record['id']}"]
    assert len(posted) == 1 and "CUDA out of memory" in posted[0]
    second = upscale.tick(settings, client=client)
    assert second["alerts"] == [] and len(posted) == 1
    assert upscale.load(record["id"])["status"] == "failed"


def test_unreachable_host_only_delays(tmp_path, host, client, probed):
    record = upscale.create(final_of(make_run(tmp_path)), framing="long_side")
    host.down = True
    result = upscale.tick(Settings(upscale_url="http://upscaler:8190"), client=client)
    assert "unreachable" in result["note"]
    assert upscale.load(record["id"])["status"] == "pending"
    host.down = False
    upscale.tick(Settings(upscale_url="http://upscaler:8190"), client=client)
    assert upscale.load(record["id"])["status"] == "remote"


def test_waits_for_a_default_preset(tmp_path, host, client, probed):
    host.default_preset = None
    record = upscale.advance(upscale.create(final_of(make_run(tmp_path)), framing="long_side"), client)
    assert record["status"] == "pending" and "default preset" in record["note"] and host.jobs == {}


def test_changed_source_is_not_uploaded(tmp_path, host, client, probed):
    folder = make_run(tmp_path)
    record = upscale.create(final_of(folder), framing="long_side")
    final_of(folder).write_bytes(b"different")
    assert upscale.advance(record, client)["status"] == "failed"
    assert host.jobs == {}


def test_cancel_retry_and_dismiss(tmp_path, host, client, probed):
    folder = make_run(tmp_path)
    record = upscale.advance(upscale.create(final_of(folder), framing="fit_uhd"), client)
    upscale.request_cancel(record["id"])
    record = upscale.advance(upscale.load(record["id"]), client)
    assert record["status"] == "cancelled" and record["host_cleared"] and host.jobs == {}
    other = upscale.create(final_of(make_run(tmp_path, name="other")), framing="long_side")
    with pytest.raises(ValueError, match="failed or cancelled"):
        upscale.dismiss(other["id"])  # Pending work is cancelled, not dismissed.
    retried = upscale.retry(record["id"])
    assert retried["retry_of"] == record["id"] and retried["framing"] == "fit_uhd"
    assert upscale.load(record["id"])["status"] == "dismissed"


def test_dismiss_clears_the_failed_job_on_the_host(tmp_path, host, client, probed):
    record = upscale.advance(upscale.create(final_of(make_run(tmp_path)), framing="long_side"), client)
    host.jobs[record["id"]].update(state="failed", error="boom")
    record = upscale.advance(record, client)
    assert record["status"] == "failed" and record["id"] in host.jobs
    upscale.dismiss(record["id"])
    upscale.tick(Settings(upscale_url="http://upscaler:8190"), client=client)
    assert host.jobs == {} and upscale.load(record["id"])["host_cleared"] is True


def test_one_active_upscale_per_final(tmp_path, host, client, probed):
    final = final_of(make_run(tmp_path))
    upscale.create(final, framing="long_side")
    with pytest.raises(ValueError):
        upscale.create(final, framing="long_side")


def test_clear_host_keeps_jobs_that_still_need_the_owner(tmp_path, host, client, probed):
    flagged = upscale.advance(upscale.create(final_of(make_run(tmp_path, name="a")), framing="long_side"), client)
    host.jobs[flagged["id"]].update(state="failed")
    upscale.advance(flagged, client)
    host.jobs["orphan"] = {"id": "orphan", "state": "done"}
    host.jobs["busy"] = {"id": "busy", "state": "running"}
    assert upscale.clearable_host_jobs(host.health()) == ["orphan"]
    assert upscale.clear_host(Settings(upscale_url="http://upscaler:8190"), ["orphan"], client=client) == 1
    assert set(host.jobs) == {flagged["id"], "busy"}


def test_wrong_token_is_reported(host):
    bad = upscale.Client("http://upscaler:8190", "nope", transport=httpx.MockTransport(host.handler))
    with pytest.raises(upscale.UpscaleError, match="token"):
        bad.health()


# Engine hook


def test_enqueue_only_when_requested_and_only_once(tmp_path, probed):
    on = make_run(tmp_path, name="on")
    off = make_run(tmp_path, name="off", upscale_on=False)
    state = json.loads((on / "state.json").read_text())
    first = upscale.enqueue_if_requested(on, state)
    assert first["origin"] == "auto" and first["framing"] == "long_side"
    assert upscale.enqueue_if_requested(on, state)["id"] == first["id"]
    assert upscale.enqueue_if_requested(off, json.loads((off / "state.json").read_text())) is None
    assert upscale.enqueue_if_requested(tmp_path / "missing", {"settings": {}}) is None  # Never raises.
    assert len(upscale.records()) == 1


def test_enqueue_uses_the_requests_aspect_framing(tmp_path, probed):
    folder = make_run(tmp_path, aspect="9:16")
    state = json.loads((folder / "state.json").read_text())
    state["settings"]["upscale_framing_by_aspect"] = {"9:16": "pad_16x9"}
    assert upscale.enqueue_if_requested(folder, state)["framing"] == "pad_16x9"


# Preflight (advisory)


def test_unreachable_upscale_host_never_blocks_the_render(tmp_path, host, monkeypatch):
    host.down = True
    client = upscale.Client("http://upscaler:8190", host.token, transport=httpx.MockTransport(host.handler))
    settings = Settings(upscale=True, upscale_url="http://upscaler:8190", output_folder=str(tmp_path))
    with_check = run_preflight(settings, object_info={}, resolve=lambda name: name, upscale_client=client,
                               agent_needed=False)
    without = run_preflight(replace(settings, upscale=False), object_info={}, resolve=lambda name: name,
                            agent_needed=False)
    check = next(c for c in with_check["checks"] if c["id"] == "upscale_service")
    assert check["status"] == "unverified" and check["advisory"] is True
    assert with_check["status"] == without["status"]
    assert "unreachable" not in summary(with_check)
    assert all(c["id"] != "upscale_service" for c in without["checks"])


def test_preflight_reports_estimate_and_unready_preset(host, client):
    settings = Settings(upscale=True, upscale_url="http://upscaler:8190", duration=10.0,
                        upscale_preset="")  # host default
    check = upscale.preflight_check(settings, client=client)
    assert check["status"] == "pass" and "20 min" in check["message"]
    check = upscale.preflight_check(replace(settings, upscale_preset="missing"), client=client)
    assert check["status"] == "fail"
    assert upscale.preflight_check(replace(settings, upscale_url=""))["status"] == "fail"


# Real media: ffprobe-based verification


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is required for real-media checks")
def test_verification_on_real_media(tmp_path):
    source, output = tmp_path / "source.mp4", tmp_path / "out_4k60.mp4"
    tone = ["-f", "lavfi", "-i", "sine=frequency=440:duration=1"]
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=672x384:rate=24:duration=1", *tone,
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(source)], check=True)
    subprocess.run(["ffmpeg", "-v", "error", "-i", str(source), "-vf", "scale=3840:2194,fps=60",
                    "-c:v", "libx265", "-x265-params", "log-level=error", "-preset", "ultrafast",
                    "-pix_fmt", "yuv420p10le", "-tag:v", "hvc1", "-c:a", "copy", str(output)], check=True)
    source_meta, output_meta = upscale.probe(source), upscale.probe(output)
    assert source_meta["fps"] == pytest.approx(24) and source_meta["has_audio"]
    assert output_meta["codec"] == "hevc"
    assert upscale.verify_output(source_meta, output_meta, "long_side") == []
    assert upscale.verify_output(source_meta, output_meta, "fit_uhd")  # Wrong size for that framing.


def test_method_labels_carry_the_speed_note_for_preferred_methods_only():
    assert upscale.method_label("rtxvsr+rife") == "RTX Video Super Resolution (fastest)"
    assert upscale.method_label("remacri+rife", "Remacri + RIFE") == "Remacri (fast)"
    assert upscale.method_label("seedvr2-3b+rife") == "SeedVR2 3B (slow)"
    assert upscale.method_label("ultrasharp+rife", "UltraSharp + RIFE") == "UltraSharp + RIFE"
    assert upscale.method_label("seedvr2-7b+rife") == "seedvr2-7b+rife"


def test_default_method_is_remacri_fast_with_4k60_off_and_older_settings_still_load():
    from simplegen.settings import Settings
    fresh = Settings()
    assert fresh.upscale is False and fresh.upscale_preset == upscale.DEFAULT_METHOD == "remacri+rife"
    assert Settings(upscale_preset="").upscale_preset == ""  # older files keep loading
