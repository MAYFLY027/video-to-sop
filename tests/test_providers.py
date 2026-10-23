from __future__ import annotations

import json
import os
import shutil
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from video_to_sop.config import ModelConfig
from video_to_sop.providers import local_asr, probe_model, probe_remote_asr


class FakeCapabilityServer(BaseHTTPRequestHandler):
    seen: list[str] = []

    def do_POST(self) -> None:
        body = self.rfile.read(int(self.headers["Content-Length"]))
        self.seen.append(self.path)
        if self.path.endswith("/chat/completions"):
            message = json.loads(body)["messages"][1]["content"]
            if isinstance(message, list):
                assert any(item.get("type") == "image_url" for item in message)
            response = {"choices": [{"message": {"content": '{"ok": true}'}}]}
        else:
            assert self.path.endswith("/audio/transcriptions")
            response = {"segments": []}
        encoded = json.dumps(response).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, *args: object) -> None:
        pass


def test_text_and_image_capabilities_are_probed_separately() -> None:
    FakeCapabilityServer.seen = []
    server = HTTPServer(("127.0.0.1", 0), FakeCapabilityServer)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        model = ModelConfig(base_url=f"http://127.0.0.1:{server.server_port}", model="fake", api_key_env="VTS_TEST_KEY")
        with patch.dict(os.environ, {"VTS_TEST_KEY": "synthetic"}):
            assert probe_model(model, vision=False, retry_limit=0)[0] is True
            assert probe_model(model, vision=True, retry_limit=0)[0] is True
        assert len(FakeCapabilityServer.seen) == 2
    finally:
        server.shutdown()
        server.server_close()


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="FFmpeg is required for the audio capability probe")
def test_audio_capability_has_independent_probe() -> None:
    FakeCapabilityServer.seen = []
    server = HTTPServer(("127.0.0.1", 0), FakeCapabilityServer)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        model = ModelConfig(base_url=f"http://127.0.0.1:{server.server_port}", model="fake-asr", api_key_env="VTS_TEST_KEY")
        with patch.dict(os.environ, {"VTS_TEST_KEY": "synthetic"}):
            assert probe_remote_asr(model, retry_limit=0)[0] is True
        assert FakeCapabilityServer.seen == ["/audio/transcriptions"]
    finally:
        server.shutdown()
        server.server_close()


def test_auto_asr_falls_back_when_cuda_fails_during_transcription(tmp_path: Path) -> None:
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    (audio_dir / "audio-full-16k.wav").write_bytes(b"unused")
    devices = []

    class FakeWhisperModel:
        def __init__(self, _name: str, *, device: str, compute_type: str) -> None:
            devices.append(device)
            self.device = device

        def transcribe(self, _path: str, **_kwargs: object):
            if self.device == "cuda":
                def failing_segments():
                    raise RuntimeError("cublas is unavailable")
                    yield

                return failing_segments(), SimpleNamespace(language="zh")
            segment = SimpleNamespace(start=0.0, end=1.0, text="测试", avg_logprob=0.0)
            return iter([segment]), SimpleNamespace(language="zh")

    fake_modules = {
        "ctranslate2": SimpleNamespace(get_cuda_device_count=lambda: 1),
        "faster_whisper": SimpleNamespace(WhisperModel=FakeWhisperModel),
    }
    with patch.dict(sys.modules, fake_modules):
        evidence, source = local_asr(tmp_path / "input.mp4", audio_dir, model_name="small", device="auto")
    assert devices == ["cuda", "cpu"]
    assert source == "faster-whisper:small:cpu:zh"
    assert [item.text for item in evidence] == ["测试"]
