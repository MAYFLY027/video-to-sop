from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from unittest.mock import patch

import pytest

from video_to_sop.config import Config, ModelConfig
from video_to_sop.core import Evidence, PipelineError
from video_to_sop.media import SourceProbe
from video_to_sop.pipeline import RunOptions, run_job


class FakeModel(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        length = int(self.headers["Content-Length"])
        request = json.loads(self.rfile.read(length))
        assert request["model"] == "test-deepseek"
        content = {"summary": "演示保存配置", "content_kind": "sop", "facts": [], "uncertainties": [],
                   "steps": [{"goal": "保存配置", "prerequisites": [], "action": "运行配置保存命令", "inputs": {},
                              "expected_result": "显示保存成功", "verification": "检查成功提示", "on_failure": "停止并核对命令",
                              "evidence_ids": ["t00001"], "unknowns": [], "statement_type": "shown", "depends_on_previous": True,
                              "ui_action": False, "command": "save-config", "shell": "PowerShell", "working_directory": "<PROJECT_DIR>"}]}
        body = json.dumps({"choices": [{"message": {"content": json.dumps(content)}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args: object) -> None:
        pass


def test_synthetic_video_to_report_and_resume(tmp_path: Path) -> None:
    media = tmp_path / "演示 文件.mp4"
    media.write_bytes(b"synthetic-media-placeholder")
    media.with_suffix(".json3").write_text(json.dumps({"events": [{"tStartMs": 0, "dDurationMs": 2000, "segs": [{"utf8": "Run save-config"}]}]}), encoding="utf-8")
    probe = SourceProbe(source=str(media), kind="local", source_id="local:test", title="演示 文件", duration_seconds=10,
                        has_audio=True, has_video=False, local_path=str(media), media_fingerprint="test")
    server = HTTPServer(("127.0.0.1", 0), FakeModel)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        config = Config(output_dir=str(tmp_path / "jobs"), text_model=ModelConfig(base_url=f"http://127.0.0.1:{server.server_port}", model="test-deepseek", api_key_env="VTS_TEST_KEY"))
        with patch.dict(os.environ, {"VTS_TEST_KEY": "synthetic"}), patch("video_to_sop.pipeline.probe_source", return_value=probe):
            output = tmp_path / "输出 目录"
            first = run_job(config, RunOptions(source=str(media), output=output, mode="sop"))
            second = run_job(config, RunOptions(source=str(media), output=output, mode="sop"))
        assert first["status"] == "completed", first
        assert second["status"] == "completed", second
        assert (output / "report.md").is_file()
        assert (output / "steps.json").is_file()
        assert (output / "manifest.json").is_file()
        assert "save-config" in (output / "report.md").read_text(encoding="utf-8")
        manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["stages"]["synthesis"]["cached"] is True
        assert manifest["cache_decisions"]["synthesis"] == {"action": "reuse", "reason": "阶段输入未变化"}
        assert manifest["status"] == "completed"
    finally:
        server.shutdown()
        server.server_close()


def test_no_text_model_records_unprocessed_time_range(tmp_path: Path) -> None:
    media = tmp_path / "silent.mp4"
    media.write_bytes(b"synthetic-media-placeholder")
    media.with_suffix(".json3").write_text(json.dumps({"events": [{"tStartMs": 5000, "dDurationMs": 2000, "segs": [{"utf8": "Example narration"}]}]}), encoding="utf-8")
    probe = SourceProbe(source=str(media), kind="local", source_id="local:test2", title="silent", duration_seconds=10,
                        has_audio=True, has_video=False, local_path=str(media), media_fingerprint="test2")
    with patch("video_to_sop.pipeline.probe_source", return_value=probe):
        result = run_job(Config(), RunOptions(source=str(media), output=tmp_path / "out", mode="notes"))
    assert result["status"] == "partial"
    manifest = json.loads((tmp_path / "out" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["stages"]["synthesis"]["intervals"][0]["processed"] is False
    assert "00:00:05" in (tmp_path / "out" / "report.md").read_text(encoding="utf-8")


def test_caption_gap_uses_local_asr_without_replacing_caption(tmp_path: Path) -> None:
    media = tmp_path / "gap.mp4"
    media.write_bytes(b"synthetic-media-placeholder")
    media.with_suffix(".json3").write_text(json.dumps({"events": [
        {"tStartMs": 0, "dDurationMs": 2000, "segs": [{"utf8": "First instruction"}]},
        {"tStartMs": 60000, "dDurationMs": 2000, "segs": [{"utf8": "Last instruction"}]},
    ]}), encoding="utf-8")
    probe = SourceProbe(source=str(media), kind="local", source_id="local:gap", title="gap", duration_seconds=62,
                        has_audio=True, has_video=False, local_path=str(media), media_fingerprint="test-gap")
    addition = Evidence(id="t00001", kind="transcript", start_ms=20000, end_ms=22000, text="Middle instruction", origin="faster-whisper:tiny:cpu:en")
    with patch("video_to_sop.pipeline.probe_source", return_value=probe), patch("video_to_sop.pipeline.local_asr", return_value=([addition], "local")):
        result = run_job(Config(asr_model="tiny"), RunOptions(source=str(media), output=tmp_path / "out", mode="notes"))
    assert result["status"] == "partial"
    transcript = json.loads((tmp_path / "out" / "transcript.json").read_text(encoding="utf-8"))
    assert [item["text"] for item in transcript["segments"]] == ["First instruction", "Middle instruction", "Last instruction"]
    manifest = json.loads((tmp_path / "out" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["stages"]["caption_gap_check"]["added_segments"] == 1


def test_metadata_only_blocks_without_fake_report(tmp_path: Path) -> None:
    probe = SourceProbe(source="https://www.youtube.com/watch?v=synthetic", kind="url", source_id="youtube:synthetic",
                        title="metadata only", duration_seconds=20, has_audio=None, has_video=None)
    with patch("video_to_sop.pipeline.probe_source", return_value=probe), patch("video_to_sop.pipeline.download_media", side_effect=PipelineError("需要登录", status="blocked", stage="media")):
        result = run_job(Config(), RunOptions(source=probe.source, output=tmp_path / "out", mode="sop"))
    assert result["status"] == "blocked"
    assert not (tmp_path / "out" / "report.md").exists()
    manifest = json.loads((tmp_path / "out" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["failure"]["stage"] == "evidence"


def test_output_directory_cannot_mix_sources(tmp_path: Path) -> None:
    output = tmp_path / "job"
    output.mkdir()
    (output / "manifest.json").write_text(json.dumps({"request_fingerprint": "another-source"}), encoding="utf-8")
    with pytest.raises(PipelineError, match="另一视频来源"):
        run_job(Config(), RunOptions(source="C:/video.mp4", output=output))


class FakeVisualAndTextModel(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        user_content = request["messages"][1]["content"]
        if isinstance(user_content, list):
            first = "timestamp: 0 ms" in user_content[0]["text"]
            content = {"visible_text": "Save" if first else "Saved", "visible_state": "Settings page" if first else "Saved confirmation",
                       "possible_action": "state before" if first else "state after", "uncertainty": "text blurred" if first and os.environ.get("VTS_TEST_UNCERTAIN") else ""}
        else:
            evidence = json.loads(user_content)["evidence"]
            ids = [item["id"] for item in evidence]
            content = {"summary": "视频展示保存设置。", "content_kind": "sop", "facts": [], "uncertainties": [],
                       "steps": [{"goal": "保存设置", "prerequisites": ["已打开设置页"], "action": "在设置页选择 Save",
                                  "inputs": {}, "expected_result": "页面出现保存后的状态", "verification": "检查页面状态",
                                  "on_failure": "暂停并核对当前页面", "evidence_ids": [item for item in ("t00001", "f00001", "f00002") if item in ids],
                                  "unknowns": [], "statement_type": "inferred", "depends_on_previous": True, "ui_action": True,
                                  "application": "Example App", "page": "Settings", "control": "Save"}]}
        body = json.dumps({"choices": [{"message": {"content": json.dumps(content)}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args: object) -> None:
        pass


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="requires FFmpeg")
def test_tutorial_frames_vision_and_pdf_end_to_end(tmp_path: Path) -> None:
    try:
        from playwright.sync_api import Error, sync_playwright

        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            browser.close()
    except ImportError:
        pytest.skip("Playwright was not installed")
    except Error as exc:
        if "Executable doesn't exist" in str(exc):
            pytest.skip("Playwright Chromium was not installed")
        raise
    media = tmp_path / "中文 空格 视频.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=320x180:rate=1",
                    "-f", "lavfi", "-i", "sine=frequency=440:duration=4", "-t", "4", "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-c:a", "aac", str(media)], check=True)
    media.with_suffix(".json3").write_text(json.dumps({"events": [
        {"tStartMs": 0, "dDurationMs": 2000, "segs": [{"utf8": "Select Save in Settings"}]},
    ]}), encoding="utf-8")
    server = HTTPServer(("127.0.0.1", 0), FakeVisualAndTextModel)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        model = ModelConfig(base_url=f"http://127.0.0.1:{server.server_port}", model="fake", api_key_env="VTS_TEST_KEY")
        config = Config(text_model=model, vision_model=model, max_frames=3, vision_max_frames=2, sample_interval_seconds=1)
        with patch.dict(os.environ, {"VTS_TEST_KEY": "synthetic"}):
            result = run_job(config, RunOptions(source=str(media), output=tmp_path / "交付 目录", mode="sop", format="pdf"))
        assert result["status"] == "completed", result
        assert (tmp_path / "交付 目录" / "report.pdf").is_file()
        assert (tmp_path / "交付 目录" / "report.html").is_file()
        steps = json.loads((tmp_path / "交付 目录" / "steps.json").read_text(encoding="utf-8"))["steps"]
        assert steps[0]["readiness"] == "信息足够"
        assert "f00001" in steps[0]["evidence_ids"]
        assert "![截图 f00001]" in (tmp_path / "交付 目录" / "report.md").read_text(encoding="utf-8")
    finally:
        server.shutdown()
        server.server_close()


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="requires FFmpeg")
def test_uncertain_vision_triggers_second_sampling_pass(tmp_path: Path) -> None:
    media = tmp_path / "uncertain.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=320x180:rate=1",
                    "-t", "4", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(media)], check=True)
    media.with_suffix(".json3").write_text(json.dumps({"events": [
        {"tStartMs": 0, "dDurationMs": 2000, "segs": [{"utf8": "Select Save in Settings"}]},
    ]}), encoding="utf-8")
    server = HTTPServer(("127.0.0.1", 0), FakeVisualAndTextModel)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        model = ModelConfig(base_url=f"http://127.0.0.1:{server.server_port}", model="fake", api_key_env="VTS_TEST_KEY")
        config = Config(text_model=model, vision_model=model, max_frames=8, vision_max_frames=4, sample_interval_seconds=1)
        with patch.dict(os.environ, {"VTS_TEST_KEY": "synthetic", "VTS_TEST_UNCERTAIN": "1"}):
            result = run_job(config, RunOptions(source=str(media), output=tmp_path / "out", mode="sop"))
        assert result["status"] == "partial"
        manifest = json.loads((tmp_path / "out" / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["stages"]["refinement"]["status"] == "completed"
        assert result["evidence"]["frames"] > int(manifest["stages"]["frames"]["reason"].split()[1])
    finally:
        server.shutdown()
        server.server_close()


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="requires FFmpeg")
def test_silent_video_uses_visual_evidence_without_fake_transcript(tmp_path: Path) -> None:
    media = tmp_path / "silent operation.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=320x180:rate=1",
                    "-t", "3", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(media)], check=True)
    config = Config(max_frames=3, vision_max_frames=2, sample_interval_seconds=1)
    result = run_job(config, RunOptions(source=str(media), output=tmp_path / "out", mode="sop"))
    assert result["status"] == "partial"
    assert result["evidence"]["frames"] > 0
    assert result["evidence"]["transcript_segments"] == 0
    manifest = json.loads((tmp_path / "out" / "manifest.json").read_text(encoding="utf-8"))
    assert "无音轨" in manifest["stages"]["asr"]["reason"]
