from __future__ import annotations

import json
from pathlib import Path

from video_to_sop.captions import parse_caption
from video_to_sop.core import Evidence
from video_to_sop.media import SourceProbe, pick_caption
from unittest.mock import patch

from video_to_sop.config import Config, ModelConfig
from video_to_sop.synthesis import _chunks, compile_document, merge_summaries
from video_to_sop.visuals import frame_times, refinement_times


def test_manual_caption_wins_over_auto() -> None:
    probe = SourceProbe(source="https://www.youtube.com/watch?v=x", kind="url", source_id="youtube:x", title="demo", duration_seconds=40,
                        subtitles={"en": [{"ext": "vtt"}]}, automatic_captions={"zh": [{"ext": "vtt"}]}, original_language="en")
    assert pick_caption(probe, ["zh", "en"]) == ("en", "manual", "vtt")


def test_original_language_auto_wins_over_unrelated_manual() -> None:
    probe = SourceProbe(source="https://www.youtube.com/watch?v=x", kind="url", source_id="youtube:x", title="demo", duration_seconds=40,
                        subtitles={"de": [{"ext": "vtt"}]}, automatic_captions={"en": [{"ext": "vtt"}]}, original_language="en")
    assert pick_caption(probe, ["zh", "en"]) == ("en", "automatic", "vtt")


def test_automatic_json3_deduplicates_rolling_text(tmp_path: Path) -> None:
    path = tmp_path / "subtitle.json3"
    path.write_text(json.dumps({"events": [
        {"tStartMs": 0, "dDurationMs": 1000, "segs": [{"utf8": "open settings"}]},
        {"tStartMs": 1000, "dDurationMs": 1000, "segs": [{"utf8": "open settings now"}]},
    ]}), encoding="utf-8")
    segments, warnings = parse_caption(path, source_type="automatic", language="en")
    assert [item.text for item in segments] == ["open settings", "now"]
    assert any("自动字幕" in item for item in warnings)


def test_visual_sampling_covers_end_of_silent_video() -> None:
    times = frame_times(3600, 15, 100, [])
    assert times[0] == 0
    assert times[-1] > 3400 * 1000
    assert len(times) <= 100


def test_ui_step_without_vision_is_not_ready() -> None:
    transcript = [Evidence(id="t00001", kind="transcript", start_ms=1000, end_ms=2000, text="click save")]
    frames = [Evidence(id="f00001", kind="frame", start_ms=1500, end_ms=1500, path="frames/f.jpg")]
    chunks = [{"content_kind": "sop", "summary": "demo", "facts": [], "uncertainties": [], "range": [1000, 2000],
               "steps": [{"goal": "save", "action": "Click Save", "evidence_ids": ["t00001", "f00001"], "ui_action": True}]}]
    document = compile_document(chunks, transcript, frames, {}, mode="sop")
    assert document["steps"][0]["readiness"] == "关键证据缺失"
    assert "画面证据" in document["steps"][0]["unknowns"][0]


def test_command_from_asr_is_unconfirmed() -> None:
    transcript = [Evidence(id="t00001", kind="transcript", start_ms=0, end_ms=1000, text="run command", origin="faster-whisper:tiny:cpu:en")]
    chunks = [{"content_kind": "sop", "summary": "demo", "facts": [], "uncertainties": [], "range": [0, 1000],
               "steps": [{"goal": "run", "action": "Run command", "evidence_ids": ["t00001"], "command": "run-cmd",
                          "shell": "PowerShell", "working_directory": "C:/project", "expected_result": "done", "verification": "check output"}]}]
    step = compile_document(chunks, transcript, [], {}, mode="sop")["steps"][0]
    assert step["readiness"] == "关键证据缺失"
    assert any("自动转写" in value for value in step["unknowns"])


def test_one_frame_cannot_confirm_a_click() -> None:
    frames = [Evidence(id="f00001", kind="frame", start_ms=1000, end_ms=1000, path="frames/one.jpg")]
    chunks = [{"content_kind": "sop", "summary": "demo", "facts": [], "uncertainties": [], "range": [1000, 1000],
               "steps": [{"goal": "save", "action": "Click Save", "evidence_ids": ["f00001"], "ui_action": True,
                          "expected_result": "Saved", "verification": "Read status"}]}]
    step = compile_document(chunks, [], frames, {"f00001": {"visible_state": "Settings", "uncertainty": ""}}, mode="sop")["steps"][0]
    assert step["readiness"] == "关键证据缺失"


def test_long_timeline_chunks_overlap_at_boundary() -> None:
    transcript = [Evidence(id=f"t{i:05d}", kind="transcript", start_ms=i * 10000, end_ms=i * 10000 + 1000, text="explanation " * 40) for i in range(1, 35)]
    chunks = _chunks(transcript, {}, {}, max_chars=1800)
    assert len(chunks) > 2
    assert all(chunks[i][-1].id == chunks[i + 1][0].id for i in range(len(chunks) - 1))


def test_uncertain_frame_triggers_bounded_local_resampling() -> None:
    frames = [Evidence(id="f00001", kind="frame", start_ms=2000, end_ms=2000), Evidence(id="f00002", kind="frame", start_ms=4000, end_ms=4000)]
    assert refinement_times(frames, {"f00001": {"uncertainty": "text blurred"}}, 5, 2) == [1000, 1500]
    assert refinement_times(frames, {"f00001": {"uncertainty": ""}}, 5, 2) == []


def test_long_summary_merges_in_groups() -> None:
    document = {"summary": "\n\n".join((f"Segment {i} " * 300) for i in range(12))}
    config = Config(text_model=ModelConfig(base_url="http://localhost", model="fake", api_key_env="KEY"))
    with patch.dict("os.environ", {"KEY": "synthetic"}), patch("video_to_sop.synthesis.chat_json", return_value={"summary": "Merged"}) as model:
        summary, warnings = merge_summaries(document, config)
    assert summary == "Merged"
    assert warnings == []
    assert model.call_count >= 2
