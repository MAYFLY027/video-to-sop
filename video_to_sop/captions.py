from __future__ import annotations

import html
import json
import re
from pathlib import Path

from .core import Evidence, PipelineError


def _clean(text: str) -> str:
    text = re.sub(r"<[^>]+>", "", text)
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def _rolling_dedup(previous: str, current: str) -> str:
    if previous == current:
        return ""
    max_overlap = min(len(previous), len(current))
    for length in range(max_overlap, 3, -1):
        if previous[-length:] == current[:length]:
            return current[length:].strip()
    return current


def parse_caption(path: Path, *, source_type: str, language: str) -> tuple[list[Evidence], list[str]]:
    rows: list[tuple[int, int, str]] = []
    try:
        if path.suffix == ".json3":
            data = json.loads(path.read_text(encoding="utf-8-sig"))
            for event in data.get("events", []):
                text = "".join(segment.get("utf8", "") for segment in event.get("segs", []))
                start = int(event.get("tStartMs", 0))
                duration = int(event.get("dDurationMs", 0))
                rows.append((start, start + duration, text))
        else:
            import webvtt

            if path.suffix == ".srt":
                cues = webvtt.from_srt(str(path))
            else:
                cues = webvtt.read(str(path))
            for cue in cues:
                rows.append((int(cue.start_in_seconds * 1000), int(cue.end_in_seconds * 1000), cue.text))
    except (OSError, ValueError, ImportError, KeyError) as exc:
        raise PipelineError(f"字幕解析失败：{exc}", stage="caption") from exc
    evidence: list[Evidence] = []
    warnings: list[str] = []
    previous_raw = ""
    for start, end, raw in rows:
        full_text = _clean(raw)
        if not full_text or end < start:
            continue
        text = _rolling_dedup(previous_raw, full_text) if source_type == "automatic" else full_text
        previous_raw = full_text
        if not text:
            continue
        evidence.append(Evidence(
            id=f"t{len(evidence) + 1:05d}", kind="transcript", start_ms=start, end_ms=end,
            text=text, origin=f"{source_type}:{language}",
        ))
    if source_type == "automatic":
        warnings.append("文字来自自动字幕，术语、命令和缺段需要复核")
    if any(evidence[i].start_ms < evidence[i - 1].start_ms for i in range(1, len(evidence))):
        warnings.append("字幕时间顺序异常")
        evidence.sort(key=lambda item: (item.start_ms, item.end_ms))
    if any(evidence[i].start_ms - evidence[i - 1].end_ms > 30000 for i in range(1, len(evidence))):
        warnings.append("字幕存在超过 30 秒的间隔；可能是正常静音，也可能有缺段")
    if not evidence:
        warnings.append("字幕文件未包含可用文本")
    return evidence, warnings
