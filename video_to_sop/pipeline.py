from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .captions import parse_caption
from .config import Config
from .core import Evidence, PipelineError, atomic_json, digest, file_sha256, package_versions, read_json, slug
from .media import SourceProbe, download_caption, download_media, ffprobe, local_caption, pick_caption, probe_source
from .providers import local_asr, remote_asr
from .render import render_html, render_pdf, write_report, write_transcript
from .synthesis import compile_document, extract_facts, merge_summaries
from .visuals import analyze_frames, extract_frames, extract_frames_at_times, refinement_times, run_ocr


@dataclass
class RunOptions:
    source: str
    output: Path | None = None
    mode: str = "auto"
    format: str = "md"
    episode: int | None = None
    cookie_file: Path | None = None
    resume: bool = True


def _stage(manifest: dict, name: str, status: str, reason: str = "", **detail: Any) -> None:
    manifest.setdefault("stages", {})[name] = {"status": status, "reason": reason, **detail}


def _manifest_path(job_dir: Path) -> Path:
    return job_dir / "manifest.json"


def _save_manifest(job_dir: Path, manifest: dict) -> None:
    atomic_json(_manifest_path(job_dir), manifest)


def _cached(cache_path: Path, key: str, resume: bool) -> Any | None:
    if not resume:
        return None
    data = read_json(cache_path)
    return data.get("value") if isinstance(data, dict) and data.get("key") == key else None


def _cache_decision(manifest: dict, stage: str, cache_path: Path, key: str, resume: bool) -> None:
    existing = read_json(cache_path)
    if not resume:
        reason = "用户关闭续跑"
    elif not isinstance(existing, dict):
        reason = "无前次阶段结果"
    elif existing.get("key") != key:
        reason = "证据、模型、提示词或相关配置变化"
    else:
        reason = "阶段输入未变化"
    manifest.setdefault("cache_decisions", {})[stage] = {"action": "reuse" if reason == "阶段输入未变化" else "rerun", "reason": reason}


def _store(cache_path: Path, key: str, value: Any) -> None:
    atomic_json(cache_path, {"key": key, "value": value})


def _public_source(probe: SourceProbe) -> dict:
    data = probe.public()
    if probe.kind == "local":
        data["source"] = Path(probe.source).name
        data["local_path"] = ""
    return data


def _load_probe(source: str, job_dir: Path, opts: RunOptions) -> SourceProbe:
    cache = job_dir / "cache" / "probe.json"
    if opts.resume and source.startswith(("http://", "https://")):
        saved = read_json(cache)
        if isinstance(saved, dict) and saved.get("source") == source and saved.get("episode") == opts.episode:
            return SourceProbe(**saved["probe"])
    probe = probe_source(source, episode=opts.episode, cookie_file=opts.cookie_file)
    atomic_json(cache, {"source": source, "episode": opts.episode, "probe": asdict(probe)})
    return probe


def _media_info(media: Path) -> tuple[bool, bool, float | None]:
    data = ffprobe(media)
    streams = data.get("streams", [])
    try:
        duration = float(data.get("format", {}).get("duration"))
    except (ValueError, TypeError):
        duration = None
    return any(s.get("codec_type") == "audio" for s in streams), any(s.get("codec_type") == "video" for s in streams), duration


def _load_transcript(job_dir: Path, probe: SourceProbe, media: Path | None, has_audio: bool | None, config: Config, opts: RunOptions, manifest: dict, warnings: list[str]) -> tuple[list[Evidence], str]:
    caption_path = None
    caption_type = ""
    caption_language = ""
    if probe.kind == "local":
        caption_path = local_caption(Path(probe.local_path))
        if caption_path:
            caption_type, caption_language = "sidecar", "unknown"
    else:
        choice = pick_caption(probe, config.source_languages)
        if choice:
            caption_language, caption_type, _ = choice
            try:
                caption_path = download_caption(probe, choice, job_dir / "cache" / "caption" / slug(f"{caption_type}-{caption_language}"), opts.cookie_file)
            except PipelineError as exc:
                warnings.append(exc.reason)
                _stage(manifest, "caption", "failed", exc.reason, source_type=caption_type, language=caption_language)
        else:
            _stage(manifest, "caption", "completed", "平台未列出可用字幕轨道")
    transcript: list[Evidence] = []
    source_label = "none"
    if caption_path:
        key = digest({"file": file_sha256(caption_path), "type": caption_type, "language": caption_language})
        cached = _cached(job_dir / "cache" / "transcript-caption.json", key, opts.resume)
        if cached is None:
            try:
                transcript, caption_warnings = parse_caption(caption_path, source_type=caption_type, language=caption_language)
                _store(job_dir / "cache" / "transcript-caption.json", key, {"segments": [asdict(item) for item in transcript], "warnings": caption_warnings})
            except PipelineError as exc:
                warnings.append(exc.reason)
                transcript, caption_warnings = [], []
        else:
            transcript = [Evidence(**item) for item in cached["segments"]]
            caption_warnings = cached["warnings"]
        warnings.extend(caption_warnings)
        if transcript:
            source_label = f"{caption_type}:{caption_language}"
            _stage(manifest, "caption", "completed", source_label, segments=len(transcript), cached=cached is not None)
    if not transcript and has_audio and media:
        key = digest({"media": file_sha256(media), "mode": config.asr_mode, "model": config.asr_model, "device": config.asr_device, "remote": config.remote_asr.public(), "asr_schema": 2})
        _cache_decision(manifest, "asr", job_dir / "cache" / "transcript-asr.json", key, opts.resume)
        cached = _cached(job_dir / "cache" / "transcript-asr.json", key, opts.resume)
        if cached is not None:
            transcript = [Evidence(**item) for item in cached["segments"]]
            source_label = cached["source"]
        elif config.asr_mode == "local":
            transcript, source_label = local_asr(media, job_dir / "cache" / "audio", model_name=config.asr_model, device=config.asr_device)
            _store(job_dir / "cache" / "transcript-asr.json", key, {"segments": [asdict(item) for item in transcript], "source": source_label})
        elif config.asr_mode == "remote":
            transcript, source_label = remote_asr(media, job_dir / "cache" / "audio", duration_seconds=probe.duration_seconds or _media_info(media)[2] or 0, model=config.remote_asr, retry_limit=config.retry_limit)
            _store(job_dir / "cache" / "transcript-asr.json", key, {"segments": [asdict(item) for item in transcript], "source": source_label})
        else:
            warnings.append("存在音轨但转写已关闭，未取得文字证据")
        _stage(manifest, "asr", "completed" if transcript else "partial", source_label, segments=len(transcript), cached=cached is not None)
    elif transcript and media and has_audio and config.asr_mode == "local":
        duration_ms = int((probe.duration_seconds or _media_info(media)[2] or 0) * 1000)
        sorted_segments = sorted(transcript, key=lambda item: item.start_ms)
        boundaries = [(0, sorted_segments[0].start_ms)]
        boundaries.extend((a.end_ms, b.start_ms) for a, b in zip(sorted_segments, sorted_segments[1:]))
        boundaries.append((sorted_segments[-1].end_ms, duration_ms))
        gaps = [(max(0, start), min(end, duration_ms)) for start, end in boundaries if end - start > 30000]
        gaps = sorted(gaps, key=lambda pair: pair[1] - pair[0], reverse=True)[:3]
        if gaps:
            key = digest({"caption": [asdict(item) for item in transcript], "media": file_sha256(media), "gaps": gaps, "model": config.asr_model, "device": config.asr_device})
            cached = _cached(job_dir / "cache" / "transcript-gaps.json", key, opts.resume)
            if cached is None:
                additions: list[Evidence] = []
                for start, end in gaps:
                    for clip_start in range(start, end, 180000):
                        try:
                            added, _ = local_asr(media, job_dir / "cache" / "audio", model_name=config.asr_model, device=config.asr_device,
                                                 start_seconds=clip_start / 1000, duration_seconds=min(180, (end - clip_start) / 1000))
                            additions.extend(added)
                        except PipelineError as exc:
                            warnings.append(f"字幕疑似缺口 {clip_start / 1000:.1f}-{end / 1000:.1f}s 补转写失败：{exc.reason}")
                _store(job_dir / "cache" / "transcript-gaps.json", key, [asdict(item) for item in additions])
            else:
                additions = [Evidence(**item) for item in cached]
            if additions:
                transcript = sorted([*transcript, *additions], key=lambda item: (item.start_ms, item.end_ms))
                for index, item in enumerate(transcript, 1):
                    item.id = f"t{index:05d}"
                warnings.append(f"字幕有疑似缺口，已用本地 ASR 补充 {len(additions)} 段；补充内容未人工校对")
                source_label += "+local-gap-asr"
            _stage(manifest, "caption_gap_check", "completed", "已抽查最长的字幕间隔", intervals=gaps, added_segments=len(additions), cached=cached is not None)
    if any(item.origin.startswith(("faster-whisper", "remote-asr")) for item in transcript):
        warnings.append("自动转写未经人工校对，术语、命令、参数和专有名称需结合画面复核")
    if any(item.warning for item in transcript):
        warnings.append("部分转写片段置信度偏低，已在 transcript.json 中逐段标注")
    if not transcript and not has_audio:
        _stage(manifest, "asr", "completed", "无音轨；将以画面为主要证据")
    return transcript, source_label


def _anchors(chunks: list[dict], transcript: list[Evidence]) -> list[int]:
    by_id = {item.id: item for item in transcript}
    anchors = []
    for chunk in chunks:
        for step in chunk.get("steps", []):
            if isinstance(step, dict):
                anchors.extend(by_id[eid].start_ms for eid in step.get("evidence_ids", []) if eid in by_id)
    return anchors


def run_job(config: Config, opts: RunOptions) -> dict:
    if opts.mode not in {"auto", "sop", "notes"} or opts.format not in {"md", "html", "pdf"}:
        raise PipelineError("mode 或 format 值无效", status="blocked", stage="config")
    source_key = digest({"source": opts.source, "episode": opts.episode})[:10]
    root = Path(config.output_dir).expanduser()
    job_dir = opts.output or root / f"{slug(Path(opts.source).stem or 'video')}-{source_key}"
    job_dir = job_dir.resolve()
    job_dir.mkdir(parents=True, exist_ok=True)
    request_fingerprint = digest({"source": opts.source, "episode": opts.episode})
    previous_manifest = read_json(_manifest_path(job_dir))
    if isinstance(previous_manifest, dict) and previous_manifest.get("request_fingerprint") not in (None, request_fingerprint):
        raise PipelineError("输出目录属于另一视频来源；请选择新的 --output 目录", status="blocked", stage="config")
    manifest: dict = {"schema_version": 1, "status": "failed", "source": opts.source if opts.source.startswith("http") else Path(opts.source).name,
                      "request_fingerprint": request_fingerprint,
                      "request": {"mode": opts.mode, "format": opts.format, "language": config.output_language},
                      "models": {"text": config.text_model.model, "vision": config.vision_model.model, "remote_asr": config.remote_asr.model},
                      "versions": package_versions(), "stages": {}, "warnings": [], "artifacts": {}}
    _save_manifest(job_dir, manifest)
    warnings: list[str] = []
    try:
        probe = _load_probe(opts.source, job_dir, opts)
        if probe.duration_seconds and probe.duration_seconds > config.max_duration_minutes * 60:
            raise PipelineError(f"视频超过配置时长上限 {config.max_duration_minutes} 分钟", status="blocked", stage="probe")
        manifest["probe"] = _public_source(probe)
        _stage(manifest, "probe", "completed", probe.kind)
        _save_manifest(job_dir, manifest)

        media: Path | None = Path(probe.local_path) if probe.kind == "local" else None
        has_audio, has_video = probe.has_audio, probe.has_video
        need_media = probe.kind == "url" and (opts.mode != "notes" or config.vision_model.ready or not pick_caption(probe, config.source_languages))
        if need_media:
            try:
                media = download_media(probe, job_dir / "cache" / "media", opts.cookie_file)
                has_audio, has_video, actual_duration = _media_info(media)
                probe.media_fingerprint = file_sha256(media)
                if actual_duration:
                    probe.duration_seconds = actual_duration
                _stage(manifest, "media", "completed", "已取得媒体", fingerprint=probe.media_fingerprint)
            except PipelineError as exc:
                warnings.append(exc.reason)
                _stage(manifest, "media", exc.status, exc.reason)
        elif media:
            _stage(manifest, "media", "completed", "本地媒体", fingerprint=probe.media_fingerprint)
        _save_manifest(job_dir, manifest)

        try:
            transcript, text_source = _load_transcript(job_dir, probe, media, has_audio, config, opts, manifest, warnings)
        except PipelineError as exc:
            transcript, text_source = [], "none"
            warnings.append(exc.reason)
            _stage(manifest, "asr", exc.status, exc.reason)
        write_transcript(job_dir, transcript, "无音轨或未取得可用字幕／转写" if not transcript else "")
        _save_manifest(job_dir, manifest)
        if not transcript and not media:
            raise PipelineError("仅取得元信息，字幕和媒体均不可用；请提供可访问的视频或本地字幕", status="blocked", stage="evidence")
        if not transcript and not has_video:
            raise PipelineError("没有可用文字或画面证据", status="blocked", stage="evidence")

        first_key = digest({"transcript": [asdict(item) for item in transcript], "text_model": config.text_model.public(), "mode": opts.mode, "prompt": 2})
        _cache_decision(manifest, "initial_synthesis", job_dir / "cache" / "initial-facts.json", first_key, opts.resume)
        first_cached = _cached(job_dir / "cache" / "initial-facts.json", first_key, opts.resume)
        if first_cached is None:
            initial_chunks, initial_warnings = extract_facts(transcript, [], {}, {}, config, mode=opts.mode)
            _store(job_dir / "cache" / "initial-facts.json", first_key, {"chunks": initial_chunks, "warnings": initial_warnings})
        else:
            initial_chunks, initial_warnings = first_cached["chunks"], first_cached["warnings"]
        warnings.extend(initial_warnings)
        _stage(manifest, "initial_synthesis", "completed" if initial_chunks else "partial", "文字分段提取", cached=first_cached is not None)
        _save_manifest(job_dir, manifest)

        frames: list[Evidence] = []
        ocr: dict[str, str] = {}
        vision: dict[str, dict] = {}
        anchors = _anchors(initial_chunks, transcript)
        need_frames = bool(media and has_video and (opts.mode != "notes" or config.vision_model.ready or not transcript))
        if need_frames:
            from dataclasses import replace

            first_pass_config = replace(
                config,
                max_frames=max(1, config.max_frames - max(1, config.max_frames // 4)),
                vision_max_frames=config.vision_max_frames if config.vision_max_frames <= 2 else config.vision_max_frames - max(1, config.vision_max_frames // 4),
            )
            frame_key = digest({"media": probe.media_fingerprint or file_sha256(media), "duration": probe.duration_seconds, "interval": config.sample_interval_seconds, "max": first_pass_config.max_frames, "anchors": anchors})
            _cache_decision(manifest, "frames", job_dir / "cache" / "frame-evidence.json", frame_key, opts.resume)
            frame_cached = _cached(job_dir / "cache" / "frame-evidence.json", frame_key, opts.resume)
            if frame_cached is None or not all((job_dir / item["path"]).is_file() for item in frame_cached):
                frames = extract_frames(media, job_dir, probe.duration_seconds or _media_info(media)[2] or 0, first_pass_config, anchors)
                _store(job_dir / "cache" / "frame-evidence.json", frame_key, [asdict(item) for item in frames])
            else:
                frames = [Evidence(**item) for item in frame_cached]
            _stage(manifest, "frames", "completed" if frames else "partial", f"采样 {len(frames)} 帧", cached=frame_cached is not None)
            ocr_key = digest({"frames": frame_key, "ocr": manifest["versions"].get("rapidocr-onnxruntime")})
            _cache_decision(manifest, "ocr", job_dir / "cache" / "ocr.json", ocr_key, opts.resume)
            ocr_cached = _cached(job_dir / "cache" / "ocr.json", ocr_key, opts.resume)
            if ocr_cached is None:
                ocr, ocr_warning = run_ocr(frames, job_dir)
                if ocr_warning:
                    warnings.append(ocr_warning)
                _store(job_dir / "cache" / "ocr.json", ocr_key, {"values": ocr, "warning": ocr_warning})
            else:
                ocr, ocr_warning = ocr_cached["values"], ocr_cached["warning"]
                if ocr_warning:
                    warnings.append(ocr_warning)
            _stage(manifest, "ocr", "completed" if ocr else "partial", ocr_warning or "已完成")
            vision_key = digest({"frames": frame_key, "model": config.vision_model.public(), "limit": first_pass_config.vision_max_frames, "ocr": ocr})
            _cache_decision(manifest, "vision", job_dir / "cache" / "vision.json", vision_key, opts.resume)
            vision_cached = _cached(job_dir / "cache" / "vision.json", vision_key, opts.resume)
            if vision_cached is None:
                vision, vision_warning = analyze_frames(frames, job_dir, first_pass_config, ocr, anchors)
                if vision_warning:
                    warnings.append(vision_warning)
                _store(job_dir / "cache" / "vision.json", vision_key, {"values": vision, "warning": vision_warning})
            else:
                vision, vision_warning = vision_cached["values"], vision_cached["warning"]
                if vision_warning:
                    warnings.append(vision_warning)
            _stage(manifest, "vision", "completed" if vision and not vision_warning else "partial", vision_warning or "已分析画面", frames_analyzed=len(vision))
            remaining_calls = max(0, config.vision_max_frames - len(vision))
            extra_times = refinement_times(frames, vision, probe.duration_seconds or 0, min(config.max_frames - len(frames), remaining_calls * 4)) if remaining_calls else []
            if extra_times:
                refinement_key = digest({"base_frames": frame_key, "vision": vision, "model": config.vision_model.public(), "times": extra_times, "ocr": manifest["versions"].get("rapidocr-onnxruntime")})
                refinement_path = job_dir / "cache" / "refinement.json"
                _cache_decision(manifest, "refinement", refinement_path, refinement_key, opts.resume)
                refinement_cached = _cached(refinement_path, refinement_key, opts.resume)
                if refinement_cached is None or not all((job_dir / item["path"]).is_file() for item in refinement_cached["frames"]):
                    extra_frames = extract_frames_at_times(media, job_dir, extra_times, start_index=len(frames) + 1)
                    extra_ocr, extra_ocr_warning = run_ocr(extra_frames, job_dir)
                    extra_vision, extra_vision_warning = analyze_frames(extra_frames, job_dir, replace(config, vision_max_frames=remaining_calls), extra_ocr, extra_times) if remaining_calls else ({}, "视觉调用上限已用尽")
                    _store(refinement_path, refinement_key, {"frames": [asdict(item) for item in extra_frames], "ocr": extra_ocr, "vision": extra_vision, "warnings": [value for value in (extra_ocr_warning, extra_vision_warning) if value]})
                else:
                    extra_frames = [Evidence(**item) for item in refinement_cached["frames"]]
                    extra_ocr = refinement_cached["ocr"]
                    extra_vision = refinement_cached["vision"]
                    extra_ocr_warning = extra_vision_warning = None
                    warnings.extend(refinement_cached.get("warnings", []))
                if extra_ocr_warning:
                    warnings.append(extra_ocr_warning)
                if extra_vision_warning:
                    warnings.append(extra_vision_warning)
                frames.extend(extra_frames)
                ocr.update(extra_ocr)
                vision.update(extra_vision)
                _stage(manifest, "refinement", "completed" if extra_vision else "partial", f"补采 {len(extra_frames)} 帧", cached=refinement_cached is not None)
        elif opts.mode == "sop" or (opts.mode == "auto" and has_video):
            warnings.append("未取得可核对的画面，界面步骤只能作为草稿")
            _stage(manifest, "frames", "partial", "媒体或画面不可用")
        _save_manifest(job_dir, manifest)

        final_key = digest({"transcript": [asdict(item) for item in transcript], "frames": [asdict(item) for item in frames], "vision": vision, "ocr": ocr, "text_model": config.text_model.public(), "mode": opts.mode, "prompt": 2})
        _cache_decision(manifest, "synthesis", job_dir / "cache" / "final-facts.json", final_key, opts.resume)
        final_cached = _cached(job_dir / "cache" / "final-facts.json", final_key, opts.resume)
        if final_cached is None:
            chunks, synth_warnings = extract_facts(transcript, frames, vision, ocr, config, mode=opts.mode)
            _store(job_dir / "cache" / "final-facts.json", final_key, {"chunks": chunks, "warnings": synth_warnings})
        else:
            chunks, synth_warnings = final_cached["chunks"], final_cached["warnings"]
        warnings.extend(synth_warnings)
        document = compile_document(chunks, transcript, frames, vision, mode=opts.mode)
        merged_summary, merge_warnings = merge_summaries(document, config)
        document["summary"] = merged_summary
        warnings.extend(merge_warnings)
        _stage(manifest, "synthesis", "completed" if chunks and not synth_warnings else "partial", "生成结构化文档", cached=final_cached is not None, intervals=document["intervals"])
        atomic_json(job_dir / "steps.json", {"schema_version": 1, "mode": document["mode"], "steps": document["steps"], "facts": document["facts"]})

        visual_required = document["mode"] in {"sop", "mixed"} and has_video is not False
        missing_vision = visual_required and (not frames or not vision)
        incomplete_steps = any(step["readiness"] == "关键证据缺失" for step in document["steps"])
        unprocessed = any(not interval["processed"] for interval in document["intervals"])
        status = "partial" if missing_vision or incomplete_steps or unprocessed or not chunks or synth_warnings or not config.text_model.ready else "completed"
        if missing_vision:
            warnings.append("SOP 的画面操作没有完成视觉核对")
        if not transcript and has_video:
            warnings.append("无文字证据，文档仅基于有限画面采样")
            status = "partial" if not vision else status
        write_report(job_dir, _public_source(probe), document, transcript, frames, warnings, status)
        _stage(manifest, "report", "completed" if status == "completed" else "partial", "报告与步骤已生成")
        artifacts = {name: name for name in ("report.md", "steps.json", "transcript.json", "transcript.md")}
        if frames:
            artifacts["frames"] = "frames/"
        if opts.format in {"html", "pdf"}:
            try:
                render_html(job_dir)
                artifacts["report.html"] = "report.html"
                if opts.format == "pdf":
                    render_pdf(job_dir)
                    artifacts["report.pdf"] = "report.pdf"
                _stage(manifest, "export", "completed", opts.format)
            except PipelineError as exc:
                warnings.append(exc.reason)
                _stage(manifest, "export", "failed", exc.reason)
                status = "partial"
                write_report(job_dir, _public_source(probe), document, transcript, frames, warnings, status)
                if "report.html" in artifacts:
                    render_html(job_dir)
        manifest["status"] = status
        manifest["warnings"] = list(dict.fromkeys(warnings))
        manifest["artifacts"] = artifacts
        manifest["evidence"] = {"transcript_source": text_source, "transcript_segments": len(transcript), "frames": len(frames), "vision_analyzed": len(vision)}
        _save_manifest(job_dir, manifest)
        return {"status": status, "job_dir": str(job_dir), "artifacts": artifacts, "warnings": manifest["warnings"], "evidence": manifest["evidence"]}
    except PipelineError as exc:
        manifest["status"] = exc.status
        manifest["failure"] = {"stage": exc.stage, "reason": exc.reason}
        manifest["warnings"] = list(dict.fromkeys([*warnings, exc.reason]))
        _stage(manifest, exc.stage, exc.status, exc.reason)
        _save_manifest(job_dir, manifest)
        return {"status": exc.status, "job_dir": str(job_dir), "reason": exc.reason, "stage": exc.stage, "artifacts": manifest.get("artifacts", {})}
    except Exception as exc:
        reason = f"未预期处理错误：{type(exc).__name__}: {exc}"
        manifest["status"] = "failed"
        manifest["failure"] = {"stage": "unknown", "reason": reason}
        manifest["warnings"] = list(dict.fromkeys([*warnings, reason]))
        _stage(manifest, "unknown", "failed", reason)
        _save_manifest(job_dir, manifest)
        return {"status": "failed", "job_dir": str(job_dir), "reason": reason, "stage": "unknown", "artifacts": manifest.get("artifacts", {})}
