from __future__ import annotations

import base64
import json
import mimetypes
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any

from .config import ModelConfig
from .core import Evidence, PipelineError, run_command


def _endpoint(base_url: str, suffix: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith(suffix):
        return base
    return base + suffix


def _post_json(url: str, payload: dict, key: str, *, retry_limit: int, stage: str) -> dict:
    request = urllib.request.Request(
        url, json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        {"Authorization": f"Bearer {key}", "Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    for attempt in range(retry_limit + 1):
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            detail = exc.read(400).decode("utf-8", errors="replace")
            if exc.code in (401, 403):
                raise PipelineError(f"模型认证或权限失败（HTTP {exc.code}）：{detail}", status="blocked", stage=stage) from exc
            if exc.code in (400, 404, 422):
                raise PipelineError(f"模型或接口不支持请求（HTTP {exc.code}）：{detail}", status="blocked", stage=stage) from exc
            if exc.code not in (408, 429, 500, 502, 503, 504) or attempt == retry_limit:
                raise PipelineError(f"模型请求失败（HTTP {exc.code}）：{detail}", stage=stage) from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            if attempt == retry_limit:
                raise PipelineError(f"模型网络请求失败：{exc}", stage=stage) from exc
        time.sleep(min(8, 2 ** attempt))
    raise PipelineError("模型请求失败", stage=stage)


def _extract_content(response: dict, stage: str) -> str:
    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise PipelineError("模型响应缺少 message.content", stage=stage) from exc
    if isinstance(content, list):
        content = "\n".join(item.get("text", "") for item in content if isinstance(item, dict))
    if not isinstance(content, str) or not content.strip():
        raise PipelineError("模型返回空内容", stage=stage)
    return content.strip()


def _parse_json_content(content: str, stage: str) -> dict:
    if content.startswith("```"):
        content = content.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        raise PipelineError(f"模型未返回有效 JSON：{content[:200]}", stage=stage) from exc
    if not isinstance(parsed, dict):
        raise PipelineError("模型 JSON 顶层必须是对象", stage=stage)
    return parsed


def chat_json(model: ModelConfig, system: str, user: str, *, retry_limit: int, stage: str, images: list[Path] | None = None) -> dict:
    if not model.ready:
        raise PipelineError("模型地址、名称或本机密钥未配置", status="blocked", stage=stage)
    user_content: Any = user
    if images:
        user_content = [{"type": "text", "text": user}]
        for path in images:
            mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
            encoded = base64.b64encode(path.read_bytes()).decode("ascii")
            user_content.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}})
    payload = {"model": model.model, "messages": [{"role": "system", "content": system}, {"role": "user", "content": user_content}], "temperature": 0}
    response = _post_json(_endpoint(model.base_url, "/chat/completions"), payload, model.key, retry_limit=retry_limit, stage=stage)
    return _parse_json_content(_extract_content(response, stage), stage)


def probe_model(model: ModelConfig, *, vision: bool, retry_limit: int) -> tuple[bool, str]:
    if not model.ready:
        return False, "未配置模型地址、名称或本机密钥"
    images = None
    temporary = None
    if vision:
        import tempfile

        temporary = tempfile.TemporaryDirectory()
        images = [Path(temporary.name) / "probe.png"]
        images[0].write_bytes(base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQIHWP4z8DwHwAFgAI/ScL/nwAAAABJRU5ErkJggg=="))
    try:
        result = chat_json(model, "Return JSON only.", "Return {\"ok\": true}." if not vision else "Inspect the image and return {\"ok\": true}.", retry_limit=retry_limit, stage="doctor", images=images)
        return result.get("ok") is True, "请求成功" if result.get("ok") is True else "响应未包含 ok=true"
    except PipelineError as exc:
        return False, exc.reason
    finally:
        if temporary:
            temporary.cleanup()


def local_asr(media: Path, audio_dir: Path, *, model_name: str, device: str, start_seconds: float = 0, duration_seconds: float | None = None) -> tuple[list[Evidence], str]:
    try:
        import ctranslate2
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise PipelineError("缺少 faster-whisper；安装 asr 可选依赖或配置远程转写", status="blocked", stage="asr") from exc
    chosen_device = device
    if device == "auto":
        try:
            chosen_device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
        except Exception:
            chosen_device = "cpu"
    model_label = Path(model_name).name if Path(model_name).is_dir() else model_name
    audio_dir.mkdir(parents=True, exist_ok=True)
    clip_id = f"{int(start_seconds * 1000)}-{int((duration_seconds or 0) * 1000)}" if start_seconds or duration_seconds else "full"
    wav = audio_dir / f"audio-{clip_id}-16k.wav"
    if not wav.is_file():
        command = ["ffmpeg", "-y"]
        if start_seconds:
            command.extend(["-ss", str(start_seconds)])
        if duration_seconds is not None:
            command.extend(["-t", str(duration_seconds)])
        command.extend(["-i", str(media), "-vn", "-ac", "1", "-ar", "16000", str(wav)])
        run_command(command, stage="asr", timeout=3600)
    def transcribe(selected_device: str) -> tuple[list[Evidence], str]:
        engine = WhisperModel(model_name, device=selected_device, compute_type="float16" if selected_device == "cuda" else "int8")
        segments, info = engine.transcribe(str(wav), vad_filter=True, beam_size=5)
        evidence = [Evidence(
            id=f"t{i:05d}", kind="transcript", start_ms=int((start_seconds + segment.start) * 1000),
            end_ms=int((start_seconds + segment.end) * 1000), text=segment.text.strip(),
            origin=f"faster-whisper:{model_label}:{selected_device}:{info.language}",
            warning="低置信转写，需核对" if segment.avg_logprob < -1.0 else "",
        ) for i, segment in enumerate(segments, 1) if segment.text.strip()]
        return evidence, info.language

    try:
        try:
            evidence, language = transcribe(chosen_device)
        except Exception:
            if device != "auto" or chosen_device != "cuda":
                raise
            chosen_device = "cpu"
            evidence, language = transcribe(chosen_device)
        return evidence, f"faster-whisper:{model_label}:{chosen_device}:{language}"
    except Exception as exc:
        raise PipelineError(f"本地转写失败：{exc}", stage="asr") from exc


def remote_asr(media: Path, audio_dir: Path, *, duration_seconds: float, model: ModelConfig, retry_limit: int) -> tuple[list[Evidence], str]:
    if not model.ready:
        raise PipelineError("远程转写未配置", status="blocked", stage="asr")
    audio_dir.mkdir(parents=True, exist_ok=True)
    all_evidence: list[Evidence] = []
    for index, start in enumerate(range(0, max(1, int(duration_seconds)), 300)):
        clip = audio_dir / f"remote-{index:04d}.mp3"
        if not clip.is_file():
            run_command(["ffmpeg", "-y", "-ss", str(start), "-t", "300", "-i", str(media), "-vn", "-ac", "1", "-ar", "16000", "-b:a", "64k", str(clip)], stage="asr", timeout=900)
        boundary = "vts-" + uuid.uuid4().hex
        body = (
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\n{model.model}\r\n"
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"response_format\"\r\n\r\nverbose_json\r\n"
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"audio.mp3\"\r\nContent-Type: audio/mpeg\r\n\r\n"
        ).encode() + clip.read_bytes() + f"\r\n--{boundary}--\r\n".encode()
        request = urllib.request.Request(
            _endpoint(model.base_url, "/audio/transcriptions"), body,
            {"Authorization": f"Bearer {model.key}", "Content-Type": f"multipart/form-data; boundary={boundary}"}, method="POST",
        )
        response = None
        for attempt in range(retry_limit + 1):
            try:
                with urllib.request.urlopen(request, timeout=180) as stream:
                    response = json.load(stream)
                break
            except urllib.error.HTTPError as exc:
                if exc.code in (400, 401, 403, 404, 422):
                    raise PipelineError(f"远程转写接口不可用（HTTP {exc.code}）", status="blocked", stage="asr") from exc
                if attempt == retry_limit:
                    raise PipelineError(f"远程转写失败（HTTP {exc.code}）", stage="asr") from exc
            except (urllib.error.URLError, TimeoutError) as exc:
                if attempt == retry_limit:
                    raise PipelineError(f"远程转写网络失败：{exc}", stage="asr") from exc
            time.sleep(min(8, 2 ** attempt))
        if not isinstance(response, dict) or not isinstance(response.get("segments"), list):
            raise PipelineError("远程转写未返回带时间戳的 segments", status="blocked", stage="asr")
        for segment in response["segments"]:
            text = str(segment.get("text", "")).strip()
            if text:
                all_evidence.append(Evidence(
                    id=f"t{len(all_evidence) + 1:05d}", kind="transcript",
                    start_ms=int((start + float(segment.get("start", 0))) * 1000),
                    end_ms=int((start + float(segment.get("end", 0))) * 1000),
                    text=text, origin=f"remote-asr:{model.model}",
                ))
    return all_evidence, f"remote-asr:{model.model}"


def probe_remote_asr(model: ModelConfig, *, retry_limit: int) -> tuple[bool, str]:
    if not model.ready:
        return False, "未配置远程音频接口"
    import tempfile

    try:
        with tempfile.TemporaryDirectory() as folder:
            tone = Path(folder) / "tone.wav"
            run_command(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "sine=frequency=440:duration=1", str(tone)], stage="doctor", timeout=20)
            remote_asr(tone, Path(folder) / "audio", duration_seconds=1, model=model, retry_limit=retry_limit)
        return True, "音频请求成功，未据此推断转写质量"
    except PipelineError as exc:
        return False, exc.reason
