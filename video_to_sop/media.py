from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from urllib.parse import urlparse

from .core import PipelineError, file_sha256, run_command, slug


@dataclass
class SourceProbe:
    source: str
    kind: str
    source_id: str
    title: str
    duration_seconds: float | None
    webpage_url: str = ""
    description: str = ""
    original_language: str = ""
    subtitles: dict = field(default_factory=dict)
    automatic_captions: dict = field(default_factory=dict)
    has_audio: bool | None = None
    has_video: bool | None = None
    is_live: bool = False
    upload_date: str = ""
    local_path: str = ""
    media_fingerprint: str = ""

    def public(self) -> dict:
        data = asdict(self)
        data["subtitles"] = {key: [item.get("ext", "") for item in tracks] for key, tracks in self.subtitles.items()}
        priority = [key for key in self.automatic_captions if key == self.original_language or key.startswith(("zh", "en"))]
        selected = list(dict.fromkeys([*priority, *sorted(self.automatic_captions)[:20]]))[:30]
        data["automatic_captions"] = {key: [item.get("ext", "") for item in self.automatic_captions[key]] for key in selected}
        data["automatic_caption_language_count"] = len(self.automatic_captions)
        data["description"] = self.description[:1000]
        return data


def is_supported_url(source: str) -> bool:
    url = urlparse(source)
    host = (url.hostname or "").lower()
    return url.scheme in {"http", "https"} and (
        host in {"bilibili.com", "www.bilibili.com", "m.bilibili.com", "b23.tv", "youtube.com", "www.youtube.com", "m.youtube.com", "youtu.be"}
    )


def ffprobe(path: Path) -> dict:
    output = run_command(
        ["ffprobe", "-v", "error", "-show_format", "-show_streams", "-of", "json", str(path)],
        stage="probe",
        timeout=60,
    )
    try:
        return json.loads(output)
    except json.JSONDecodeError as exc:
        raise PipelineError("ffprobe 返回无效 JSON", stage="probe") from exc


def probe_local(path: Path) -> SourceProbe:
    if not path.is_file():
        raise PipelineError(f"本地文件不存在：{path}", status="blocked", stage="probe")
    data = ffprobe(path)
    streams = data.get("streams", [])
    try:
        duration = float(data.get("format", {}).get("duration"))
    except (ValueError, TypeError):
        duration = None
    fingerprint = file_sha256(path)
    return SourceProbe(
        source=str(path), kind="local", source_id=f"local:{fingerprint[:16]}", title=path.stem,
        duration_seconds=duration, has_audio=any(s.get("codec_type") == "audio" for s in streams),
        has_video=any(s.get("codec_type") == "video" for s in streams),
        local_path=str(path.resolve()), media_fingerprint=fingerprint,
    )


def _yt_dlp():
    try:
        import yt_dlp
    except ImportError as exc:
        raise PipelineError("缺少 yt-dlp；请先安装基础依赖", status="blocked", stage="probe") from exc
    return yt_dlp


def _classify_download_error(exc: Exception, stage: str) -> PipelineError:
    message = str(exc)
    lower = message.lower()
    if any(term in lower for term in ("sign in", "login", "log in", "cookies", "private", "members-only", "authentication", "403 forbidden")):
        return PipelineError("访问需要登录或授权；请在本机显式提供 Cookie 文件或可访问的视频文件", status="blocked", stage=stage)
    if any(term in lower for term in ("unsupported url", "video unavailable", "unavailable", "not available", "404", "removed")):
        return PipelineError(f"媒体不可访问：{message[-300:]}", status="blocked", stage=stage)
    return PipelineError(f"平台获取失败：{message[-400:]}", stage=stage)


def _opts(cookie_file: Path | None = None) -> dict:
    options = {"quiet": True, "no_warnings": True, "ignoreerrors": False, "socket_timeout": 30, "retries": 2, "fragment_retries": 2}
    if cookie_file:
        if not cookie_file.is_file():
            raise PipelineError(f"Cookie 文件不存在：{cookie_file}", status="blocked", stage="probe")
        options["cookiefile"] = str(cookie_file)
    return options


def probe_url(source: str, *, episode: int | None = None, cookie_file: Path | None = None) -> SourceProbe:
    yt_dlp = _yt_dlp()
    try:
        with yt_dlp.YoutubeDL({**_opts(cookie_file), "skip_download": True}) as ydl:
            info = ydl.extract_info(source, download=False)
            if not info:
                raise PipelineError("平台未返回媒体元数据", status="blocked", stage="probe")
            if info.get("_type") == "playlist" or info.get("entries"):
                entries = [item for item in info.get("entries", []) if item]
                if episode is None:
                    raise PipelineError(f"检测到合集（{len(entries)} 集）；请用 --episode 指定从 1 开始的集数", status="blocked", stage="probe")
                if not 1 <= episode <= len(entries):
                    raise PipelineError(f"--episode 必须在 1 到 {len(entries)} 之间", status="blocked", stage="probe")
                info = entries[episode - 1]
                if info.get("_type") == "url" or not info.get("formats"):
                    info = ydl.extract_info(info.get("webpage_url") or info.get("url"), download=False)
    except PipelineError:
        raise
    except Exception as exc:
        raise _classify_download_error(exc, "probe") from exc
    if info.get("is_live") or info.get("live_status") in {"is_live", "is_upcoming"}:
        raise PipelineError("第一版只处理已发布的非直播视频", status="blocked", stage="probe")
    formats = info.get("formats", [])
    return SourceProbe(
        source=source, kind="url", source_id=f"{info.get('extractor_key', 'web')}:{info.get('id', '')}",
        title=info.get("title") or "video", duration_seconds=info.get("duration"),
        webpage_url=info.get("webpage_url") or source, description=info.get("description") or "",
        original_language=info.get("language") or "", subtitles=info.get("subtitles") or {},
        automatic_captions=info.get("automatic_captions") or {},
        has_audio=any(f.get("acodec") not in (None, "none") for f in formats) if formats else None,
        has_video=any(f.get("vcodec") not in (None, "none") for f in formats) if formats else None,
        upload_date=info.get("upload_date") or "",
    )


def probe_source(source: str, *, episode: int | None = None, cookie_file: Path | None = None) -> SourceProbe:
    if is_supported_url(source):
        return probe_url(source, episode=episode, cookie_file=cookie_file)
    if urlparse(source).scheme in {"http", "https"}:
        raise PipelineError("第一版仅支持 B站、YouTube 链接或本地文件", status="blocked", stage="probe")
    return probe_local(Path(source).expanduser())


def pick_caption(probe: SourceProbe, languages: list[str]) -> tuple[str, str, str] | None:
    ranked = []
    language_order = languages
    for source_type, pool in (("manual", probe.subtitles), ("automatic", probe.automatic_captions)):
        for language, formats in pool.items():
            for fmt in formats:
                extension = fmt.get("ext", "")
                if extension not in {"vtt", "srt", "json3"}:
                    continue
                language_rank = next((i for i, wanted in enumerate(language_order) if wanted and language.lower().startswith(wanted.lower())), len(language_order))
                original_match = bool(probe.original_language and language.lower().startswith(probe.original_language.lower()))
                format_rank = {"vtt": 0, "srt": 1, "json3": 2}[extension]
                ranked.append(((0 if original_match else 1, 0 if source_type == "manual" else 1, language_rank, format_rank), language, source_type, extension))
    if not ranked:
        return None
    _, language, source_type, extension = min(ranked)
    return language, source_type, extension


def local_caption(path: Path) -> Path | None:
    for extension in (".vtt", ".srt", ".json3"):
        candidate = path.with_suffix(extension)
        if candidate.is_file():
            return candidate
    return None


def download_caption(probe: SourceProbe, choice: tuple[str, str, str], dest: Path, cookie_file: Path | None) -> Path:
    yt_dlp = _yt_dlp()
    language, source_type, extension = choice
    dest.mkdir(parents=True, exist_ok=True)
    existing = list(dest.glob("caption.*"))
    if existing:
        return existing[0]
    options = {
        **_opts(cookie_file), "skip_download": True, "writesubtitles": source_type == "manual",
        "writeautomaticsub": source_type == "automatic", "subtitleslangs": [re.escape(language)],
        "subtitlesformat": extension, "outtmpl": str(dest / "caption.%(ext)s"),
    }
    try:
        with yt_dlp.YoutubeDL(options) as ydl:
            ydl.download([probe.webpage_url])
    except Exception as exc:
        raise _classify_download_error(exc, "caption") from exc
    candidates = [p for p in dest.glob("caption.*") if p.suffix in {".vtt", ".srt", ".json3"}]
    if not candidates:
        raise PipelineError("字幕元数据存在，但字幕文件获取失败", stage="caption")
    return candidates[0]


def download_media(probe: SourceProbe, dest: Path, cookie_file: Path | None) -> Path:
    yt_dlp = _yt_dlp()
    dest.mkdir(parents=True, exist_ok=True)
    existing = [p for p in dest.glob("media.*") if p.suffix.lower() in {".mp4", ".mkv", ".webm", ".mov", ".m4a", ".mp3"}]
    if existing:
        return existing[0]
    options = {
        **_opts(cookie_file), "format": "bestvideo[height<=720]+bestaudio/best[height<=720]/best",
        "outtmpl": str(dest / "media.%(ext)s"), "merge_output_format": "mp4",
        "noplaylist": True,
    }
    try:
        with yt_dlp.YoutubeDL(options) as ydl:
            ydl.download([probe.webpage_url])
    except Exception as exc:
        raise _classify_download_error(exc, "media") from exc
    candidates = [p for p in dest.glob("media.*") if p.suffix.lower() in {".mp4", ".mkv", ".webm", ".mov", ".m4a", ".mp3"}]
    if not candidates:
        raise PipelineError("平台返回元数据，但媒体文件获取失败", stage="media")
    return candidates[0]
