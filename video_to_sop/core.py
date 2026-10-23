from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


STATUSES = {"completed", "partial", "blocked", "failed"}


class PipelineError(Exception):
    def __init__(self, reason: str, *, status: str = "failed", stage: str = "unknown"):
        super().__init__(reason)
        self.reason = reason
        self.status = status
        self.stage = stage


@dataclass
class Evidence:
    id: str
    kind: str
    start_ms: int
    end_ms: int
    text: str = ""
    path: str = ""
    origin: str = ""
    warning: str = ""


@dataclass
class Step:
    id: str
    depends_on: list[str]
    goal: str
    prerequisites: list[str]
    action: str
    inputs: dict[str, str]
    expected_result: str
    verification: str
    on_failure: str
    evidence_ids: list[str]
    readiness: str
    unknowns: list[str]
    statement_type: str = "shown"
    start_ms: int = 0
    end_ms: int = 0
    command: str = ""
    working_directory: str = ""
    shell: str = ""
    application: str = ""
    page: str = ""
    control: str = ""


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2) + "\n"


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json_dumps(value), encoding="utf-8")
    os.replace(tmp, path)


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, default=str).encode()).hexdigest()


def file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def slug(value: str) -> str:
    value = re.sub(r"[<>:\"/\\|?*\x00-\x1f]", "-", value).strip(" .-")
    value = re.sub(r"\s+", "-", value)
    return value[:64] or "video"


def run_command(args: list[str], *, stage: str, timeout: int = 300) -> str:
    try:
        result = subprocess.run(args, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout, check=False)
    except FileNotFoundError as exc:
        raise PipelineError(f"缺少依赖：{args[0]}", status="blocked", stage=stage) from exc
    except subprocess.TimeoutExpired as exc:
        raise PipelineError(f"{args[0]} 超时", stage=stage) from exc
    if result.returncode:
        message = (result.stderr or result.stdout).strip()[-1200:]
        raise PipelineError(f"{args[0]} 失败：{message}", stage=stage)
    return result.stdout


def tool_version(name: str) -> str | None:
    if not shutil.which(name):
        return None
    try:
        return subprocess.run([name, "-version"], capture_output=True, text=True, timeout=10).stdout.splitlines()[0]
    except (OSError, subprocess.TimeoutExpired, IndexError):
        return None


def ms_label(ms: int) -> str:
    seconds = max(0, ms) // 1000
    return f"{seconds // 3600:02d}:{seconds // 60 % 60:02d}:{seconds % 60:02d}"


def asdict_list(items: list[Any]) -> list[dict[str, Any]]:
    return [asdict(item) for item in items]


def package_versions() -> dict[str, str | None]:
    from importlib import metadata

    result = {}
    for package in ("yt-dlp", "webvtt-py", "faster-whisper", "rapidocr-onnxruntime", "markdown-it-py", "playwright"):
        try:
            result[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            result[package] = None
    result["ffmpeg"] = tool_version("ffmpeg")
    result["ffprobe"] = tool_version("ffprobe")
    return result
