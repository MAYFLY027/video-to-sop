from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .core import PipelineError


@dataclass
class ModelConfig:
    base_url: str = ""
    model: str = ""
    api_key_env: str = ""

    @property
    def key(self) -> str:
        return os.environ.get(self.api_key_env, "") if self.api_key_env else ""

    @property
    def ready(self) -> bool:
        return bool(self.base_url and self.model and self.key)

    def public(self) -> dict[str, str | bool]:
        return {"base_url": self.base_url, "model": self.model, "api_key_env": self.api_key_env, "key_present": bool(self.key)}


@dataclass
class Config:
    output_dir: str = "video-to-sop-output"
    output_language: str = "zh-CN"
    source_languages: list[str] = field(default_factory=lambda: ["zh", "en"])
    max_duration_minutes: int = 120
    sample_interval_seconds: int = 15
    max_frames: int = 300
    vision_max_frames: int = 30
    concurrency: int = 2
    retry_limit: int = 2
    asr_mode: str = "local"
    asr_model: str = "small"
    asr_device: str = "auto"
    text_model: ModelConfig = field(default_factory=ModelConfig)
    vision_model: ModelConfig = field(default_factory=ModelConfig)
    remote_asr: ModelConfig = field(default_factory=ModelConfig)

    def public(self) -> dict[str, Any]:
        values = vars(self).copy()
        for name in ("text_model", "vision_model", "remote_asr"):
            values[name] = getattr(self, name).public()
        return values


def load_config(path: str | None) -> Config:
    raw: dict[str, Any] = {}
    if path:
        config_path = Path(path).expanduser().resolve()
        try:
            raw = tomllib.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, tomllib.TOMLDecodeError) as exc:
            raise PipelineError(f"配置文件不可读取：{exc}", status="blocked", stage="config") from exc
        for name in ("output_dir", "asr_model"):
            value = raw.get(name)
            if isinstance(value, str) and (name == "output_dir" or "/" in value or "\\" in value):
                candidate = Path(value).expanduser()
                if not candidate.is_absolute() and (name == "output_dir" or (config_path.parent / candidate).is_dir()):
                    raw[name] = str((config_path.parent / candidate).resolve())
    allowed = set(Config.__dataclass_fields__)
    unknown = set(raw) - allowed
    if unknown:
        raise PipelineError(f"未知配置项：{', '.join(sorted(unknown))}", status="blocked", stage="config")
    model_names = {"text_model", "vision_model", "remote_asr"}
    for name in model_names:
        raw[name] = ModelConfig(**raw.get(name, {}))
    config = Config(**raw)
    if config.asr_mode not in {"local", "remote", "off"}:
        raise PipelineError("asr_mode 必须是 local、remote 或 off", status="blocked", stage="config")
    if config.asr_device not in {"auto", "cpu", "cuda"}:
        raise PipelineError("asr_device 必须是 auto、cpu 或 cuda", status="blocked", stage="config")
    for name in ("max_duration_minutes", "sample_interval_seconds", "max_frames", "vision_max_frames", "concurrency"):
        if getattr(config, name) <= 0:
            raise PipelineError(f"{name} 必须大于 0", status="blocked", stage="config")
    if not 0 <= config.retry_limit <= 5:
        raise PipelineError("retry_limit 必须在 0 到 5 之间", status="blocked", stage="config")
    return config
