from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import load_config
from .core import PipelineError, json_dumps, package_versions
from .media import probe_source
from .pipeline import RunOptions, run_job
from .providers import probe_model, probe_remote_asr


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="video-to-sop", description="把单个视频整理成带证据的 SOP 或结构化文档")
    parser.add_argument("--config", help="本机 TOML 配置文件")
    commands = parser.add_subparsers(dest="command", required=True)
    doctor = commands.add_parser("doctor", help="检查依赖与模型配置")
    doctor.add_argument("--probe-models", action="store_true", help="实际发送最小文本和图片请求验证模型能力")
    inspect = commands.add_parser("inspect", help="只探测来源，不生成内容")
    inspect.add_argument("source")
    inspect.add_argument("--episode", type=int)
    inspect.add_argument("--cookie-file", type=Path)
    run = commands.add_parser("run", help="处理视频并输出文档")
    run.add_argument("source")
    run.add_argument("--output", type=Path, help="本次处理的产物目录")
    run.add_argument("--mode", choices=["auto", "sop", "notes"], default="auto")
    run.add_argument("--format", choices=["md", "html", "pdf"], default="md", help="Markdown 始终生成")
    run.add_argument("--episode", type=int)
    run.add_argument("--cookie-file", type=Path)
    run.add_argument("--no-resume", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        if args.command == "doctor":
            versions = package_versions()
            result = {"python": sys.version.split()[0], "dependencies": versions,
                      "models": {"text": config.text_model.public(), "vision": config.vision_model.public(), "remote_asr": config.remote_asr.public()},
                      "asr_mode": config.asr_mode}
            if args.probe_models:
                result["model_probes"] = {
                    "text": probe_model(config.text_model, vision=False, retry_limit=config.retry_limit),
                    "vision": probe_model(config.vision_model, vision=True, retry_limit=config.retry_limit),
                    "audio": probe_remote_asr(config.remote_asr, retry_limit=config.retry_limit),
                }
            print(json_dumps(result), end="")
            return 0
        if args.command == "inspect":
            probe = probe_source(args.source, episode=args.episode, cookie_file=args.cookie_file)
            print(json_dumps(probe.public()), end="")
            return 0
        result = run_job(config, RunOptions(
            source=args.source, output=args.output, mode=args.mode, format=args.format,
            episode=args.episode, cookie_file=args.cookie_file, resume=not args.no_resume,
        ))
        print(json_dumps(result), end="")
        return {"completed": 0, "partial": 2, "blocked": 3, "failed": 1}[result["status"]]
    except PipelineError as exc:
        print(json_dumps({"status": exc.status, "stage": exc.stage, "reason": exc.reason}), end="")
        return {"blocked": 3, "partial": 2, "failed": 1}.get(exc.status, 1)


if __name__ == "__main__":
    raise SystemExit(main())
