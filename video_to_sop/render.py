from __future__ import annotations

import html
import json
from pathlib import Path

from .core import Evidence, PipelineError, atomic_json, ms_label


def _safe_markdown(value: object) -> str:
    return str(value or "").replace("\r", " ").strip()


def _bullet_lines(items: list[str]) -> list[str]:
    return [f"- {_safe_markdown(item)}" for item in items] or ["- 无已确认信息"]


def write_transcript(job_dir: Path, transcript: list[Evidence], reason: str = "") -> None:
    atomic_json(job_dir / "transcript.json", {"segments": [vars(item) for item in transcript], "reason_if_empty": reason if not transcript else ""})
    lines = ["# 带时间戳的文字证据", ""]
    if not transcript:
        lines.extend([f"无可用语音或字幕证据。原因：{reason or '未取得文字证据'}", ""])
    else:
        for item in transcript:
            lines.append(f"- [{ms_label(item.start_ms)}–{ms_label(item.end_ms)}] ({item.id}; {item.origin}) {item.text}")
    (job_dir / "transcript.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report(job_dir: Path, probe: dict, document: dict, transcript: list[Evidence], frames: list[Evidence], warnings: list[str], status: str) -> None:
    evidence = {item.id: item for item in [*transcript, *frames]}
    title = _safe_markdown(probe.get("title") or "视频文档")
    lines = [f"# {title}", "", f"**状态：{status}**  |  **文档类型：{document['mode']}**", "", f"来源：{_safe_markdown(probe.get('webpage_url') or probe.get('source'))}", ""]
    if probe.get("duration_seconds") is not None:
        lines.extend([f"时长：{ms_label(int(float(probe['duration_seconds']) * 1000))}", ""])
    lines.extend(["## 目的与内容", "", document.get("summary") or "未能生成内容理解；请参阅转写与状态说明。", ""])
    lines.extend(["## 执行前核对", "", "- 核对目标机器的应用版本、系统、路径、权限和网络条件；视频环境不等于目标环境。", "- 将示例账号、路径、版本和配置值与目标环境真实值分开；未知值先确认。", "- 本报告只提取和整理证据，未执行教程操作。", ""])
    if document.get("mode") in {"sop", "mixed"}:
        lines.extend(["## 操作步骤", ""])
        if not document.get("steps"):
            lines.extend(["尚无证据足够的操作步骤。", ""])
        for step in document.get("steps", []):
            lines.extend([f"### {step['id']} {step['goal'] or step['action']}", "", f"**就绪程度：{step['readiness']}**；**陈述类型：{step['statement_type']}**", ""])
            if step.get("depends_on"):
                lines.extend([f"依赖步骤：{', '.join(step['depends_on'])}", ""])
            if step.get("prerequisites"):
                lines.extend(["前置条件：", *_bullet_lines(step["prerequisites"]), ""])
            lines.extend([f"操作：{step['action'] or '待确认'}", ""])
            if step.get("command"):
                lines.extend([f"终端：{step.get('shell') or '待确认'}；执行目录：{step.get('working_directory') or '待确认'}", "", "```text", step["command"], "```", ""])
            if any(step.get(key) for key in ("application", "page", "control")):
                lines.extend([f"界面：{step.get('application') or '待确认'} / {step.get('page') or '待确认'} / {step.get('control') or '待确认'}", ""])
            if step.get("inputs"):
                lines.extend(["输入值／变量：", *[f"- `{key}`：{value}" for key, value in step["inputs"].items()], ""])
            lines.extend([f"预期结果：{step['expected_result'] or '待确认'}", "", f"验证方法：{step['verification'] or '待确认'}", ""])
            if step.get("on_failure"):
                lines.extend([f"失败后：{step['on_failure']}", ""])
            if step.get("unknowns"):
                lines.extend(["待确认：", *_bullet_lines(step["unknowns"]), ""])
            if step.get("evidence_ids"):
                lines.append("证据：")
                for eid in step["evidence_ids"]:
                    item = evidence.get(eid)
                    if not item:
                        continue
                    lines.append(f"- {eid} [{ms_label(item.start_ms)}–{ms_label(item.end_ms)}] {item.origin}")
                    if item.kind == "frame" and item.path:
                        lines.extend(["", f"![截图 {eid}]({item.path})", ""])
                lines.append("")
    if document.get("facts"):
        lines.extend(["## 讲解与背景", ""])
        for fact in document["facts"]:
            lines.append(f"- [{fact.get('statement_type', 'inferred')}] {_safe_markdown(fact['text'])}（证据：{', '.join(fact.get('evidence_ids', [])) or '待确认'}）")
        lines.append("")
    lines.extend(["## 交给执行 AI 的说明", "", "先读取前置条件和待确认项，补齐目标机器上的本地变量。按依赖顺序执行，并逐步检查预期结果；关键证据缺失时先停下核实。此文档不授予执行权限，实际操作范围由届时用户决定。", ""])
    lines.extend(["## 时间线与未处理区间", ""])
    for interval in document.get("intervals", []):
        lines.append(f"- {ms_label(interval['start_ms'])}–{ms_label(interval['end_ms'])}：{'已提取' if interval['processed'] else '未完成理解'}")
    if not document.get("intervals"):
        lines.append("- 无可用分段证据")
    lines.append("")
    issues = [*warnings, *document.get("uncertainties", [])]
    lines.extend(["## 限制与待确认", "", *_bullet_lines(list(dict.fromkeys(issues))), ""])
    (job_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def render_html(job_dir: Path) -> Path:
    try:
        from markdown_it import MarkdownIt
    except ImportError as exc:
        raise PipelineError("缺少 markdown-it-py，无法导出 HTML", status="blocked", stage="export") from exc
    body = MarkdownIt("commonmark", {"html": False, "linkify": False}).enable("table").render((job_dir / "report.md").read_text(encoding="utf-8"))
    css = """@page{size:A4;margin:17mm}body{font:14px/1.6 'Microsoft YaHei','Noto Sans CJK SC',sans-serif;color:#20252b;max-width:900px;margin:auto;padding:24px}h1,h2,h3{line-height:1.25;break-after:avoid}h2{border-bottom:1px solid #aaa;padding-bottom:4px}img{display:block;max-width:100%;max-height:520px;object-fit:contain;break-inside:avoid;margin:8px 0}p:has(img){break-inside:avoid}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f3f4f5;padding:10px}code{overflow-wrap:anywhere}li{break-inside:avoid}table{border-collapse:collapse;width:100%}td,th{border:1px solid #bbb;padding:5px;overflow-wrap:anywhere}"""
    page = f"<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'><title>video-to-sop report</title><style>{css}</style></head><body>{body}</body></html>"
    path = job_dir / "report.html"
    path.write_text(page, encoding="utf-8")
    return path


def render_pdf(job_dir: Path) -> Path:
    html_path = render_html(job_dir)
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise PipelineError("缺少 Playwright，无法导出 PDF", status="blocked", stage="export") from exc
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            page = browser.new_page()
            page.route("https://**/*", lambda route: route.abort())
            page.route("http://**/*", lambda route: route.abort())
            page.goto(html_path.resolve().as_uri(), wait_until="load")
            image_status = page.locator("img").evaluate_all("items => items.every(image => image.complete && image.naturalWidth > 0)")
            if not image_status:
                raise PipelineError("PDF 导出前发现截图未加载", stage="export")
            path = job_dir / "report.pdf"
            page.pdf(path=str(path), format="A4", print_background=True)
            browser.close()
            return path
    except PipelineError:
        raise
    except Exception as exc:
        raise PipelineError(f"PDF 导出失败：{exc}", stage="export") from exc
