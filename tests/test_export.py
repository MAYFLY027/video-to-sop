from __future__ import annotations

import base64
from pathlib import Path

import pytest

from video_to_sop.render import render_html, render_pdf


def _sample(job_dir: Path) -> None:
    frames = job_dir / "frames"
    frames.mkdir(parents=True)
    try:
        from PIL import Image, ImageDraw

        canvas = Image.new("RGB", (640, 360), "#eef2f5")
        draw = ImageDraw.Draw(canvas)
        draw.rectangle((45, 35, 595, 325), fill="#ffffff", outline="#415466", width=3)
        draw.text((85, 75), "Settings", fill="#152a3b")
        draw.rectangle((385, 235, 540, 280), fill="#1d7468")
        draw.text((435, 250), "Save", fill="#ffffff")
        canvas.save(frames / "sample.png")
    except ImportError:
        (frames / "sample.png").write_bytes(base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQIHWP4z8DwHwAFgAI/ScL/nwAAAABJRU5ErkJggg=="
        ))
    long_command = "python -m tool --path C:/项目/输出 --flag " + "argument-" * 30
    (job_dir / "report.md").write_text(
        "# 中文 SOP 测试\n\n## 操作\n\n"
        "| 参数 | 值 |\n| --- | --- |\n| 输出 | C:/项目/输出 |\n\n"
        f"```text\n{long_command}\n```\n\n![截图](frames/sample.png)\n",
        encoding="utf-8",
    )


def test_html_keeps_table_command_and_relative_image(tmp_path: Path) -> None:
    _sample(tmp_path)
    html = render_html(tmp_path).read_text(encoding="utf-8")
    assert "<table>" in html
    assert "argument-" in html
    assert 'src="frames/sample.png"' in html


def test_pdf_with_chinese_table_long_command_and_image(tmp_path: Path) -> None:
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
    _sample(tmp_path)
    pdf = render_pdf(tmp_path)
    assert pdf.read_bytes().startswith(b"%PDF-")
    assert pdf.stat().st_size > 5000
