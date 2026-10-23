# video-to-sop

Windows-first video-to-document Skill with an independent Python CLI. It probes one Bilibili, YouTube, or local video, selects timed text evidence, samples frames for tutorial verification, and writes an AI handoff document. It never executes video instructions.

For a one-folder Windows x64 package with a bundled Python runtime, FFmpeg, local ASR model, OCR, and PDF browser, use the separately built portable ZIP and follow `README-PORTABLE.md`. The source ZIP in this folder is small but requires installing its dependencies.

## Install on Windows

1. Install Python 3.12+, `uv`, and FFmpeg/ffprobe. Ensure `ffmpeg -version` and `ffprobe -version` work in PowerShell.
2. From this folder run `uv sync --all-extras`, then `uv run --no-sync playwright install chromium`. The optional ASR and OCR downloads can be substantial; `uv sync` installs the smaller text-only path.
3. Copy `config.example.toml` to a local, untracked location and fill in the exact DeepSeek model and endpoint available on the running machine. Set `VTS_TEXT_API_KEY` in the local environment. Set vision and remote ASR endpoints and key environment variables only if those capabilities are permitted and available. The CLI does not read Hermes' private credentials automatically.
4. Run `uv run --no-sync video-to-sop --config C:\path\to\local-config.toml doctor --probe-models`. The optional probe sends tiny text, image, and audio requests to configured endpoints. Without `--probe-models`, doctor only checks local dependencies and configuration presence.

If the destination Hermes version supports a project-local skill folder, install this folder in its documented skills location or link it according to that version's instructions. Do not hard-code a Hermes path. Verify invocation with a natural-language request on the destination machine.

## Use

```powershell
uv run --no-sync video-to-sop --config C:\path\to\local-config.toml inspect "https://www.bilibili.com/video/BV..."
uv run --no-sync video-to-sop --config C:\path\to\local-config.toml run "C:\Videos\示例 教程.mp4" --mode sop --format pdf
uv run --no-sync video-to-sop --config C:\path\to\local-config.toml run "https://www.youtube.com/watch?v=..." --mode auto --format md
```

`--output` chooses an exact job directory. Otherwise a stable per-source directory is created under `output_dir`. `--episode N` selects one-based playlist entry. `--cookie-file` accepts an explicit local Cookie file; the tool never scans browsers. Resume is enabled by default; `--no-resume` reprocesses stages. Exit codes: 0 completed, 2 partial, 3 blocked, 1 failed.

Markdown is always generated. `--format html` also writes HTML; `--format pdf` writes HTML and PDF. A PDF failure leaves Markdown and marks the job partial. Outputs: `report.md`, `steps.json`, `transcript.json`, `transcript.md`, `manifest.json`, and referenced `frames/`; optional `report.html` and `report.pdf`.

## Processing limits

- Metadata detection is not proof that subtitles or media can be fetched. Acquisition failures are recorded separately from confirmed absence.
- Automatic subtitles may repeat or omit words. The parser removes rolling duplicate text and records suspicious gaps without calling them definite omissions. Commands and parameters still require evidence review.
- Sampling cannot capture every click. A single screenshot cannot prove an action. Missing vision capability makes tutorial output partial.
- The OpenAI-compatible text/vision and transcription adapters cover common interfaces. A company proxy with a different wire protocol needs an adapter; text API availability does not imply image or audio support.
- Local faster-whisper may download model weights on first use. GPU is selected only when the local runtime detects CUDA; CPU remains available. No local GPU is inferred from a remote API token.
- The lock constrains PyAV below 18 because faster-whisper 1.2.1 still calls a decoder argument removed by newer PyAV releases. Revalidate this bound when a newer faster-whisper release is adopted.
- Output is a handoff document, not permission to execute. `readiness=信息足够` means the source evidence supports the written step, not that it has been tested on the target machine.

## Tests and packaging

Run `uv sync --extra test` followed by `uv run --no-sync pytest` for the core suite. To include the PDF and all optional integrations, install all extras and run with Playwright Chromium available. Synthetic tests do not count as Bilibili or YouTube platform verification. `TEST-RECORD.md` reports the execution environment and any tests blocked by network or dependencies. The distributable archive contains source, tests, templates and docs, not generated media, caches, Cookies or credentials.
