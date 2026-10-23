# Portable Windows package

This package is a single Windows x64 Skill folder. Extract the outer ZIP, keep its directory structure, and run `video-to-sop.cmd doctor` from PowerShell or Command Prompt. It contains Python 3.12, locked Python dependencies, FFmpeg/ffprobe, the CPU-compatible faster-whisper small model, RapidOCR assets, and a Playwright headless browser. No system-wide Python, uv, FFmpeg, ASR model download, or browser install is required.

On first run the launcher creates `config.local.toml` inside this folder from `config.example.toml`. Set the exact text model name and endpoint used by your DeepSeek provider there, and put its key in the local `VTS_TEXT_API_KEY` environment variable. The launcher does not read Hermes credentials. Add a separately verified vision endpoint only when image input works; text success does not establish vision support. Do not put API keys or Cookies in the Skill folder.

```powershell
cd "C:\path\to\video-to-sop"
.\video-to-sop.cmd doctor
.\video-to-sop.cmd run "C:\Videos\Example.mp4" --mode auto --format md
.\video-to-sop.cmd run "https://www.bilibili.com/video/BV..." --mode sop --format pdf
```

To test model connectivity, run `video-to-sop.cmd doctor --probe-models`. That sends small requests to the configured services. Use `--cookie-file` only with an explicitly provided local Cookie file. Outputs and caches go under `video-to-sop-output/` inside this folder by default. Move the entire folder as a unit; its launcher uses paths relative to its own location.

This package runs on Windows x64. Internet access is still needed for online videos and configured model services. Login-gated videos still need authorized access. It does not make DeepSeek support images or audio, and it does not guarantee every platform link is downloadable. Offline local videos and packaged ASR/OCR/PDF dependencies do not need a first-run dependency download.

Package contents and licensing: `runtime/python` is the official CPython 3.12.10 Windows embeddable distribution; `runtime/ffmpeg` contains FFmpeg/ffprobe 9.0.2 from the Windows essentials build; `runtime/browsers` contains the Playwright Chromium headless shell; `assets/asr/small` contains the Systran faster-whisper-small model. See `THIRD_PARTY.md` and license files under `licenses/` before redistribution outside your organization.
