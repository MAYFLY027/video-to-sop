# Test record - 2026-10-23

## Environment

- Windows; Python 3.12.10; uv 0.10.7.
- FFmpeg/ffprobe 9.0.2 portable build used for tests; not included in the archive.
- Locked dependencies include yt-dlp 2026.10.21, faster-whisper 1.2.1, PyAV 17.1.0, rapidocr-onnxruntime 1.4.4, markdown-it-py 4.2.0, and Playwright 1.64.1. This sandbox required a workspace-local `uv` cache; the standard per-user cache was not writable here.
- Hermes CLI and a real DeepSeek endpoint were unavailable in this workspace. Their actual invocation, credentials, response format and model capability remain unverified on the destination machine.
- Final local test rerun: `python -m pytest tests -q --basetemp <workspace-test-dir>` passed all 22 tests. The default Windows user temp directory denied pytest fixture creation in this environment, so a workspace-local base temp directory was used.
- Portable packaging rerun: after adding config-relative asset resolution and CUDA-to-CPU ASR fallback, the source environment passed 24 tests; the bundled embedded-Python environment also passed 24 tests using its own FFmpeg and Playwright headless shell. An eight-second public-video excerpt with a Chinese/space filename produced one CPU ASR segment, a frame, OCR data, Markdown, HTML and PDF from the portable launcher. The report remained partial because no DeepSeek text or vision endpoint was configured.
- Archive relocation check: the full ZIP was extracted to a new path containing Chinese characters and a space. With `PATH` limited to Windows system directories before launcher execution, `doctor` found the bundled tools. The extracted launcher processed the local excerpt with the packaged ASR model and PDF browser, writing all artifacts under its own `video-to-sop-output/` directory. Hermes natural-language registration and live DeepSeek capability are still destination-machine checks.

## Results

| Test | Result | Evidence and limit |
| --- | --- | --- |
| Synthetic routing and handoff | Passed | 22 pytest cases: subtitle priority and rolling cleanup, gap ASR, metadata-only blocking, long chunk overlap and merge, cache reuse, independent text/image/audio probes, silent video, second-pass frame sampling, UI evidence requirements, Chinese/space paths, PDF. |
| Public YouTube `NxiRMqcHipE` | Partial, real platform access verified | yt-dlp obtained 44 timed English automatic-caption segments. Text model was not configured, so no content interpretation or full SOP claim. |
| Public Bilibili `BV1bK411W797` part 1 | Partial, real media path verified | Download, ffprobe, seven sampled frames, and OCR ran. This music video has no listed subtitle; ASR was disabled in this run. |
| Public Bilibili `BV1gcFMzREu2` | Partial, real no-subtitle route verified | 43-second tutorial downloaded; local faster-whisper tiny on CPU produced 27 timestamped segments and six frames. Some Chinese terms were misrecognized, proving the need for the unreviewed-ASR warning and visual term check. |
| Public Bilibili `BV1JPHn62Era` | Partial, real 10:40 content test | No subtitle tracks were listed; media was acquired and local faster-whisper small on CPU generated 419 timestamped segments. The pipeline sampled 60 frames and ran OCR. A separate human-reviewed, timestamped notes report was prepared from the transcript and sampled imagery; the video is an illustrated experience account, so no setup steps were fabricated. Text and vision model endpoints were unavailable, and their synthesis path remains unverified on this video. |
| PDF visual inspection | Passed on synthetic documents | Rendered pages show Chinese text, a table, a long wrapped command, and referenced images without clipping or overlap. Full pipeline generated a two-page PDF with screenshot evidence and Chinese/space filenames. |
| Resume | Passed | A second synthetic run reused synthesis cache; the real YouTube rerun reused its caption and recorded why synthesis was rerun after a prompt-schema change. |

The public test videos and generated documents remain local test data outside this archive. No video contents, cookies, tokens or model responses from private services are packaged.

## Remaining destination-machine checks

1. Install and run in the actual Windows + Hermes environment; verify natural-language activation and artifact links.
2. Configure the exact DeepSeek text model and run a real synthesis request. Test image input separately on the configured vision endpoint; do not infer it from text success.
3. Test a permitted real sample through full text + visual synthesis and inspect the step-level evidence and PDF. The synthetic fake-model result tests the interface and pipeline, not model accuracy.
4. Review ASR model size/quality on representative Chinese technical speech and check model download size, runtime, and GPU availability.
