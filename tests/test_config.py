from __future__ import annotations

from pathlib import Path

from video_to_sop.config import load_config


def test_relative_asset_and_output_paths_follow_config_location(tmp_path: Path) -> None:
    skill = tmp_path / "中文 skill"
    model = skill / "assets" / "asr" / "small"
    model.mkdir(parents=True)
    config_path = skill / "config.local.toml"
    config_path.write_text('output_dir = "video-to-sop-output"\nasr_model = "assets/asr/small"\n', encoding="utf-8")

    config = load_config(str(config_path))
    assert config.output_dir == str(skill / "video-to-sop-output")
    assert config.asr_model == str(model)
