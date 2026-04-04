"""Load and validate pipeline_config.json for WebRTC / orchestrator (VAD → ASR → LLM → TTS → avatar)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

REQUIRED_TOP_LEVEL = frozenset({"vad", "asr", "llm", "tts", "avatar"})


def load_pipeline_config(path: Path | str) -> dict[str, Any]:
    cfg_path = Path(path)
    if not cfg_path.is_file():
        raise FileNotFoundError(f"pipeline_config JSON not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    if not isinstance(cfg, dict):
        raise ValueError("pipeline_config JSON must be an object at the root.")
    return cfg


def validate_top_level_keys(cfg: dict[str, Any]) -> None:
    missing = REQUIRED_TOP_LEVEL - set(cfg.keys())
    if missing:
        raise ValueError(f"pipeline_config JSON is missing keys: {sorted(missing)}")


def resolve_pipeline_config_path(cli_arg: str | None, repo_root: Path) -> Path:
    """
    Order: CLI --pipeline-config, env NULLXES_PIPELINE_CONFIG, then
    config/pipeline_config.defaults.json under repo_root if present.
    """
    if cli_arg:
        return Path(cli_arg)
    env = os.environ.get("NULLXES_PIPELINE_CONFIG", "").strip()
    if env:
        return Path(env)
    default = repo_root / "config" / "pipeline_config.defaults.json"
    if default.is_file():
        return default
    raise FileNotFoundError(
        "Pipeline config not specified: pass --pipeline-config, set NULLXES_PIPELINE_CONFIG, "
        f"or create {default}"
    )
