"""pipeline_config.json load/validate used by scripts/run_webrtc_server.py."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.server.pipeline_config import (
    load_pipeline_config,
    resolve_pipeline_config_path,
    validate_top_level_keys,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_defaults_json_loads_and_has_required_keys():
    path = _repo_root() / "config" / "pipeline_config.defaults.json"
    cfg = load_pipeline_config(path)
    validate_top_level_keys(cfg)
    assert cfg.get("schema_version") == 1
    assert (cfg.get("asr") or {}).get("backend") == "faster_whisper"
    assert (cfg.get("avatar") or {}).get("backend") == "longcat_worker_http"


def test_runpod_example_loads():
    path = _repo_root() / "config" / "pipeline_config.runpod.example.json"
    cfg = load_pipeline_config(path)
    validate_top_level_keys(cfg)


def test_validate_top_level_keys_missing():
    with pytest.raises(ValueError, match="missing keys"):
        validate_top_level_keys({"vad": {}, "asr": {}, "llm": {}})


def test_resolve_pipeline_config_path_cli_wins(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    repo = _repo_root()
    p = tmp_path / "x.json"
    p.write_text('{"vad":{},"asr":{},"llm":{},"tts":{},"avatar":{}}', encoding="utf-8")
    monkeypatch.delenv("NULLXES_PIPELINE_CONFIG", raising=False)
    assert resolve_pipeline_config_path(str(p), repo) == p


def test_resolve_pipeline_config_path_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    repo = _repo_root()
    p = tmp_path / "env.json"
    p.write_text('{"vad":{},"asr":{},"llm":{},"tts":{},"avatar":{}}', encoding="utf-8")
    monkeypatch.setenv("NULLXES_PIPELINE_CONFIG", str(p))
    assert resolve_pipeline_config_path(None, repo) == p


def test_resolve_fallback_to_defaults():
    repo = _repo_root()
    got = resolve_pipeline_config_path(None, repo)
    assert got.name == "pipeline_config.defaults.json"
    assert got.is_file()
