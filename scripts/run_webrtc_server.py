#!/usr/bin/env python3
import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict

from aiohttp import web


def _repo_root() -> Path:
    # scripts/ -> repo root
    return Path(__file__).resolve().parents[1]


def _configure_logging(level: str) -> None:
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _load_pipeline_config(path: str) -> Dict[str, Any]:
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"pipeline_config JSON not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    if not isinstance(cfg, dict):
        raise ValueError("pipeline_config JSON must be an object at the root.")
    return cfg


def _validate_top_level_keys(cfg: Dict[str, Any]) -> None:
    required = {"vad", "asr", "llm", "tts", "avatar"}
    missing = required - set(cfg.keys())
    if missing:
        raise ValueError(f"pipeline_config JSON is missing keys: {sorted(missing)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="ARACHNE-X WebRTC server runner (production entrypoint)")
    parser.add_argument("--host", type=str, default=os.environ.get("HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8080")))
    parser.add_argument("--pipeline-config", type=str, required=True, help="Path to pipeline_config.json")
    parser.add_argument("--log-level", type=str, default=os.environ.get("LOG_LEVEL", "INFO"))
    parser.add_argument("--graceful-timeout", type=float, default=float(os.environ.get("GRACEFUL_TIMEOUT", "10.0")))
    args = parser.parse_args()

    _configure_logging(args.log_level)
    logger = logging.getLogger("run_webrtc_server")

    # Ensure imports work for `from src....`
    repo_root = _repo_root()
    sys.path.insert(0, str(repo_root))

    pipeline_cfg = _load_pipeline_config(args.pipeline_config)
    _validate_top_level_keys(pipeline_cfg)

    from src.server.webrtc_server import create_app  # delayed import for sys.path setup

    app = create_app(pipeline_cfg)

    # Basic production knobs (aiohttp will still handle SIGTERM/SIGINT defaults).
    app["graceful_timeout"] = args.graceful_timeout

    logger.info("Starting WebRTC server on %s:%s", args.host, args.port)
    web.run_app(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

