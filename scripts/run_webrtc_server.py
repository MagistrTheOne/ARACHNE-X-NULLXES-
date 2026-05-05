#!/usr/bin/env python3
import argparse
import logging
import os
import sys
from pathlib import Path

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


def main() -> None:
    parser = argparse.ArgumentParser(description="ARACHNE-X WebRTC server runner (production entrypoint)")
    parser.add_argument("--host", type=str, default=os.environ.get("HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8080")))
    parser.add_argument(
        "--pipeline-config",
        type=str,
        default=None,
        help="Path to pipeline_config.json (else NULLXES_PIPELINE_CONFIG or config/pipeline_config.defaults.json)",
    )
    parser.add_argument("--log-level", type=str, default=os.environ.get("LOG_LEVEL", "INFO"))
    parser.add_argument("--graceful-timeout", type=float, default=float(os.environ.get("GRACEFUL_TIMEOUT", "10.0")))
    args = parser.parse_args()

    _configure_logging(args.log_level)
    logger = logging.getLogger("run_webrtc_server")

    # Ensure imports work for `from src....`
    repo_root = _repo_root()
    sys.path.insert(0, str(repo_root))

    from src.server.pipeline_config import (
        load_pipeline_config,
        resolve_pipeline_config_path,
        validate_top_level_keys,
    )
    from src.server.webrtc_server import create_app  # delayed import for sys.path setup

    cfg_path = resolve_pipeline_config_path(args.pipeline_config, repo_root)
    pipeline_cfg = load_pipeline_config(cfg_path)
    validate_top_level_keys(pipeline_cfg)
    logger.info("Loaded pipeline config from %s", cfg_path)

    app = create_app(pipeline_cfg)

    # Basic production knobs (aiohttp will still handle SIGTERM/SIGINT defaults).
    app["graceful_timeout"] = args.graceful_timeout

    logger.info("Starting WebRTC server on %s:%s", args.host, args.port)
    web.run_app(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

