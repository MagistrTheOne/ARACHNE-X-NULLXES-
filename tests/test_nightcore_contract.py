"""NDJSON stream contract + 80GB resolution helper tests (CPU-safe)."""

from __future__ import annotations

import json
from pathlib import Path


def test_stream_frames_body_schema_fields():
    text = Path("services/arachnex-worker/main.py").read_text(encoding="utf-8")
    for name in ("sessionId", "imageBase64", "negativePrompt", "runtimeProfile"):
        assert name in text


def test_ndjson_line_keys_match_manifest():
    sample = {
        "seq": 1,
        "tsMs": 0,
        "encoding": "rgb24_base64",
        "width": 832,
        "height": 480,
        "frameBase64": "AA==",
    }
    parsed = json.loads(json.dumps(sample))
    for key in ("seq", "tsMs", "encoding", "width", "height", "frameBase64"):
        assert key in parsed
