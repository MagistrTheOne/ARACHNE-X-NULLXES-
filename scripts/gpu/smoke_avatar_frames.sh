#!/usr/bin/env bash
# NDJSON smoke test for POST /v1/realtime/avatar_frames (longcat-worker or compatible).
# Usage: NULLXES_URL=http://127.0.0.1:8080 scripts/gpu/smoke_avatar_frames.sh
# Optional: X_NULLXES_KEY=... if worker enforces LONGCAT_INFERENCE_SERVICE_KEY
set -euo pipefail

BASE="${NULLXES_URL:-http://127.0.0.1:8080}"
BASE="${BASE%/}"
PATH_FRAMES="${NULLXES_AVATAR_FRAMES_PATH:-/v1/realtime/avatar_frames}"
KEY_HEADER="${NULLXES_KEY_HEADER:-X-NULLXES-Avatar-Inference-Key}"

PAYLOAD=$(python3 - <<'PY'
import array
import base64
import json
import os
import random

# 1x1 transparent PNG
png = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)
img_b64 = base64.b64encode(png).decode("ascii")
random.seed(0)
audio = array.array("f", [random.random() * 0.01 for _ in range(4000)])
aud_b64 = base64.b64encode(audio.tobytes()).decode("ascii")

body = {
    "sessionId": "smoke_session",
    "prompt": "smoke test",
    "imageBase64": img_b64,
    "audioFloat32Base64": aud_b64,
    "resolution": "480p",
    "numFrames": 4,
    "numInferenceSteps": 1,
    "engine": os.environ.get("NULLXES_SMOKE_ENGINE", "longcat"),
}
print(json.dumps(body))
PY
)

CURL_ARGS=(-sS -X POST "$BASE$PATH_FRAMES" -H "Content-Type: application/json" -d "$PAYLOAD")
if [[ -n "${X_NULLXES_KEY:-}" ]]; then
  CURL_ARGS+=(-H "$KEY_HEADER: $X_NULLXES_KEY")
fi

echo "POST $BASE$PATH_FRAMES"
if ! out=$(curl "${CURL_ARGS[@]}"); then
  echo "curl failed" >&2
  exit 1
fi

if echo "$out" | head -1 | grep -q '"error"'; then
  echo "$out" | head -c 2000
  echo
  exit 1
fi

echo "ok (first lines):"
echo "$out" | head -3
