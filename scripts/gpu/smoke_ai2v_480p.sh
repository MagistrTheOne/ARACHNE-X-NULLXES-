#!/usr/bin/env bash
# H200 smoke: operational ai2v @ 480p with optional identity bank (chunked path).
# Validates MP4 mux + .run.json sampling_metrics after normalize_ai2v_video_output fix.
set -euo pipefail

ROOT="${ARACHNE_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$ROOT"

CKPT="${NULLXES_CHECKPOINT_DIR:?export NULLXES_CHECKPOINT_DIR}"
IMAGE="${SMOKE_IMAGE:-assets/avatar/single/katya/main.jpg}"
AUDIO="${SMOKE_AUDIO:-assets/avatar/single/katya/audio.wav}"
IDENTITY_BANK="${SMOKE_IDENTITY_BANK:-output/katya_identity_bank.pt}"
OUT_DIR="${SMOKE_OUT_DIR:-output/smoke_480p}"
RESOLUTION="${SMOKE_RESOLUTION:-480p}"
MP4="$OUT_DIR/smoke_operational_480p.mp4"

mkdir -p "$OUT_DIR"
source "${ROOT}/.venv/bin/activate"
export PYTHONPATH="$ROOT"
export NULLXES_CHECKPOINT_DIR="$CKPT"

if [[ ! -f "$IMAGE" ]]; then
  echo "SMOKE_IMAGE missing: $IMAGE" >&2
  exit 1
fi
if [[ ! -f "$AUDIO" ]]; then
  echo "SMOKE_AUDIO missing: $AUDIO" >&2
  exit 1
fi

EXTRA=()
if [[ -f "$IDENTITY_BANK" ]]; then
  EXTRA+=(--identity_bank_path "$IDENTITY_BANK" --identity_id 1)
fi

echo "[smoke-480p] operational chunked ai2v resolution=$RESOLUTION"
python scripts/infer.py \
  --checkpoint_dir "$CKPT" \
  --mode ai2v \
  --runtime_profile operational \
  --resolution "$RESOLUTION" \
  --image "$IMAGE" \
  --audio "$AUDIO" \
  --prompt "speaking clearly to camera, stable identity, precise lipsync" \
  --negative_prompt "anime, cartoon, blurry, distorted face, watermark" \
  "${EXTRA[@]}" \
  --output "$MP4"

python - <<'PY'
import json
import subprocess
import sys
from pathlib import Path

mp4 = Path(sys.argv[1])
run_json = mp4.with_suffix(".run.json")
if not mp4.is_file():
    raise SystemExit(f"missing mp4: {mp4}")
if mp4.stat().st_size < 1024:
    raise SystemExit(f"mp4 too small: {mp4} ({mp4.stat().st_size} bytes)")

probe = subprocess.run(
    ["ffprobe", "-v", "error", "-select_streams", "v:0",
     "-show_entries", "stream=width,height,nb_frames,duration",
     "-of", "json", str(mp4)],
    capture_output=True,
    text=True,
    check=True,
)
meta = json.loads(probe.stdout)
stream = (meta.get("streams") or [{}])[0]
w = int(stream.get("width") or 0)
h = int(stream.get("height") or 0)
if w < 400 or h < 400:
    raise SystemExit(f"unexpected resolution: {w}x{h}")

if run_json.is_file():
    payload = json.loads(run_json.read_text(encoding="utf-8"))
    metrics = payload.get("sampling_metrics") or {}
    chunks = metrics.get("chunk_count")
    print(f"[smoke-480p] OK mp4={mp4.name} {w}x{h} chunk_count={chunks}")
else:
    print(f"[smoke-480p] OK mp4={mp4.name} {w}x{h} (no .run.json)")
PY
"$MP4"

echo "[smoke-480p] PASS"
