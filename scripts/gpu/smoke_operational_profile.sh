#!/usr/bin/env bash
# Manual H200 smoke: operational vs cinematic (document-only gate before worker default).
set -euo pipefail

CKPT="${NULLXES_CHECKPOINT_DIR:?set NULLXES_CHECKPOINT_DIR}"
IMAGE="${SMOKE_IMAGE:?set SMOKE_IMAGE}"
AUDIO="${SMOKE_AUDIO:?set SMOKE_AUDIO}"
OUT_DIR="${SMOKE_OUT_DIR:-/tmp/arachne_smoke}"

mkdir -p "$OUT_DIR"

export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

echo "[smoke] operational profile"
python scripts/infer.py \
  --checkpoint_dir "$CKPT" \
  --mode ai2v \
  --runtime_profile operational \
  --image "$IMAGE" \
  --audio "$AUDIO" \
  --prompt "speaking clearly to camera" \
  --output "$OUT_DIR/operational.mp4"

echo "[smoke] cinematic baseline"
python scripts/infer.py \
  --checkpoint_dir "$CKPT" \
  --mode ai2v \
  --runtime_profile cinematic \
  --image "$IMAGE" \
  --audio "$AUDIO" \
  --prompt "speaking clearly to camera" \
  --output "$OUT_DIR/cinematic.mp4"

echo "[smoke] compare .run.json sidecars and lipsync/identity visually"
