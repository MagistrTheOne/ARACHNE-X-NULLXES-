#!/usr/bin/env bash
# Svetlana: enroll fresh identity bank (svetlanaV2) + ai2v digitization.
# Broken/corrupt Svetlana.pt (zip error) — do not use; re-enroll from sveta.jpg.
set -euo pipefail

ARACHNE_ROOT="${ARACHNE_ROOT:-/workspace/ARACHNE-X}"
NULLXES_CHECKPOINT_DIR="${NULLXES_CHECKPOINT_DIR:-$ARACHNE_ROOT/weights/arachne-avatar-runtime}"
IMAGE="${IMAGE:-assets/avatar/single/svetlana/sveta.jpg}"
AUDIO="${AUDIO:-assets/avatar/single/svetlana/audio.wav}"
IDENTITY_ID="${IDENTITY_ID:-1}"
BANK_OUT="${BANK_OUT:-output/svetlanaV2_identity_bank.pt}"
OUTPUT="${OUTPUT:-output/svetlana_ai2v.mp4}"
ENROLL_ONLY="${ENROLL_ONLY:-0}"
SKIP_ENROLL="${SKIP_ENROLL:-0}"

cd "$ARACHNE_ROOT"
source .venv/bin/activate

pkill -f "python scripts/infer.py" 2>/dev/null || true
sleep 2

export ARACHNE_ROOT NULLXES_CHECKPOINT_DIR
export ARACHNE_AVATAR_CKPT="${ARACHNE_AVATAR_CKPT:-$ARACHNE_ROOT/weights/ARACHNE-X-ULTRA-AVATAR}"
export PYTHONPATH="$ARACHNE_ROOT"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p output

verify_bank() {
  python - <<PY
import sys, torch
path = "$1"
try:
    p = torch.load(path, map_location="cpu")
except Exception as e:
    print("INVALID", path, e)
    sys.exit(1)
need = {"version", "identity_embedding", "identity_bank_size"}
missing = need - set(p.keys())
if missing:
    print("INVALID missing keys", missing)
    sys.exit(1)
print("OK", path, "shape", tuple(p["identity_embedding"].shape))
PY
}

if [[ "$SKIP_ENROLL" != "1" ]]; then
  echo "=== enroll_identity -> $BANK_OUT (identity_id=$IDENTITY_ID) ==="
  python scripts/infer.py \
    --checkpoint_dir "$NULLXES_CHECKPOINT_DIR" \
    --mode enroll_identity \
    --image "$IMAGE" \
    --identity_id "$IDENTITY_ID" \
    --identity_bank_save_path "$BANK_OUT" \
    --resolution 720p
  verify_bank "$BANK_OUT"
fi

if [[ "$ENROLL_ONLY" == "1" ]]; then
  echo "enroll_only done: $BANK_OUT"
  exit 0
fi

if [[ ! -f "$BANK_OUT" ]]; then
  echo "missing bank $BANK_OUT; run with SKIP_ENROLL=0" >&2
  exit 1
fi
verify_bank "$BANK_OUT"

PROMPT="SVETLANA, ultra realistic professional woman, speaking naturally straight to camera, stable identity, precise audio-driven lipsync, cinematic portrait lighting, photorealistic skin, minimal head movement, high temporal consistency"
NEG="anime, cartoon, blurry, low quality, distorted face, duplicated mouth, frozen lips, bad anatomy, warped eyes, lowres, deformed face, flicker, watermark, text, jitter"

echo "=== ai2v + identity bank ==="
python scripts/infer.py \
  --checkpoint_dir "$NULLXES_CHECKPOINT_DIR" \
  --mode ai2v \
  --image "$IMAGE" \
  --audio "$AUDIO" \
  --prompt "$PROMPT" \
  --negative_prompt "$NEG" \
  --identity_bank_path "$BANK_OUT" \
  --identity_id "$IDENTITY_ID" \
  --identity_strength 1.0 \
  --resolution 720p \
  --num_frames_mode sync \
  --num_inference_steps 35 \
  --text_guidance_scale 4.0 \
  --audio_guidance_scale 5.5 \
  --output "$OUTPUT"

ls -lh "$OUTPUT" "$BANK_OUT" "${OUTPUT%.mp4}.run.json" 2>/dev/null || ls -lh "$OUTPUT" "$BANK_OUT"
