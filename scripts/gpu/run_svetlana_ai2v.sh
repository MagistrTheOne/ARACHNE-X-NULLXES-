#!/usr/bin/env bash
# Svetlana digitization: image + audio -> ai2v MP4 (avatar runtime, not VIDEO audio_i2v).
set -euo pipefail

ARACHNE_ROOT="${ARACHNE_ROOT:-/workspace/ARACHNE-X}"
NULLXES_CHECKPOINT_DIR="${NULLXES_CHECKPOINT_DIR:-$ARACHNE_ROOT/weights/arachne-avatar-runtime}"
PRESET="${PRESET:-assets/avatar/single/svetlana/svetlana.json}"
OUTPUT="${OUTPUT:-output/svetlana_ai2v.mp4}"
IDENTITY_ID="${IDENTITY_ID:-0}"
USE_IDENTITY_BANK="${USE_IDENTITY_BANK:-1}"
SMOKE="${SMOKE:-0}"

cd "$ARACHNE_ROOT"
source .venv/bin/activate

echo "=== stop stale GPU jobs ==="
pkill -f "python scripts/infer.py" 2>/dev/null || true
sleep 2
python - <<'PY' || true
import gc, torch
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    free, total = torch.cuda.mem_get_info()
    print(f"gpu_after_cleanup free_gb={free/1024**3:.2f} total_gb={total/1024**3:.2f}")
PY
nvidia-smi || true

export ARACHNE_ROOT
export NULLXES_CHECKPOINT_DIR
export ARACHNE_AVATAR_CKPT="${ARACHNE_AVATAR_CKPT:-$ARACHNE_ROOT/weights/ARACHNE-X-ULTRA-AVATAR}"
export PYTHONPATH="$ARACHNE_ROOT"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [[ ! -d "$NULLXES_CHECKPOINT_DIR/avatar_single" ]]; then
  echo "missing merged runtime: $NULLXES_CHECKPOINT_DIR" >&2
  echo "build per RUNPOD_H200_AVATAR_SETUP.md §2.4" >&2
  exit 1
fi

IMAGE="$(python -c "import json; print(json.load(open('$PRESET'))['cond_image'])")"
AUDIO="$(python -c "import json; print(json.load(open('$PRESET'))['cond_audio'])")"
PROMPT="$(python -c "import json; print(json.load(open('$PRESET'))['prompt'])")"
NEG="$(python -c "import json; print(json.load(open('$PRESET'))['negative_prompt'])")"
BANK="$(python -c "import json; print(json.load(open('$PRESET'))['_arachne_x_infer'].get('identity_bank_path',''))")"

for f in "$IMAGE" "$AUDIO"; do
  if [[ ! -f "$f" ]]; then
    echo "missing file: $f" >&2
    exit 1
  fi
done

IDENTITY_ARGS=()
if [[ "$USE_IDENTITY_BANK" == "1" && -n "$BANK" && -f "$BANK" ]]; then
  if python - <<PY
import torch
torch.load("$BANK", map_location="cpu")
PY
  then
    IDENTITY_ARGS=(--identity_bank_path "$BANK" --identity_id "$IDENTITY_ID" --identity_strength 1.0)
    echo "identity_bank OK: $BANK"
  else
    echo "WARN: identity bank unreadable, continuing without bank: $BANK" >&2
  fi
else
  echo "identity_bank skipped (USE_IDENTITY_BANK=$USE_IDENTITY_BANK)"
fi

EXTRA=()
if [[ "$SMOKE" == "1" ]]; then
  EXTRA+=(--resolution 480p --num_frames 17 --num_inference_steps 2 --text_guidance_scale 3.0 --audio_guidance_scale 3.0)
else
  EXTRA+=(--resolution 720p --num_frames_mode sync --num_inference_steps 35 --text_guidance_scale 4.0 --audio_guidance_scale 5.5)
fi

python scripts/infer.py \
  --checkpoint_dir "$NULLXES_CHECKPOINT_DIR" \
  --mode ai2v \
  --image "$IMAGE" \
  --audio "$AUDIO" \
  --prompt "$PROMPT" \
  --negative_prompt "$NEG" \
  "${IDENTITY_ARGS[@]}" \
  "${EXTRA[@]}" \
  --output "$OUTPUT"

ls -lh "$OUTPUT" "${OUTPUT%.mp4}.run.json" 2>/dev/null || ls -lh "$OUTPUT"
