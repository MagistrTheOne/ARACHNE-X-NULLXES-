#!/usr/bin/env bash
# Elena clone WAV -> audio_i2v (kills stale infer.py on pod, then runs).
set -euo pipefail

ARACHNE_ROOT="${ARACHNE_ROOT:-/workspace/ARACHNE-X}"
VIDEO_CKPT="${VIDEO_CKPT:-$ARACHNE_ROOT/weights/ARACHNE-X-ULTRA-VIDEO}"
AVATAR_CKPT="${AVATAR_CKPT:-$ARACHNE_ROOT/weights/ARACHNE-X-ULTRA-AVATAR}"
IMAGE="${IMAGE:-assets/avatar/single/elena/elena/image/1.png}"
AUDIO="${AUDIO:-output/elena_clone_tts_test.wav}"
OUTPUT="${OUTPUT:-output/imagine_elena_clone.mp4}"

cd "$ARACHNE_ROOT"
source .venv/bin/activate

echo "=== stop stale infer.py on GPU ==="
pkill -f "python scripts/infer.py" 2>/dev/null || true
sleep 2
nvidia-smi || true

export ARACHNE_ROOT
export VIDEO_CKPT
export AVATAR_CKPT
export ARACHNE_AVATAR_CKPT="$AVATAR_CKPT"
export ARACHNE_GEMMA_MODEL="${ARACHNE_GEMMA_MODEL:-$ARACHNE_ROOT/weights/hf/gemma-2-2b-it}"
export PYTHONPATH="$ARACHNE_ROOT"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export ARACHNE_SEQUENTIAL_CFG="${ARACHNE_SEQUENTIAL_CFG:-1}"
mkdir -p output

if [[ ! -f "$AUDIO" ]]; then
  echo "missing audio: $AUDIO (run Elena Qwen clone TTS first)" >&2
  exit 1
fi

python scripts/infer.py \
  --checkpoint_dir "$VIDEO_CKPT" \
  --mode audio_i2v \
  --image "$IMAGE" \
  --audio "$AUDIO" \
  --prompt "Elena speaking naturally to camera, stable identity, HR interview, calm professional tone" \
  --prompt_compiler gemma \
  --resolution 480p \
  --num_frames 49 \
  --num_inference_steps 25 \
  --text_guidance_scale 4.0 \
  --audio_conditioning_scale 1.0 \
  --output "$OUTPUT"

ls -lh "$OUTPUT" "${OUTPUT%.mp4}.run.json" 2>/dev/null || ls -lh "$OUTPUT"
