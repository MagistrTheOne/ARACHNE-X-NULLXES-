#!/usr/bin/env bash
# A/B smokes for experimental audio_i2v (lab only).
set -euo pipefail

ARACHNE_ROOT="${ARACHNE_ROOT:-/workspace/ARACHNE-X}"
VIDEO_CKPT="${VIDEO_CKPT:-$ARACHNE_ROOT/weights/ARACHNE-X-ULTRA-VIDEO}"
IMAGE="${IMAGE:-assets/avatar/single/elena/image.jpg}"
AUDIO="${AUDIO:-assets/avatar/single/elena/audio.wav}"
ADAPTER="${ADAPTER:-output/audio_i2v_adapter.safetensors}"

cd "$ARACHNE_ROOT"
source .venv/bin/activate
export PYTHONPATH="$ARACHNE_ROOT"
mkdir -p output

echo "=== A: base i2v ==="
python scripts/infer.py \
  --checkpoint_dir "$VIDEO_CKPT" \
  --mode i2v \
  --image "$IMAGE" \
  --audio "$AUDIO" \
  --prompt "Professional portrait, subtle motion, stable identity." \
  --negative_prompt "blurry, low quality, watermark" \
  --resolution 480p \
  --num_frames 49 \
  --num_inference_steps 25 \
  --text_guidance_scale 4.0 \
  --output output/smoke_i2v_base.mp4

echo "=== B: audio_i2v scale=0 (must match base path) ==="
python scripts/infer.py \
  --checkpoint_dir "$VIDEO_CKPT" \
  --mode audio_i2v \
  --image "$IMAGE" \
  --audio "$AUDIO" \
  --prompt "Professional portrait, subtle motion, stable identity." \
  --negative_prompt "blurry, low quality, watermark" \
  --resolution 480p \
  --num_frames 49 \
  --num_inference_steps 25 \
  --text_guidance_scale 4.0 \
  --audio_conditioning_scale 0.0 \
  --preset_hint audio_i2v_scale0 \
  --output output/smoke_audio_i2v_scale0.mp4

echo "=== C: audio_i2v scale=1 ==="
python scripts/infer.py \
  --checkpoint_dir "$VIDEO_CKPT" \
  --mode audio_i2v \
  --image "$IMAGE" \
  --audio "$AUDIO" \
  --prompt "Professional portrait speaking naturally, stable identity, subtle head motion." \
  --negative_prompt "blurry, low quality, watermark" \
  --resolution 480p \
  --num_frames 49 \
  --num_inference_steps 25 \
  --text_guidance_scale 4.0 \
  --audio_conditioning_scale 1.0 \
  ${ADAPTER:+--audio_conditioning_adapter "$ADAPTER"} \
  --preset_hint audio_i2v_scale1 \
  --output output/smoke_audio_i2v_scale1.mp4

echo "=== D: long-window stress (97 frames) ==="
python scripts/infer.py \
  --checkpoint_dir "$VIDEO_CKPT" \
  --mode audio_i2v \
  --image "$IMAGE" \
  --audio "$AUDIO" \
  --prompt "Professional portrait speaking naturally, stable identity." \
  --negative_prompt "blurry, low quality, watermark" \
  --resolution 480p \
  --num_frames 97 \
  --num_inference_steps 25 \
  --text_guidance_scale 4.0 \
  --audio_conditioning_scale 1.0 \
  ${ADAPTER:+--audio_conditioning_adapter "$ADAPTER"} \
  --preset_hint audio_i2v_long97 \
  --output output/smoke_audio_i2v_long97.mp4

ls -lh output/smoke_*.mp4 output/smoke_*.run.json 2>/dev/null || true
echo "audio_i2v smoke complete"
