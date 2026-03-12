#!/usr/bin/env bash
set -euo pipefail

cd /workspace/ARACHNE-X
source .venv/bin/activate
export PYTHONPATH=/workspace/ARACHNE-X

python scripts/infer.py \
  --checkpoint_dir /workspace/weights/ARACHNE-X \
  --mode t2v \
  --prompt "A cinematic close-up of a beautiful young woman, 24+ years old, with long white hair, wearing a black NULLXES suit, futuristic luxury fashion aesthetic, soft studio lighting, highly detailed face, natural eye movement, subtle head motion, elegant clean background, realistic skin texture, premium high-end look" \
  --negative_prompt "low quality, blurry, deformed face, bad anatomy, flicker, artifacts, watermark, text" \
  --height 480 \
  --width 832 \
  --num_frames 93 \
  --num_inference_steps 8 \
  --text_guidance_scale 4.0 \
  --output /workspace/out_nullxes_t2v.mp4
