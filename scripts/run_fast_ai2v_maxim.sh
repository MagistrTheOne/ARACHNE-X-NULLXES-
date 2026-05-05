#!/usr/bin/env bash
set -euo pipefail

cd /workspace/ARACHNE-X
source .venv/bin/activate
export PYTHONPATH=/workspace/ARACHNE-X

python scripts/infer.py \
  --checkpoint_dir /workspace/weights/ARACHNE-X-Avatar \
  --mode ai2v \
  --image /workspace/ARACHNE-X/assets/avatar/single/MaximOnyushko/image.png \
  --audio /workspace/ARACHNE-X/assets/avatar/single/MaximOnyushko/voice.wav \
  --prompt "A realistic close-up of a young man speaking directly to camera, natural facial expression, precise lip movements synchronized with speech, subtle head motion, stable identity, soft cinematic lighting, high facial detail" \
  --resolution 480p \
  --num_frames 93 \
  --num_inference_steps 8 \
  --text_guidance_scale 4.0 \
  --audio_guidance_scale 4.0 \
  --output /workspace/out_maxim_ai2v.mp4
