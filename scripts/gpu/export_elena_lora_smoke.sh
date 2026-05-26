#!/usr/bin/env bash
# Export Elena LoRA training latents (5 pairs). Run on RunPod H200 with .venv + NULLXES_CHECKPOINT_DIR.
set -euo pipefail

ARACHNE_ROOT="${ARACHNE_ROOT:-/workspace/ARACHNE-X}"
cd "$ARACHNE_ROOT"
source .venv/bin/activate
export PYTHONPATH="${ARACHNE_ROOT}${PYTHONPATH:+:$PYTHONPATH}"

CKPT="${NULLXES_CHECKPOINT_DIR:-$ARACHNE_ROOT/weights/arachne-avatar-runtime}"
if [[ ! -d "$CKPT/tokenizer" && ! -d "$CKPT/avatar_single" ]]; then
  echo "ERROR: checkpoint not found: $CKPT" >&2
  echo "Set: export ARACHNE_ROOT=$ARACHNE_ROOT" >&2
  echo "     export NULLXES_CHECKPOINT_DIR=\$ARACHNE_ROOT/weights/arachne-avatar-runtime" >&2
  echo "Merged runtime must exist (RUNPOD §2.4–2.5)." >&2
  exit 1
fi
PRESET_JSON="assets/avatar/single/elena/elena.json"
OUT_DIR="${1:-training_latents/elenahr_v1}"
RESOLUTION="${LORA_EXPORT_RESOLUTION:-720p}"

NEG="$(python3 -c "import json; print(json.load(open('$PRESET_JSON'))['negative_prompt'])")"
PREFIX="ELENA, audio-driven lipsync, mouth and jaw follow speech, fixed camera no zoom no dolly, head locked minimal movement, high temporal consistency, natural pace not fast motion, "

mkdir -p "$OUT_DIR"

export_pair() {
  local id="$1" image="$2" audio="$3" prompt_file="$4" num_frames="$5"
  local body="${PREFIX}$(tr -d '\r' < "$prompt_file" | tr '\n' ' ')"
  python scripts/export_latent_training_sample.py \
    --checkpoint_dir "$CKPT" \
    --image "$image" \
    --audio "$audio" \
    --prompt "$body" \
    --negative_prompt "$NEG" \
    --resolution "$RESOLUTION" \
    --num_frames "$num_frames" \
    --output "${OUT_DIR}/elena_$(printf '%03d' "$id").pt" \
    --seed "$((1000 + id))"
}

export_pair 1 assets/avatar/single/elena/elena/image/1.png assets/avatar/single/elena/elena/audio/elena1.wav assets/avatar/single/elena/elena/prompt/1.txt 69
export_pair 2 assets/avatar/single/elena/elena/image/2.png assets/avatar/single/elena/elena/audio/elena2.wav assets/avatar/single/elena/elena/prompt/2.txt 37
export_pair 3 assets/avatar/single/elena/elena/image/3.png assets/avatar/single/elena/elena/audio/elena3.wav assets/avatar/single/elena/elena/prompt/3.txt 41
export_pair 4 assets/avatar/single/elena/elena/image/4.png assets/avatar/single/elena/elena/audio/elena4.wav assets/avatar/single/elena/elena/prompt/4.txt 45
export_pair 5 assets/avatar/single/elena/elena/image/5.png assets/avatar/single/elena/elena/audio/elena5.wav assets/avatar/single/elena/elena/prompt/5.txt 53

echo "Done: $(ls -1 "$OUT_DIR"/*.pt | wc -l) files in $OUT_DIR"
