#!/usr/bin/env bash
# RunPod A/B: prompt compiler off vs openai vs gemma (Elena pair 5, ~6s audio).
set -euo pipefail

CKPT="${1:?usage: ab_prompt_compiler_elena.sh CHECKPOINT_DIR}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

IMAGE="${ELENA_IMAGE:-assets/avatar/single/elena/elena/prompt/5.png}"
AUDIO="${ELENA_AUDIO:-assets/avatar/single/elena/elena6_6s.wav}"
PROMPT_FILE="${ELENA_PROMPT:-assets/avatar/single/elena/elena/prompt/5.txt}"
PROMPT="$(cat "$PROMPT_FILE")"

COMMON=(
  --checkpoint_dir "$CKPT"
  --mode ai2v
  --image "$IMAGE"
  --audio "$AUDIO"
  --prompt "$PROMPT"
  --resolution 720p
  --num_frames_mode duration
  --embedding_fps_auto
  --num_inference_steps 35
  --text_guidance_scale 4.0
  --audio_guidance_scale 5.5
  --lora_path "${ELENA_LORA:-}"
)

run_one() {
  local backend="$1"
  local out="$2"
  echo "=== prompt_compiler=$backend -> $out ==="
  python scripts/infer.py \
    "${COMMON[@]}" \
    --prompt_compiler "$backend" \
    --prompt_compiler_fallback openai \
    --output "$out"
}

run_one off "elena_ab_compiler_off.mp4"
run_one openai "elena_ab_compiler_openai.mp4"
if [[ -n "${SKIP_GEMMA_AB:-}" ]]; then
  echo "SKIP_GEMMA_AB set; skipping gemma"
else
  run_one gemma "elena_ab_compiler_gemma.mp4"
fi

echo "Done. Compare elena_ab_compiler_*.mp4"
