#!/usr/bin/env bash
# Elena HR LoRA train on RunPod (foreground, unbuffered logs).
# Usage:
#   bash scripts/gpu/train_elenahr_lora.sh          # fresh 0→60
#   bash scripts/gpu/train_elenahr_lora.sh resume   # from lora_step_15 → 60
set -euo pipefail

ARACHNE_ROOT="${ARACHNE_ROOT:-/workspace/ARACHNE-X}"
cd "$ARACHNE_ROOT"
source .venv/bin/activate
export PYTHONPATH="${ARACHNE_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

CKPT="${NULLXES_CHECKPOINT_DIR:-$ARACHNE_ROOT/weights/arachne-avatar-runtime}"
DATASET_DIR="${ELENAHR_DATASET_DIR:-training_latents/elenahr_v1}"
OUT_DIR="${ELENAHR_OUTPUT_DIR:-output/elenahr_lora_v1}"
MODE="${1:-fresh}"

if [[ ! -d "$CKPT/avatar_single" && ! -d "$CKPT/tokenizer" ]]; then
  echo "ERROR: checkpoint missing: $CKPT" >&2
  exit 1
fi
if ! ls "$DATASET_DIR"/*.pt >/dev/null 2>&1; then
  echo "ERROR: no .pt in $DATASET_DIR — run export_elena_lora_smoke.sh first" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"
LOG="${OUT_DIR}/train.log"

pkill -f "train_lora_avatar.py" 2>/dev/null || true
sleep 1
python -c "import gc,torch; gc.collect(); torch.cuda.empty_cache()"

echo "=== elenahr LoRA train mode=$MODE ===" | tee "$LOG"
echo "checkpoint=$CKPT dataset=$DATASET_DIR out=$OUT_DIR" | tee -a "$LOG"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader 2>/dev/null | tee -a "$LOG" || true

RESUME_ARGS=()
if [[ "$MODE" == "resume" ]]; then
  RESUME_CKPT="${ELENAHR_RESUME:-$OUT_DIR/lora_step_15.safetensors}"
  START_STEP="${ELENAHR_START_STEP:-16}"
  if [[ ! -f "$RESUME_CKPT" ]]; then
    echo "ERROR: resume checkpoint not found: $RESUME_CKPT" >&2
    ls -la "$OUT_DIR"/*.safetensors 2>/dev/null || true
    exit 1
  fi
  RESUME_ARGS=(--resume_lora_path "$RESUME_CKPT" --start_step "$START_STEP")
  echo "resume=$RESUME_CKPT start_step=$START_STEP" | tee -a "$LOG"
elif [[ "$MODE" != "fresh" ]]; then
  echo "Usage: $0 [fresh|resume]" >&2
  exit 1
fi

# After shard load: DiT→GPU + LoRA scan can take 2–5 min with no tqdm — watch [train_lora_avatar] lines.
exec python -u scripts/train_lora_avatar.py \
  --checkpoint_dir "$CKPT" \
  --dataset_dir "$DATASET_DIR" \
  --output_dir "$OUT_DIR" \
  --lora_key elenahr \
  --lora_rank 16 --lora_alpha 8 \
  --max_steps 60 --save_every 15 \
  --lr 5e-5 --batch_size 1 \
  --num_workers 0 \
  "${RESUME_ARGS[@]}" \
  2>&1 | tee -a "$LOG"
