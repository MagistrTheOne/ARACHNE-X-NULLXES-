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
PID_FILE="${OUT_DIR}/train.pid"

stop_elenahr_gpu_jobs() {
  echo "=== stop: prior ARACHNE train / latent-export on GPU ==="

  if [[ -f "$PID_FILE" ]]; then
    old_pid="$(cat "$PID_FILE" 2>/dev/null || true)"
    if [[ -n "${old_pid:-}" ]] && kill -0 "$old_pid" 2>/dev/null; then
      echo "kill PID file $PID_FILE → pid=$old_pid (SIGTERM)"
      kill -TERM "$old_pid" 2>/dev/null || true
      sleep 2
      kill -KILL "$old_pid" 2>/dev/null || true
    fi
    rm -f "$PID_FILE"
  fi

  local patterns=(
    "train_lora_avatar.py"
    "scripts/train_lora_avatar.py"
    "export_latent_training_sample.py"
  )
  for pat in "${patterns[@]}"; do
    if pgrep -f "$pat" >/dev/null 2>&1; then
      echo "pkill -TERM -f $pat"
      pkill -TERM -f "$pat" 2>/dev/null || true
    fi
  done
  sleep 3

  for pat in "${patterns[@]}"; do
    if pgrep -f "$pat" >/dev/null 2>&1; then
      echo "pkill -KILL -f $pat"
      pkill -KILL -f "$pat" 2>/dev/null || true
    fi
  done
  sleep 1

  if pgrep -af "train_lora_avatar|export_latent_training_sample" >/dev/null 2>&1; then
    echo "WARN: still running:" >&2
    pgrep -af "train_lora_avatar|export_latent_training_sample" >&2 || true
    echo "ERROR: could not stop old GPU jobs — free VRAM manually (nvidia-smi) and retry" >&2
    exit 1
  fi
  echo "=== stop: no train/export processes ==="

  python -c "import gc,torch; gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None"
  nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null \
    || echo "(nvidia-smi unavailable)"
}

stop_elenahr_gpu_jobs

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
  LOG="${OUT_DIR}/train_resume.log"
elif [[ "$MODE" != "fresh" ]]; then
  echo "Usage: $0 [fresh|resume]" >&2
  exit 1
fi

{
  echo ""
  echo "=== elenahr LoRA train mode=$MODE $(date -Iseconds) ==="
  echo "checkpoint=$CKPT dataset=$DATASET_DIR out=$OUT_DIR"
  if [[ "${#RESUME_ARGS[@]}" -gt 0 ]]; then
    echo "resume=$RESUME_CKPT start_step=$START_STEP"
  fi
} | tee -a "$LOG"

# Anti-snow (flow-match Min-SNR + audio RMS). Resume must keep --lora_scope default if checkpoint used default.
LORA_SCOPE="${ELENAHR_LORA_SCOPE:-default}"
MIN_SNR="${ELENAHR_MIN_SNR_GAMMA:-5}"
EMA="${ELENAHR_EMA_DECAY:-0.999}"

# After shard load: DiT→GPU + LoRA scan ~2–5 min — wait for [train_lora_avatar] lines.
python -u scripts/train_lora_avatar.py \
  --checkpoint_dir "$CKPT" \
  --dataset_dir "$DATASET_DIR" \
  --output_dir "$OUT_DIR" \
  --lora_key elenahr \
  --lora_scope "$LORA_SCOPE" \
  --lora_rank 16 --lora_alpha 8 \
  --max_steps 60 --save_every 15 \
  --lr 5e-5 --batch_size 1 \
  --num_workers 0 \
  --min_snr_gamma "$MIN_SNR" \
  --normalize_audio_embs \
  --ema_decay "$EMA" \
  "${RESUME_ARGS[@]}" \
  2>&1 | tee -a "$LOG" &
train_pid=$!
echo "$train_pid" > "$PID_FILE"
echo "train pid=$train_pid (PID file: $PID_FILE)"
trap 'echo "interrupt → stopping train pid=$train_pid"; kill -TERM "$train_pid" 2>/dev/null; wait "$train_pid" 2>/dev/null; rm -f "$PID_FILE"; exit 130' INT TERM
wait "$train_pid"
trap - INT TERM
rm -f "$PID_FILE"
echo "=== train finished $(date -Iseconds) ===" | tee -a "$LOG"
ls -la "$OUT_DIR"/*.safetensors 2>/dev/null | tee -a "$LOG" || true
