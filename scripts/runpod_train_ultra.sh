#!/usr/bin/env bash
#
# ARACHNE-X-ULTRA — один прогон обучения на RunPod (или любом Linux с GPU)
# и merge обученных весов в production checkpoint_dir.
#
# Перед первым запуском:
#   1) Клонировать репо, установить зависимости: pip install -r requirements.txt
#   2) Подготовить корень весов (WeightsLayout) и папку с .pt/.npz для LatentDataset
#   3) Выставить переменные ниже или передать через export перед вызовом скрипта
#
# Использование:
#   chmod +x scripts/runpod_train_ultra.sh
#   export ARACHNE_REPO_ROOT=/workspace/ARACHNE-X
#   export ARACHNE_CHECKPOINT_DIR=/workspace/weights/ULTRA_bundle
#   export ARACHNE_DATASET_DIR=/workspace/data/latents_avatar
#   export ARACHNE_TRAIN_MODE=avatar
#   export ARACHNE_MERGE_INTO=/workspace/weights/ULTRA_production
#   ./scripts/runpod_train_ultra.sh
#
# Опционально — один раз стянуть веса с Hub (не для прод-инференса):
#   export ARACHNE_ALLOW_HUB_DOWNLOAD=1
#   export HF_TOKEN=hf_...   # если приватный репо
#
set -euo pipefail

REPO_ROOT="${ARACHNE_REPO_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO_ROOT"

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

# --- обязательные переменные ---
: "${ARACHNE_CHECKPOINT_DIR:?Set ARACHNE_CHECKPOINT_DIR (full WeightsLayout root)}"
: "${ARACHNE_DATASET_DIR:?Set ARACHNE_DATASET_DIR (folder with *.pt or *.npz)}"
: "${ARACHNE_MERGE_INTO:?Set ARACHNE_MERGE_INTO (production root; final/ merges into dit/ or avatar_single/)}"

TRAIN_MODE="${ARACHNE_TRAIN_MODE:-avatar}"
OUTPUT_DIR="${ARACHNE_OUTPUT_DIR:-${REPO_ROOT}/outputs_train_runpod}"
MAX_STEPS="${ARACHNE_MAX_STEPS:-500}"
SAVE_EVERY="${ARACHNE_SAVE_EVERY:-250}"
BATCH_SIZE="${ARACHNE_BATCH_SIZE:-1}"
LR="${ARACHNE_LR:-1e-4}"
CONFIG="${ARACHNE_CONFIG:-}"

ALLOW_HUB=0
if [[ "${ARACHNE_ALLOW_HUB_DOWNLOAD:-0}" =~ ^(1|true|yes)$ ]]; then
  ALLOW_HUB=1
fi

echo "== ARACHNE-X train (ULTRA) =="
echo "  REPO_ROOT           = $REPO_ROOT"
echo "  CHECKPOINT_DIR      = $ARACHNE_CHECKPOINT_DIR"
echo "  DATASET_DIR         = $ARACHNE_DATASET_DIR"
echo "  MERGE_INTO          = $ARACHNE_MERGE_INTO"
echo "  MODE                = $TRAIN_MODE"
echo "  OUTPUT_DIR          = $OUTPUT_DIR"
echo "  MAX_STEPS           = $MAX_STEPS"
echo "  allow_hub_download  = $ALLOW_HUB"
echo

if [[ ! -d "$ARACHNE_DATASET_DIR" ]]; then
  echo "ERROR: ARACHNE_DATASET_DIR is not a directory: $ARACHNE_DATASET_DIR" >&2
  exit 2
fi
shopt -s nullglob
EXIST=( "${ARACHNE_DATASET_DIR}"/*.pt "${ARACHNE_DATASET_DIR}"/*.npz )
if [[ ${#EXIST[@]} -eq 0 ]]; then
  echo "ERROR: No .pt or .npz files in $ARACHNE_DATASET_DIR" >&2
  exit 2
fi
shopt -u nullglob

CMD=(
  python "$REPO_ROOT/scripts/arachne_x_train.py"
  --checkpoint-dir "$ARACHNE_CHECKPOINT_DIR"
  --dataset-dir "$ARACHNE_DATASET_DIR"
  --output-dir "$OUTPUT_DIR"
  --mode "$TRAIN_MODE"
  --batch-size "$BATCH_SIZE"
  --lr "$LR"
  --max-steps "$MAX_STEPS"
  --save-every "$SAVE_EVERY"
  --merge-into "$ARACHNE_MERGE_INTO"
)
if [[ -n "$CONFIG" ]]; then
  CMD+=(--config "$CONFIG")
fi
if [[ "$ALLOW_HUB" -eq 1 ]]; then
  CMD+=(--allow-hub-download)
fi
if [[ -n "${ARACHNE_WEIGHTS_CACHE_DIR:-}" ]]; then
  CMD+=(--weights-cache-dir "$ARACHNE_WEIGHTS_CACHE_DIR")
fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"

echo
echo "== Done. Next steps =="
echo "  1) Smoke infer: python scripts/infer.py --checkpoint_dir \"$ARACHNE_MERGE_INTO\" --mode ai2v ..."
echo "  2) Publish HF:  huggingface-cli upload <org>/<repo-name> \"$ARACHNE_MERGE_INTO\" . --repo-type model"
echo "     (или загрузите только diff; структура корня должна совпадать с WeightsLayout.)"
