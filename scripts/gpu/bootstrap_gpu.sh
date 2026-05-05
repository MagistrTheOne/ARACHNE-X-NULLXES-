#!/usr/bin/env bash
# Optional Linux/RunPod GPU node prep: venv, common pins, PYTHONPATH reminder.
# Run from repo root or set ARACHNE_ROOT to this repository path.
set -euo pipefail

ARACHNE_ROOT="${ARACHNE_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$ARACHNE_ROOT"

PY="${PYTHON:-python3}"
if ! command -v "$PY" >/dev/null 2>&1; then
  echo "python3 not found; install Python 3.10+ first." >&2
  exit 1
fi

VENV="${GPU_VENV:-${ARACHNE_ROOT}/.venv-gpu}"
"$PY" -m venv "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install -U pip wheel

# Pin avoids transformers breakage when huggingface_hub 1.x is pulled.
python -m pip install "huggingface_hub>=0.34,<1.0"

echo "venv ready: $VENV"
echo "Activate: source \"$VENV/bin/activate\""
echo "Optional: export PYTHONPATH=\"\$PWD/services/longcat-worker:\$PWD\" (adjust for Wan/LTX trees)"
