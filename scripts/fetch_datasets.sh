#!/usr/bin/env bash
# Один прогон с корня репозитория на поде (после git clone):
#   export HF_TOKEN=...   # для tinytigerpan/tiger200k_preview после accept на HF
#   bash scripts/fetch_datasets.sh
# Доп. аргументы передаются в fetch_hf_datasets.py, например:
#   bash scripts/fetch_datasets.sh --tiger
#   bash scripts/fetch_datasets.sh --hdvila --hdvila-max-rows 500000
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
python3 -m pip install -r requirements-datasets.txt -q
if [ "$#" -eq 0 ]; then
  exec python3 scripts/fetch_hf_datasets.py --all
else
  exec python3 scripts/fetch_hf_datasets.py "$@"
fi
