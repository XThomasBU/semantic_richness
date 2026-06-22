#!/bin/bash -l
# Bootstrap CIs for all models with rotation results (PACS + Omniglot categories).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

module load miniconda 2>/dev/null || true
conda activate blind 2>/dev/null || true

python "${SCRIPT_DIR}/rotation_bootstrap_ci.py" \
  --all_models \
  --n_boot "${N_BOOT:-2000}" \
  --seed "${SEED:-0}" \
  "$@"
