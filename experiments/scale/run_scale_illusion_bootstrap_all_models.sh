#!/bin/bash -l
# Bootstrap CIs for all main scale-illusion model runs under results/.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

module load miniconda 2>/dev/null || true
conda activate blind 2>/dev/null || true

cd "${REPO_ROOT}"

python -m experiments.scale.scale_illusion_bootstrap_ci \
  --all_models \
  --n_boot "${N_BOOT:-2000}" \
  --seed "${SEED:-0}" \
  "$@"
