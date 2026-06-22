#!/bin/bash -l
# Example: bootstrap CIs for Qwen3-8B identity recognition on PACS.
#
# Usage:
#   bash run_rotation_bootstrap_ci.sh
#   bash run_rotation_bootstrap_ci.sh qwen_3_8B prompt_rotate_modified --by_angle

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="${1:-qwen_3_8B}"
PROMPT_NAME="${2:-prompt_identity_new}"
EXTRA_ARGS=("${@:3}")

module load miniconda 2>/dev/null || true
conda activate blind 2>/dev/null || true

python "${SCRIPT_DIR}/rotation_bootstrap_ci.py" \
  --model_dir "${MODEL_DIR}" \
  --prompt_name "${PROMPT_NAME}" \
  --by_angle \
  "${EXTRA_ARGS[@]}"
