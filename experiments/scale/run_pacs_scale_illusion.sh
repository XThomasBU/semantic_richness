#!/bin/bash -l
set -euo pipefail

#$ -P ivc-ml
#$ -pe omp 16          # Request 16 CPU cores
#$ -l gpus=1           # Request 1 GPU
#$ -l gpu_c=7.0
#$ -l h_rt=42:00:00
#$ -N scale_illusion_pacs
#$ -t 1-1              # Array job: one task per model (update upper bound if MODELS length changes)
#$ -j y                 # Merge standard output and error
#$ -o logs  # Update to your cluster log directory

module load miniconda
conda activate blind

# Run Scale Illusion (PACS) experiment: same-object vs different-object at different scales
# Array job: task ID selects which model to run (one model per job).

# Set paths
BASE_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
PACS_DIR="${BASE_DIR}/DATA/pacs_subset"
OUTPUT_DIR="${BASE_DIR}/results_pacs"

# Prompt setup (single prompt version for all jobs)
PROMPT_VERSION="v2"
PROMPT_TAG="prompt_v2"

# One model per array index (1-based SGE_TASK_ID)
MODELS=(
    "Qwen/Qwen3-VL-30B-A3B-Instruct"
)

IDX=$((SGE_TASK_ID - 1))
if [[ ${IDX} -lt 0 || ${IDX} -ge ${#MODELS[@]} ]]; then
    echo "Invalid SGE_TASK_ID=${SGE_TASK_ID} (expected 1-${#MODELS[@]})"
    exit 1
fi

MODEL="${MODELS[${IDX}]}"
echo "Array task ${SGE_TASK_ID}/${#MODELS[@]}: running model ${MODEL}"

# Create output directory
mkdir -p "${OUTPUT_DIR}"

MODEL_SLUG=$(echo "${MODEL}" | tr '/.' '__' | tr '[:upper:]' '[:lower:]')
CSV_SLUG=$(echo "${MODEL}" | sed 's/[^a-zA-Z0-9]\+/_/g' | tr '[:upper:]' '[:lower:]' | sed 's/^_*//;s/_*$//')
LOG_PATH="${OUTPUT_DIR}/${MODEL_SLUG}_scale_illusion_pacs.log"

MODEL_OUTPUT_DIR="${OUTPUT_DIR}/scale_illusion_pacs_${MODEL_SLUG}"
RESUME_CSV="${MODEL_OUTPUT_DIR}/scale_illusion_pacs_${PROMPT_TAG}/${CSV_SLUG}_scale_illusion_pacs.csv"
mkdir -p "${MODEL_OUTPUT_DIR}"

python -m experiments.scale.scale_illusion_pacs \
    --pacs_dir "${PACS_DIR}" \
    --model "${MODEL}" \
    --output_dir "${MODEL_OUTPUT_DIR}" \
    --log_path "${LOG_PATH}" \
    --scale_factors 0.1 0.3 0.5 0.9 \
    --rate_limit_delay 0.0 \
    --prompt_version "${PROMPT_VERSION}" \
    --prompt_tag "${PROMPT_TAG}" \
    --resume_csv "${RESUME_CSV}"

echo "Task ${SGE_TASK_ID} completed. Results: ${MODEL_OUTPUT_DIR}/scale_illusion_pacs_${PROMPT_TAG}/"
