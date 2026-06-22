#!/bin/bash -l
set -euo pipefail

# Set SCC project
#$ -P ivc-ml
#$ -t 1-1  # Array job specification
#$ -pe omp 16
#$ -l gpus=1
#$ -l gpu_c=8.0
#$ -l h_rt=42:00:00
#$ -N spatial_illusion
#$ -j y # Merge standard output and error
#$ -o logs  # Update to your cluster log directory

module load miniconda
conda activate blind

# Spatial Illusion (rotation): rotation_recog protocol — images_original +
# precomputed images_rotated; negative distractors upright; first stroke per character.
# Same DATA_DIR as rotation experiment

BASE_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
ROTATION_DATA_DIR="${BASE_DIR}/DATA/omniglot-master/python"
OUTPUT_DIR="${BASE_DIR}/results"

mkdir -p "${OUTPUT_DIR}"

MODELS=(
    "gpt-5.2"
    "gemini-2.5-pro"
)

for MODEL in "${MODELS[@]}"; do
    MODEL_SLUG=$(echo "${MODEL}" | tr '/.' '__' | tr '[:upper:]' '[:lower:]')
    CSV_SLUG=$(echo "${MODEL}" | sed 's/[^a-zA-Z0-9]\+/_/g' | tr '[:upper:]' '[:lower:]' | sed 's/^_*//;s/_*$//')
    LOG_PATH="${OUTPUT_DIR}/${MODEL_SLUG}_spatial_illusion.log"
    RESUME_CSV="${OUTPUT_DIR}/spatial_illusion/${CSV_SLUG}_spatial_illusion.csv"

    python -m experiments.identity.spatial_illusion \
        --alphabet_dir "${BASE_DIR}" \
        --omniglot_dir "${BASE_DIR}" \
        --rotation_data_dir "${ROTATION_DATA_DIR}" \
        --model "${MODEL}" \
        --output_dir "${OUTPUT_DIR}" \
        --log_path "${LOG_PATH}" \
        --image_size 336 \
        --rate_limit_delay 0.0 \
        --experiment rotation \
        --resume_csv "${RESUME_CSV}"
done

echo "All model runs completed. Results saved to ${OUTPUT_DIR}/spatial_illusion/"
