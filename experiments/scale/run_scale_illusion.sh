#!/bin/bash -l
set -euo pipefail

#$ -P ivc-ml
#$ -pe omp 16          # Request 16 CPU cores
#$ -l gpus=1           # Request 1 GPU
#$ -l gpu_c=8.0
#$ -l h_rt=42:00:00
#$ -N scale_illusion
#$ -j y                 # Merge standard output and error
#$ -o logs  # Update to your cluster log directory

module load miniconda
conda activate transformers4571  

# Example script to run the Scale Illusion experiment (all scripts + English)

# Set paths
BASE_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
OUTPUT_DIR="${BASE_DIR}/results"

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Prompt setup (single prompt version for all jobs)
PROMPT_VERSION="v2"
PROMPT_TAG="prompt_v2"

# Run experiments for each model separately
MODELS=(
    "allenai/Molmo2-8B"
)

for MODEL in "${MODELS[@]}"; do
    MODEL_SLUG=$(echo "${MODEL}" | tr '/.' '__' | tr '[:upper:]' '[:lower:]')
    CSV_SLUG=$(echo "${MODEL}" | sed 's/[^a-zA-Z0-9]\+/_/g' | tr '[:upper:]' '[:lower:]' | sed 's/^_*//;s/_*$//')
    LOG_PATH="${OUTPUT_DIR}/${MODEL_SLUG}_scale_illusion.log"

    MODEL_OUTPUT_DIR="${OUTPUT_DIR}/scale_illusion_${MODEL_SLUG}"
    RESUME_CSV="${MODEL_OUTPUT_DIR}/scale_illusion_${PROMPT_TAG}/${CSV_SLUG}_scale_illusion.csv"
    mkdir -p "${MODEL_OUTPUT_DIR}"

    python -m experiments.scale.scale_illusion \
        --alphabet_dir "${BASE_DIR}" \
        --omniglot_dir "${BASE_DIR}" \
        --model "${MODEL}" \
        --output_dir "${MODEL_OUTPUT_DIR}" \
        --log_path "${LOG_PATH}" \
        --num_samples 0 \
        --scale_factors 0.1 0.3 0.5 0.9 \
        --rate_limit_delay 0.0 \
        --prompt_version "${PROMPT_VERSION}" \
        --prompt_tag "${PROMPT_TAG}" \
        --resume_csv "${RESUME_CSV}"
done

echo "All model runs completed. Results saved to ${OUTPUT_DIR}/scale_illusion_<model>/"
echo "  - Main results: ${OUTPUT_DIR}/scale_illusion_<model>/scale_illusion/"
echo "  - Sanity checks: ${OUTPUT_DIR}/scale_illusion_<model>/scale_illusion/sanity_check/"
