#!/bin/bash -l
set -euo pipefail

# Set SCC project
#$ -P ivc-ml
#$ -t 1-1  # Array job specification (prompt variants)
#$ -pe omp 16 # Request 24 CPU cores
#$ -l gpus=1 # Request 1 GPU
#$ -l gpu_c=7.0
#$ -l h_rt=21:00:00
#$ -N scale_illusion_gemini
#$ -j y # Merge standard output and error
#$ -o logs  # Update to your cluster log directory

module load miniconda
conda activate blind

# Gemini-only scale illusion run (restricted Omniglot scripts)

BASE_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
OUTPUT_DIR="${BASE_DIR}/results"
LOG_PATH="${OUTPUT_DIR}/scale_illusion_gemini.log"

# Prompt variants (v1=direct)
PROMPTS=("v2")
PROMPT_TAGS=("prompt_v2")

PROMPT_VERSION="${PROMPTS[$((SGE_TASK_ID-1))]}"
PROMPT_TAG="${PROMPT_TAGS[$((SGE_TASK_ID-1))]}"

# Resume from corresponding CSV if it exists (each prompt variant has its own)
RESUME_CSV="${OUTPUT_DIR}/scale_illusion_gemini_2_5_pro_${PROMPT_TAG}/gemini_2_5_pro_scale_illusion.csv"

mkdir -p "${OUTPUT_DIR}"

python -m experiments.scale.scale_illusion \
    --alphabet_dir "${BASE_DIR}" \
    --omniglot_dir "${BASE_DIR}" \
    --model gemini-2.5-pro \
    --output_dir "${OUTPUT_DIR}" \
    --log_path "${LOG_PATH}" \
    --scale_factors 0.1 0.5 0.9 \
    --prompt_version "${PROMPT_VERSION}" \
    --prompt_tag "${PROMPT_TAG}" \
    --resume_csv "${RESUME_CSV}"

echo "Gemini run completed. Results saved to ${OUTPUT_DIR}/scale_illusion_gemini_2_5_pro/"
echo "  - Main results: ${OUTPUT_DIR}/scale_illusion_gemini_2_5_pro/"
echo "  - Sanity checks: ${OUTPUT_DIR}/scale_illusion_gemini_2_5_pro/sanity_check/"

# --omniglot_scripts Angelic Bengali Braille Greek Latin Malayalam Korean Kannada Keble \
