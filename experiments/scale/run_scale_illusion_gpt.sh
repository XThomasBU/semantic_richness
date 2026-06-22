#!/bin/bash -l
set -euo pipefail

# Set SCC project
#$ -P ivc-ml
#$ -t 1-1  # Array job specification (prompt variants)
#$ -pe omp 16 # Request 24 CPU cores
#$ -l gpus=1 # Request 1 GPU
#$ -l gpu_c=8.0
#$ -l h_rt=21:00:00
#$ -N scale_illusion_gpt
#$ -j y # Merge standard output and error

module load miniconda
conda activate blind

# GPT-only scale illusion run (restricted Omniglot scripts)

BASE_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
OUTPUT_DIR="${BASE_DIR}/results"
LOG_PATH="${OUTPUT_DIR}/scale_illusion_gpt.log"

# Prompt variants (v1=direct)
PROMPTS=("v2")
PROMPT_TAGS=("prompt_v2")

PROMPT_VERSION="v2"
PROMPT_TAG="prompt_v2"

# Resume from corresponding CSV if it exists (each prompt variant has its own)
RESUME_CSV="${OUTPUT_DIR}/scale_illusion_${PROMPT_TAG}/gpt_5_2_scale_illusion.csv"

mkdir -p "${OUTPUT_DIR}"

python -m experiments.scale.scale_illusion \
    --alphabet_dir "${BASE_DIR}" \
    --omniglot_dir "${BASE_DIR}" \
    --model gpt-5.2 \
    --output_dir "${OUTPUT_DIR}" \
    --log_path "${LOG_PATH}" \
    --scale_factors 0.1 0.3 0.5 0.9 \
    --prompt_version "${PROMPT_VERSION}" \
    --prompt_tag "${PROMPT_TAG}" \
    --resume_csv "${RESUME_CSV}"

echo "GPT run completed. Results saved to ${OUTPUT_DIR}/scale_illusion_gpt/"
echo "  - Main results: ${OUTPUT_DIR}/scale_illusion_gpt/"
echo "  - Sanity checks: ${OUTPUT_DIR}/scale_illusion_gpt/sanity_check/"

# --omniglot_scripts Angelic Bengali Braille Greek Latin Malayalam Korean Kannada Keble \


#   --focus_scripts English hand_digits hand_english Greek Latin Braille Malayalam Bengali Sanskrit Angelic \
#   --max_examples_per_script 3