#!/bin/bash -l
set -euo pipefail

#$ -P ivc-ml
#$ -pe omp 16
#$ -l gpus=1
#$ -l gpu_c=8.0
#$ -l h_rt=42:00:00
#$ -N spatial_illusion_pacs
#$ -t 1-1              # Array: one task per model (update bound if MODELS length changes)
#$ -j y
#$ -o logs  # Update to your cluster log directory

module load miniconda
conda activate transformers4571

# Spatial Illusion (PACS rotation): zmeurer pac/rotation_recog protocol
#   - Data: pass via --data_dir argument
#   - 200 balanced samples per domain; positive = X vs rotate(X); negative = X vs Y
#   - Domains: art_painting, cartoon, photo, sketch (all by default)
#
# Optional per-domain parallel jobs (like zmeurer run_rotation_recog_parallel.sh):
#   #$ -t 1-4
#   and set USE_DOMAIN_ARRAY=1 below.

BASE_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
DATA_DIR="${BASE_DIR}/DATA"
OUTPUT_DIR="${BASE_DIR}/results_pacs"

PROMPT_VERSION="direct"
PROMPT_TAG=""   # e.g. prompt_rotate_modified — set to append tag to output subdir

USE_DOMAIN_ARRAY=0
DOMAINS=("art_painting" "cartoon" "photo" "sketch")

MODELS=(
    "OpenGVLab/InternVL3_5-8B"
)

IDX=$((SGE_TASK_ID - 1))
if [[ ${USE_DOMAIN_ARRAY} -eq 1 ]]; then
    if [[ ${IDX} -lt 0 || ${IDX} -ge ${#DOMAINS[@]} ]]; then
        echo "Invalid SGE_TASK_ID=${SGE_TASK_ID} for domain array (expected 1-${#DOMAINS[@]})"
        exit 1
    fi
    DOMAIN_ARG=(--domain "${DOMAINS[${IDX}]}")
    TASK_LABEL="domain=${DOMAINS[${IDX}]}"
else
    DOMAIN_ARG=()
    if [[ ${IDX} -lt 0 || ${IDX} -ge ${#MODELS[@]} ]]; then
        echo "Invalid SGE_TASK_ID=${SGE_TASK_ID} (expected 1-${#MODELS[@]})"
        exit 1
    fi
    TASK_LABEL="model=${MODELS[${IDX}]}"
fi

if [[ ${USE_DOMAIN_ARRAY} -eq 1 ]]; then
    MODEL="${MODELS[0]}"
else
    MODEL="${MODELS[${IDX}]}"
fi

echo "Task ${SGE_TASK_ID}: ${TASK_LABEL} model=${MODEL}"

mkdir -p "${OUTPUT_DIR}"

MODEL_SLUG=$(echo "${MODEL}" | tr '/.' '__' | tr '[:upper:]' '[:lower:]')
CSV_SLUG=$(echo "${MODEL}" | sed 's/[^a-zA-Z0-9]\+/_/g' | tr '[:upper:]' '[:lower:]' | sed 's/^_*//;s/_*$//')
LOG_PATH="${OUTPUT_DIR}/${MODEL_SLUG}_spatial_illusion_pacs.log"

MODEL_OUTPUT_DIR="${OUTPUT_DIR}/spatial_illusion_pacs_${MODEL_SLUG}"
if [[ -n "${PROMPT_TAG}" ]]; then
    EXP_SUBDIR="spatial_illusion_pacs_${PROMPT_TAG}"
else
    EXP_SUBDIR="spatial_illusion_pacs"
fi
RESUME_CSV="${MODEL_OUTPUT_DIR}/${EXP_SUBDIR}/${CSV_SLUG}_spatial_illusion_pacs.csv"
mkdir -p "${MODEL_OUTPUT_DIR}"

PROMPT_ARGS=(--prompt_version "${PROMPT_VERSION}")
if [[ -n "${PROMPT_TAG}" ]]; then
    PROMPT_ARGS+=(--prompt_tag "${PROMPT_TAG}")
fi

python -m experiments.identity.spatial_illusion_pacs \
    --data_dir "${DATA_DIR}" \
    --model "${MODEL}" \
    --output_dir "${MODEL_OUTPUT_DIR}" \
    --log_path "${LOG_PATH}" \
    --image_size 336 \
    --rate_limit_delay 0.0 \
    --resume_csv "${RESUME_CSV}" \
    ${DOMAIN_ARG[@]+"${DOMAIN_ARG[@]}"} \
    ${PROMPT_ARGS[@]+"${PROMPT_ARGS[@]}"}

echo "Task ${SGE_TASK_ID} completed. Results: ${MODEL_OUTPUT_DIR}/${EXP_SUBDIR}/"
