#!/bin/bash -l

# Set SCC project
#$ -P ivc-ml
#$ -pe omp 4 # Request 4 CPU cores
#$ -l gpus=1 # Request 1 GPU
#$ -l gpu_c=7.0 # gpu compute capability
#$ -l h_rt=20:00:00
#$ -N qwen_rotation_recog
#$ -j y # Merge standard output and error
#$ -o logs  # Update to your cluster log directory

module load miniconda
conda activate blind

#PROMPT_NAME="prompt_3"
#TEXT_PROMPT="If I rotate the first image, can I get the second image? Answer in curly brackets, e.g. {Yes} or {No}."

MODEL="qwen"
PROMPT_NAME="prompt_4"
TEXT_PROMPT="If I rotate the first image, can I get the second image? Answer in curly brackets, e.g. {Yes} or {No}."
BASE_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
LOG_DIR="${BASE_DIR}/${MODEL}_logs/${PROMPT_NAME}"
mkdir -p "$LOG_DIR"
LOG_PATH="${LOG_DIR}/rotation_recog.log"
OUTPUT_FOLDER="${BASE_DIR}/${MODEL}_responses/${PROMPT_NAME}/parallel_csvs"
mkdir -p "$OUTPUT_FOLDER"
FINETUNE_DIR="" # Path to finetuned model directory
DATA_DIR="${BASE_DIR}/DATA/omniglot-master/python"


python -m experiments.rotation.rotation_recog --script "Armenian" --prompt_name "$PROMPT_NAME" --text_prompt "$TEXT_PROMPT" --model "$MODEL" --gemini_start_idx 0 --log_path "$LOG_PATH" --output_dir "$OUTPUT_FOLDER" --data_dir "$DATA_DIR"
echo "$PROMPT_NAME completed script: Armenian"