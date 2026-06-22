#!/bin/bash -l
# ------ SGE options ------
#$ -P ivc-ml
#$ -N qwen_rotation_recog_parallel
#$ -j y
#$ -o logs  # Update to your cluster log directory
#$ -pe omp 4                 # 4 CPU cores
#$ -l gpus=1                 # 1 GPU
#$ -l gpu_c=7.0
#$ -l h_rt=20:00:00
#$ -t 1-50                   # Job array: tasks 1–10

# --------------------------
# Script list
# --------------------------
SCRIPT_LIST=("Gurmukhi" "Latin" "Futurama" "Cyrillic" "Atemayar_Qelisayer" "Syriac_(Estrangelo)" "Armenian" "Greek" "Mongolian" "Glagolitic" "Hebrew" "Sanskrit" "Angelic" "Braille" "Kannada" "Balinese" "Atlantean" "Japanese_(katakana)" "Arcadian" "ULOG" "Alphabet_of_the_Magi" "Tifinagh" "Keble" "Asomtavruli_(Georgian)" "Anglo-Saxon_Futhorc" "Tagalog" "Tengwar" "Mkhedruli_(Georgian)" "Gujarati" "Oriya" "Old_Church_Slavonic_(Cyrillic)" "Malayalam" "Syriac_(Serto)" "Bengali" "Grantha" "Burmese_(Myanmar)" "Avesta" "Malay_(Jawi_-_Arabic)" "Tibetan" "Sylheti" "Blackfoot_(Canadian_Aboriginal_Syllabics)" "Japanese_(hiragana)" "Ojibwe_(Canadian_Aboriginal_Syllabics)" "Early_Aramaic" "Aurek-Besh" "Inuktitut_(Canadian_Aboriginal_Syllabics)" "Korean" "Manipuri" "N_Ko" "Ge_ez")
# Figure out which script corresponds to this task ID
IDX=$((SGE_TASK_ID - 1))
SCRIPT_NAME=${SCRIPT_LIST[$IDX]}
PROMPT_NAME="prompt_3"
TEXT_PROMPT="If I rotate the first image, can I get the second image? Answer in curly brackets, e.g. {Yes} or {No}."
MODEL="qwen"
BASE_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
LOG_DIR="${BASE_DIR}/${MODEL}_logs/${PROMPT_NAME}"
mkdir -p "$LOG_DIR"
LOG_PATH="${LOG_DIR}/rotation_recog.log"
OUTPUT_FOLDER="${BASE_DIR}/${MODEL}_responses/${PROMPT_NAME}/parallel_csvs"
mkdir -p "$OUTPUT_FOLDER"
FINETUNE_DIR="" # Path to finetuned model directory
DATA_DIR="${BASE_DIR}/DATA/omniglot-master/python"

echo "Task $SGE_TASK_ID running script: $SCRIPT_NAME"

# --------------------------
# Run Python module
# --------------------------

module load miniconda
conda activate blind

python -m experiments.rotation.rotation_recog --script "$SCRIPT_NAME" --prompt_name "$PROMPT_NAME" --text_prompt "$TEXT_PROMPT" --model "$MODEL" --gemini_start_idx 0 --log_path "$LOG_PATH" --output_dir "$OUTPUT_FOLDER" --data_dir "$DATA_DIR"


echo "Task $SGE_TASK_ID completed script: $SCRIPT_NAME"