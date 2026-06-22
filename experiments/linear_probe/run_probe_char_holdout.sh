#!/bin/bash -l
set -euo pipefail

#$ -P ivc-ml
#$ -pe omp 4
#$ -l gpus=1
#$ -l gpu_c=8.0
#$ -l h_rt=40:00:00
#$ -N linear_probe_char_holdout
#$ -j y
#$ -o logs  # Update to your cluster log directory

module load miniconda
conda activate blind

cd "$(dirname "$0")/../.."

FEATURES_DIR=./results/linear_probe/features
OUTPUT_DIR=./results/linear_probe_char_holdout

for enc in siglip clip dino qwen; do
  python -m experiments.linear_probe.run_probe_char_holdout \
    --features_path "${FEATURES_DIR}/${enc}_images_all_all_strokes_features.pt" \
    --output_dir "${OUTPUT_DIR}"
done
