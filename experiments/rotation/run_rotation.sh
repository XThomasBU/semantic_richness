#!/bin/bash -l

# Set SCC project
#$ -P ivc-ml
#$ -pe omp 4 # Request 4 CPU cores
#$ -l gpus=1 # Request 1 GPU
#$ -l gpu_c=7.0 # gpu compute capability
#$ -l h_rt=20:00:00
#$ -N qwen_rotation
#$ -j y # Merge standard output and error
#$ -o logs  # Update to your cluster log directory

module load miniconda
conda activate blind

python -m experiments.rotation.rotator