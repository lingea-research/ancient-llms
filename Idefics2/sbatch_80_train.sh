#!/bin/bash

#SBATCH -J train-i2
#SBATCH --account open-29-5
#SBATCH --partition qgpu
#SBATCH --gpus 1
#SBATCH --time 24:00:00

export HF_HOME="/mnt/proj2/open-29-5/mtm24-ancient/.cache/huggingface"
export HF_DATASETS_CACHE="/mnt/proj2/open-29-5/mtm24-ancient/.cache/huggingface/datasets"

source /mnt/proj2/open-29-5/wmt24-mirek/miniforge3/bin/activate "/mnt/proj2/open-29-5/mtm24-ancient/envs/idefics2"

python3 train.py

