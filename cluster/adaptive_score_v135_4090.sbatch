#!/bin/bash
#SBATCH --job-name=adapt-score
#SBATCH --partition=main
#SBATCH --gpus=1
#SBATCH --constraint=rtx_4090
#SBATCH --mem=32G
#SBATCH --output=logs/adapt-score-%j.out
#SBATCH --error=logs/adapt-score-%j.err
set -euo pipefail
cd "$HOME/mas_rl_detective"
source /storage/modules/packages/anaconda/etc/profile.d/conda.sh
conda activate masdetective
python -u adaptive_score_v135.py --item-batch 4 --max-length 768
