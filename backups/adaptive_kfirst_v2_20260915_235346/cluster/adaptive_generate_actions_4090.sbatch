#!/bin/bash
#SBATCH --job-name=adapt-gen
#SBATCH --partition=main
#SBATCH --gpus=1
#SBATCH --constraint=rtx_4090
#SBATCH --mem=32G
#SBATCH --output=logs/adapt-gen-%j.out
#SBATCH --error=logs/adapt-gen-%j.err
set -euo pipefail
cd "$HOME/mas_rl_detective"
source /storage/modules/packages/anaconda/etc/profile.d/conda.sh
conda activate masdetective
python -u adaptive_generate_action_library.py --adapter artifacts/v11_2_split/policy --variants 1 --no-4bit
