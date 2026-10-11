#!/bin/bash
#SBATCH --partition=main
#SBATCH --time=0-00:30:00
#SBATCH --job-name=robgenlbl
#SBATCH --output=logs/robgen-label-%J.out
#SBATCH --error=logs/robgen-label-%J.err
#SBATCH --mem=24G
#SBATCH --cpus-per-task=8
#SBATCH --gpus=0

set -euo pipefail
cd "$HOME/mas_rl_detective"
mkdir -p logs artifacts/robust_visible_generator_v1/train_rows
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate masdetective

python -u analysis/build_robust_visible_generator_rows_v1.py
