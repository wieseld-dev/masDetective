#!/bin/bash
#SBATCH --partition=main
#SBATCH --time=0-00:45:00
#SBATCH --job-name=robgenval
#SBATCH --output=logs/robgen-eval-%J.out
#SBATCH --error=logs/robgen-eval-%J.err
#SBATCH --mem=40G
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --constraint=rtx_4090

set -euo pipefail
cd "$HOME/mas_rl_detective"
mkdir -p logs artifacts/robust_visible_generator_v1/eval
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate masdetective

python -u analysis/eval_robust_visible_generator_v1.py
