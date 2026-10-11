#!/bin/bash
#SBATCH --partition=main
#SBATCH --time=0-00:55:00
#SBATCH --job-name=sel-v2
#SBATCH --output=logs/sel-v2-%J.out
#SBATCH --error=logs/sel-v2-%J.err
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --gpus=0

set -euo pipefail
cd "$HOME/mas_rl_detective"
export PYTHONPATH="$HOME/mas_rl_detective:${PYTHONPATH:-}"
mkdir -p logs artifacts/visible_selector_v2_pairwise
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate masdetective
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
python -u analysis/train_visible_selector_v2_pairwise.py --particles-per-k 8 --pca-dim 24
