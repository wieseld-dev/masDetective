#!/bin/bash
#SBATCH --job-name=k-gap-exact
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:50:00
#SBATCH --output=logs/k-gap-exact-%j.out
#SBATCH --error=logs/k-gap-exact-%j.err

set -euo pipefail
cd "$HOME/mas_rl_detective"

export PYTHONPATH="$HOME/mas_rl_detective:${PYTHONPATH:-}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate masdetective

python -u analysis/k_gap_exact_current_v1.py \
  --budget 5 \
  --oracle-beta 1.0 \
  --batch-size 4096
