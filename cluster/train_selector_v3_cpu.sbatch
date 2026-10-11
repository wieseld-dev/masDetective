#!/bin/bash
#SBATCH --job-name=sel-v3
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=0-00:45:00
#SBATCH --output=logs/sel-v3-%j.out
#SBATCH --error=logs/sel-v3-%j.err

set -euo pipefail

cd "$HOME/mas_rl_detective"

export PYTHONPATH="$HOME/mas_rl_detective:${PYTHONPATH:-}"
export PATH="$HOME/.conda/envs/masdetective/bin:$PATH"

PY="$HOME/.conda/envs/masdetective/bin/python"

"$PY" -u scripts/train_selector_v3_pairwise.py \
    --root artifacts/adaptive_v11_curriculum_v2 \
    --out artifacts/adaptive_v11_curriculum_v2/selector_v3 \
    --epochs 60 \
    --batch-size 4096 \
    --lr 1e-3 \
    --weight-decay 1e-4 \
    --min-gap 0.02 \
    --patience 8
