#!/bin/bash
#SBATCH --job-name=la-score
#SBATCH --partition=main
#SBATCH --gpus=1
#SBATCH --constraint=rtx_4090
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/la-score-%j.out
#SBATCH --error=logs/la-score-%j.err

set -euo pipefail

cd "$HOME/mas_rl_detective"

export PYTHONPATH="$HOME/mas_rl_detective:${PYTHONPATH:-}"
export PATH="$HOME/.conda/envs/masdetective/bin:$PATH"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1

"$HOME/.conda/envs/masdetective/bin/python" \
  -u scripts/adaptive_v11_score_round.py \
  --round 1 \
  --split dev \
  --root artifacts/local_ambiguity_v1
