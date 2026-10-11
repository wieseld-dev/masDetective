#!/bin/bash
#SBATCH --job-name=haa-score
#SBATCH --partition=main
#SBATCH --gpus=1
#SBATCH --constraint=rtx_4090
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/haa-score-%j.out
#SBATCH --error=logs/haa-score-%j.err

set -euo pipefail

cd "$HOME/mas_rl_detective"

export PYTHONPATH="$HOME/mas_rl_detective:${PYTHONPATH:-}"
export PATH="$HOME/.conda/envs/masdetective/bin:$PATH"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1

PY="$HOME/.conda/envs/masdetective/bin/python"

: "${HAA_ROOT:?}"
: "${HAA_ROUND:?}"

"$PY" -u scripts/adaptive_v11_score_round.py \
  --round "$HAA_ROUND" \
  --split dev \
  --root "$HAA_ROOT"
