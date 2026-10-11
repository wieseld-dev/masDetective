#!/bin/bash
#SBATCH --job-name=ctrl-score
#SBATCH --partition=main
#SBATCH --gpus=1
#SBATCH --constraint=rtx_4090
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/ctrl-score-%j.out
#SBATCH --error=logs/ctrl-score-%j.err
set -euo pipefail
cd "$HOME/mas_rl_detective"
export PYTHONPATH="$HOME/mas_rl_detective:${PYTHONPATH:-}"
export PATH="$HOME/.conda/envs/masdetective/bin:$PATH"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
PY="$HOME/.conda/envs/masdetective/bin/python"
: "${CTRL_ROOT:?set CTRL_ROOT}"
: "${CTRL_ROUND:?set CTRL_ROUND}"
: "${CTRL_SPLIT:?set CTRL_SPLIT}"
"$PY" -u scripts/adaptive_v11_score_round.py \
  --round "$CTRL_ROUND" \
  --split "$CTRL_SPLIT" \
  --root "$CTRL_ROOT"
