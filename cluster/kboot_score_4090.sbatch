#!/bin/bash
#SBATCH --job-name=kb-score
#SBATCH --partition=main
#SBATCH --gpus=1
#SBATCH --constraint=rtx_4090
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/kb-score-%j.out
#SBATCH --error=logs/kb-score-%j.err
set -euo pipefail
cd "$HOME/mas_rl_detective"
export PYTHONPATH="$HOME/mas_rl_detective:${PYTHONPATH:-}"
export PATH="$HOME/.conda/envs/masdetective/bin:$PATH"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
PY="$HOME/.conda/envs/masdetective/bin/python"
: "${KBOOT_ROOT:?set KBOOT_ROOT}"
: "${KBOOT_ROUND:?set KBOOT_ROUND}"
: "${KBOOT_SPLIT:?set KBOOT_SPLIT}"
"$PY" -u scripts/adaptive_v11_score_round.py \
  --round "$KBOOT_ROUND" \
  --split "$KBOOT_SPLIT" \
  --root "$KBOOT_ROOT"
