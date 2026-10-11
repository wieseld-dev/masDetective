#!/bin/bash
#SBATCH --job-name=q38on-score
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --constraint=rtx_4090
#SBATCH --exclude=cs-4090-09
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/q38on-score-%j.out
#SBATCH --error=logs/q38on-score-%j.err

set -euo pipefail

cd "$HOME/mas_rl_detective"

export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export PATH="$HOME/.conda/envs/masdetective/bin:$PATH"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

: "${Q38_ROOT:?}"
: "${Q38_ROUND:?}"
: "${Q38_SPLIT:?}"

PY="$HOME/.conda/envs/masdetective/bin/python"

"$PY" -u \
  scripts/adaptive_v11_score_round.py \
  --round "$Q38_ROUND" \
  --split "$Q38_SPLIT" \
  --root "$Q38_ROOT"
