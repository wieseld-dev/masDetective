#!/bin/bash
#SBATCH --job-name=v14a-score
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --constraint=rtx_4090
#SBATCH --exclude=cs-4090-09
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/v14a-score-%j.out
#SBATCH --error=logs/v14a-score-%j.err

set -euo pipefail

cd "$HOME/mas_rl_detective"

export PYTHONPATH="$HOME/mas_rl_detective:${PYTHONPATH:-}"
export PATH="$HOME/.conda/envs/masdetective/bin:$PATH"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

PY="$HOME/.conda/envs/masdetective/bin/python"

: "${V14A_ROOT:?}"
: "${V14A_ROUND:?}"
: "${V14A_SPLIT:?}"

"$PY" -u scripts/adaptive_v11_score_round.py \
    --round "$V14A_ROUND" \
    --split "$V14A_SPLIT" \
    --root "$V14A_ROOT"
