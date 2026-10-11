#!/bin/bash
#SBATCH --job-name=bt-s5-score
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx_4090:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=20:00:00
#SBATCH --output=logs/bt-s5-score-%j.out
#SBATCH --error=logs/bt-s5-score-%j.err

set -euo pipefail

cd "$HOME/mas_rl_detective"

export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export PATH="$HOME/.conda/envs/masdetective/bin:$PATH"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

PY="$HOME/.conda/envs/masdetective/bin/python"

: "${S5_ROOT:?}"
: "${S5_ROUND:?}"

"$PY" -u analysis/bt_simple5_runtime_guard_v1.py \
  --root "$S5_ROOT" \
  --round "$S5_ROUND" \
  --phase post_exec

"$PY" -u scripts/adaptive_v11_score_round.py \
  --round "$S5_ROUND" \
  --split dev \
  --root "$S5_ROOT"

"$PY" -u analysis/bt_simple5_runtime_guard_v1.py \
  --root "$S5_ROOT" \
  --round "$S5_ROUND" \
  --phase post_score

echo "SCORE_PASS ROUND=$S5_ROUND"
