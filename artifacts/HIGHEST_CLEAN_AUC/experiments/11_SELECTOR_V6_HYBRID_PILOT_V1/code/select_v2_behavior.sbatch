#!/bin/bash
#SBATCH --job-name=v6-v2-behav
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --constraint=rtx_4090
#SBATCH --exclude=cs-4090-09
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=10:00:00
#SBATCH --output=artifacts/HIGHEST_CLEAN_AUC/experiments/11_SELECTOR_V6_HYBRID_PILOT_V1/logs/%x-%j.out
#SBATCH --error=artifacts/HIGHEST_CLEAN_AUC/experiments/11_SELECTOR_V6_HYBRID_PILOT_V1/logs/%x-%j.err

set -euo pipefail

cd "$HOME/mas_rl_detective"

export PYTHONPATH="$HOME/mas_rl_detective:${PYTHONPATH:-}"
export PATH="$HOME/.conda/envs/masdetective/bin:$PATH"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1

PY="$HOME/.conda/envs/masdetective/bin/python"

: "${V14A_ROOT:?}"
: "${V14A_ROUND:?}"

HCA="artifacts/HIGHEST_CLEAN_AUC"

"$PY" -u \
  analysis/select_v14_candidates_llmv2.py \
  --root "$V14A_ROOT" \
  --round "$V14A_ROUND" \
  --split train \
  --adapter "$HCA/models/selector_v2"

RR=$(printf "%02d" "$V14A_ROUND")

echo
echo "===== AFTER SELECTOR ====="

wc -l \
  "$V14A_ROOT/candidate_pool_train_round_${RR}.jsonl" \
  "$V14A_ROOT/candidates_train_round_${RR}.jsonl"
