#!/bin/bash
#SBATCH --job-name=v4-v2-select
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --constraint=rtx_4090
#SBATCH --exclude=cs-4090-09
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=10:00:00
#SBATCH --output=artifacts/HIGHEST_CLEAN_AUC/experiments/08_SELECTOR_V4_HYBRID_TRAIN_V1/logs/%x-%j.out
#SBATCH --error=artifacts/HIGHEST_CLEAN_AUC/experiments/08_SELECTOR_V4_HYBRID_TRAIN_V1/logs/%x-%j.err

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

echo "===== V2 BEHAVIOR SELECTOR ====="
echo "IMPORTANT: all 4 candidates were ALREADY executed+scored."
echo "This step only chooses the public trajectory arm."

"$PY" -u \
  analysis/select_v14_candidates_llmv2.py \
  --root "$V14A_ROOT" \
  --round "$V14A_ROUND" \
  --split train \
  --adapter "$HCA/models/selector_v2"

echo
echo "===== POOL / SELECTED ====="

RR=$(printf '%02d' "$V14A_ROUND")

wc -l \
  "$V14A_ROOT/candidate_pool_train_round_${RR}.jsonl" \
  "$V14A_ROOT/candidates_train_round_${RR}.jsonl"

echo "DONE"
