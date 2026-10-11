#!/bin/bash
#SBATCH --job-name=sel-v4-eval
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --constraint=rtx_4090
#SBATCH --exclude=cs-4090-09
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --output=artifacts/HIGHEST_CLEAN_AUC/experiments/09_SELECTOR_V4_G4_RANKING_V1/logs/%x-%j.out
#SBATCH --error=artifacts/HIGHEST_CLEAN_AUC/experiments/09_SELECTOR_V4_G4_RANKING_V1/logs/%x-%j.err

set -euo pipefail

cd "$HOME/mas_rl_detective"

export PYTHONPATH="$HOME/mas_rl_detective:${PYTHONPATH:-}"
export PATH="$HOME/.conda/envs/masdetective/bin:$PATH"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

PY="$HOME/.conda/envs/masdetective/bin/python"

HCA="artifacts/HIGHEST_CLEAN_AUC"
EXP="$HCA/experiments/09_SELECTOR_V4_G4_RANKING_V1"
EVAL="$EXP/eval_r7"

echo "===== V4 OFFLINE HYBRID R7 SELECTION ====="

"$PY" \
  "$HCA/code/entrypoints/analysis/select_v14_candidates_llmv2.py" \
  --root "$EVAL" \
  --round 7 \
  --split dev \
  --adapter "$EXP/model/best_adapter"

echo
echo "===== OUTPUT ====="

wc -l \
  "$EVAL/candidate_pool_dev_round_07.jsonl" \
  "$EVAL/candidates_dev_round_07.jsonl" \
  "$EVAL/selector_diag_dev_round_07.jsonl"

echo
echo "DONE"
