#!/bin/bash
#SBATCH --job-name=sel-v3-auc
#SBATCH --partition=gpu
#SBATCH --constraint=rtx_4090
#SBATCH --gpus=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=40G
#SBATCH --time=10:00:00
#SBATCH --exclude=cs-4090-09
#SBATCH --output=artifacts/HIGHEST_CLEAN_AUC/experiments/05_SELECTOR_V3_AUC/logs/%x-%j.out
#SBATCH --error=artifacts/HIGHEST_CLEAN_AUC/experiments/05_SELECTOR_V3_AUC/logs/%x-%j.err

set -euo pipefail

cd "$HOME/mas_rl_detective"

export PYTHONPATH="$HOME/mas_rl_detective:${PYTHONPATH:-}"
export PATH="$HOME/.conda/envs/masdetective/bin:$PATH"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1

PY="$HOME/.conda/envs/masdetective/bin/python"

HCA="artifacts/HIGHEST_CLEAN_AUC"
EXP="$HCA/experiments/05_SELECTOR_V3_AUC"
TRAIN="$EXP/data/selector_v3_auc_compat.jsonl"
MODEL="$EXP/model"
EVAL="$EXP/eval_r7"

echo "===== TRAIN SELECTOR V3-AUC ====="
date

"$PY" analysis/train_llm_selector_v1.py \
  --train "$TRAIN" \
  --output-dir "$MODEL" \
  --model-name "Qwen/Qwen2.5-3B-Instruct" \
  --epochs 4 \
  --val-fraction 0.25 \
  --lambda-k 0.0 \
  --patience 2 \
  --seed 1397

echo
echo "===== V3 TRAINING SUMMARY ====="

cat "$MODEL/summary.json"

echo
echo "===== OFFLINE R7 SELECTION ====="

"$PY" \
  "$HCA/code/entrypoints/analysis/select_v14_candidates_llmv2.py" \
  --root "$EVAL" \
  --round 7 \
  --split dev \
  --adapter "$MODEL/best_adapter"

echo
echo "===== DONE ====="
date
