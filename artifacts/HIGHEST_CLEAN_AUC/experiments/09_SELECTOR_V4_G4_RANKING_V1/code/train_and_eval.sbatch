#!/bin/bash
#SBATCH --job-name=sel-v4-g4
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --constraint=rtx_4090
#SBATCH --exclude=cs-4090-09
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
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

TRAIN="$EXP/data/selector_v4_g4.jsonl"
MODEL="$EXP/model"
EVAL="$EXP/eval_r7"

echo "=========================================="
echo " SELECTOR V4 — G4 RANKING"
echo "=========================================="
date

echo
echo "===== TRAIN ====="

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
echo "===== SUMMARY ====="

cat "$MODEL/summary.json"

echo
echo "===== OFFLINE HYBRID R7 ====="

"$PY" \
  "$HCA/code/entrypoints/analysis/select_v14_candidates_llmv2.py" \
  --root "$EVAL" \
  --round 7 \
  --split dev \
  --adapter "$MODEL/best_adapter"

echo
echo "=========================================="
echo " DONE"
echo "=========================================="
date
