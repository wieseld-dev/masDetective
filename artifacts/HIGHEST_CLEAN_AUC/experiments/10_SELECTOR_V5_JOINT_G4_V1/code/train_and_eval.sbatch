#!/bin/bash
#SBATCH --job-name=sel-v5-joint
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --constraint=rtx_4090
#SBATCH --exclude=cs-4090-09
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=10:00:00
#SBATCH --output=artifacts/HIGHEST_CLEAN_AUC/experiments/10_SELECTOR_V5_JOINT_G4_V1/logs/%x-%j.out
#SBATCH --error=artifacts/HIGHEST_CLEAN_AUC/experiments/10_SELECTOR_V5_JOINT_G4_V1/logs/%x-%j.err

set -euo pipefail

cd "$HOME/mas_rl_detective"

export PYTHONPATH="$HOME/mas_rl_detective:${PYTHONPATH:-}"
export PATH="$HOME/.conda/envs/masdetective/bin:$PATH"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

PY="$HOME/.conda/envs/masdetective/bin/python"

HCA="artifacts/HIGHEST_CLEAN_AUC"
EXP="$HCA/experiments/10_SELECTOR_V5_JOINT_G4_V1"

TRAIN="$HCA/experiments/09_SELECTOR_V4_G4_RANKING_V1/data/selector_v4_g4.jsonl"

echo "============================================"
echo " SELECTOR V5 — JOINT G=4"
echo "============================================"
date

echo
echo "===== TRAIN ====="

"$PY" -u \
  "$EXP/code/train_joint_v5.py" \
  --train "$TRAIN" \
  --output-dir "$EXP/model" \
  --epochs 4 \
  --batch-size 2 \
  --grad-accum 8 \
  --lr 1e-4 \
  --val-fraction 0.25 \
  --soft-tau 0.06 \
  --soft-weight 0.65 \
  --hard-weight 0.35 \
  --patience 2 \
  --seed 1397

echo
echo "===== OFFLINE HYBRID R7 ====="

"$PY" -u \
  "$EXP/code/select_joint_v5.py" \
  --root "$EXP/eval_r7" \
  --round 7 \
  --split dev \
  --adapter "$EXP/model/best_adapter"

echo
echo "===== FINISHED ====="
date
