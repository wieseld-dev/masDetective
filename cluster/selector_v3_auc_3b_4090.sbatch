#!/bin/bash
#SBATCH --job-name=selv3-3b
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=rtx_4090
#SBATCH --exclude=cs-4090-09
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=20:00:00
#SBATCH --output=logs/selv3-3b-%j.out
#SBATCH --error=logs/selv3-3b-%j.err

set -euo pipefail

cd "$HOME/mas_rl_detective"

export PYTHONPATH="$HOME/mas_rl_detective:${PYTHONPATH:-}"
export PATH="$HOME/.conda/envs/masdetective/bin:$PATH"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PY="$HOME/.conda/envs/masdetective/bin/python"

"$PY" -u \
  analysis/train_llm_selector_v3_auc.py \
  --train artifacts/llm_selector_v3_auc_data/train.jsonl \
  --challenge artifacts/llm_selector_v3_auc_data/challenge.jsonl \
  --output-dir artifacts/llm_selector_v3_auc_3b \
  --model-name Qwen/Qwen2.5-3B-Instruct \
  --epochs 6 \
  --lr 2e-5 \
  --grad-acc 8 \
  --max-length 1024 \
  --eval-batch-size 8 \
  --val-fraction 0.15 \
  --lambda-k 0.0 \
  --target-tau 0.50 \
  --pred-tau 1.0 \
  --patience 2 \
  --seed 1397
