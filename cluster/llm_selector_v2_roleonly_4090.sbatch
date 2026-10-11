#!/bin/bash
#SBATCH --job-name=sel-role-v2
#SBATCH --partition=gpu
#SBATCH --constraint=rtx_4090
#SBATCH --gres=gpu:1
#SBATCH --exclude=cs-4090-09
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=10:00:00
#SBATCH --output=logs/sel-role-v2-%j.out
#SBATCH --error=logs/sel-role-v2-%j.err

set -euo pipefail

cd "$HOME/mas_rl_detective"

PY="/home/wieseld/.conda/envs/masdetective/bin/python"

export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

"$PY" -u \
  analysis/train_llm_selector_v1.py \
  --model-name Qwen/Qwen2.5-3B-Instruct \
  --epochs 6 \
  --lr 2e-5 \
  --grad-acc 8 \
  --max-length 1024 \
  --eval-batch-size 16 \
  --val-fraction 0.20 \
  --lambda-k 0.00 \
  --target-tau 0.50 \
  --pred-tau 1.0 \
  --patience 2 \
  --seed 1397 \
  --output-dir artifacts/llm_selector_v2_role_only
