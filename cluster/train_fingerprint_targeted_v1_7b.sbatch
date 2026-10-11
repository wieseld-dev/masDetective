#!/bin/bash
#SBATCH --job-name=fp-target-v1
#SBATCH --partition=gpu
#SBATCH --constraint=rtx_4090
#SBATCH --gres=gpu:1
#SBATCH --exclude=cs-4090-09
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --output=logs/fp-target-v1-%j.out
#SBATCH --error=logs/fp-target-v1-%j.err

set -euo pipefail

cd "$HOME/mas_rl_detective"

PY="/home/wieseld/.conda/envs/masdetective/bin/python"

export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

"$PY" -u \
  analysis/train_fingerprint_targeted_v1.py \
  --v13-discovery \
    artifacts/v13_probe_disambiguation.jsonl \
  --v11-discovery \
    artifacts/fingerprint_targeted_v1/empty_v11.jsonl \
  --output-dir \
    artifacts/fingerprint_targeted_v1_7b \
  --base-model \
    Qwen/Qwen2.5-7B-Instruct \
  --seed 1331 \
  --lora-r 32 \
  --lora-alpha 64 \
  --lora-dropout 0.05 \
  --max-length 1024 \
  --batch-size 1 \
  --pair-batch-size 1 \
  --eval-batch-size 4 \
  --eval-pair-batch-size 2 \
  --grad-accum 8 \
  --epochs 6 \
  --lr 5e-5 \
  --neg-ratio 2.0 \
  --self-target-neg-weight 1.0 \
  --hardneg-weight 1.0 \
  --pair-margin 1.0 \
  --pair-loss-weight 0.5 \
  --early-stop-patience 2 \
  --min-delta 1e-4
