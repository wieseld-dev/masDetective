#!/bin/bash
#SBATCH --job-name=q38on-upd
#SBATCH --partition=rtx_pro_6000
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/q38on-upd-%j.out
#SBATCH --error=logs/q38on-upd-%j.err

set -euo pipefail

cd "$HOME/mas_rl_detective"

export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

: "${Q38_ROOT:?}"
: "${Q38_ROUND:?}"
: "${Q38_REF:?}"
: "${Q38_LR:?}"
: "${Q38_EPOCHS:?}"

PY="$HOME/.venvs/masdetective-q38/bin/python"

"$PY" -u \
  scripts/q38_adaptive_grpo_update_v1.py \
  --round "$Q38_ROUND" \
  --root "$Q38_ROOT" \
  --v11-adapter "$Q38_REF" \
  --model-name "Qwen/Qwen3.8-27B" \
  --epochs "$Q38_EPOCHS" \
  --lr "$Q38_LR" \
  --clip 0.20 \
  --kl-beta 0.05 \
  --max-seq-length 3072 \
  --no-4bit
