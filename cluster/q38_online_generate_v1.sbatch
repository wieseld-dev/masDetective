#!/bin/bash
#SBATCH --job-name=q38on-gen
#SBATCH --partition=rtx_pro_6000
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/q38on-gen-%j.out
#SBATCH --error=logs/q38on-gen-%j.err

set -euo pipefail

cd "$HOME/mas_rl_detective"

export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

: "${Q38_ROOT:?}"
: "${Q38_ROUND:?}"
: "${Q38_SPLIT:?}"
: "${Q38_GROUP_SIZE:?}"
: "${Q38_REF:?}"

PY="$HOME/.venvs/masdetective-q38/bin/python"

ARGS=(
  --round "$Q38_ROUND"
  --split "$Q38_SPLIT"
  --group-size "$Q38_GROUP_SIZE"
  --root "$Q38_ROOT"
  --v11-adapter "$Q38_REF"
  --reference-adapter "$Q38_REF"
  --model-name "Qwen/Qwen3.8-27B"
  --max-seq-length 3072
  --max-new-tokens 128
  --temperature 0.85
  --top-p 0.95
  --no-4bit
)

if [ -n "${Q38_ADAPTER:-}" ]; then
    ARGS+=(--adapter "$Q38_ADAPTER")
fi

"$PY" -u \
  scripts/q38_adaptive_generate_round_v1.py \
  "${ARGS[@]}"
