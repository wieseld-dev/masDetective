#!/bin/bash
#SBATCH --job-name=s5-q38sel
#SBATCH --partition=rtx_pro_6000
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=96G
#SBATCH --time=20:00:00
#SBATCH --output=logs/s5-q38sel-%j.out
#SBATCH --error=logs/s5-q38sel-%j.err

set -euo pipefail

cd "$HOME/mas_rl_detective"

export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PY="$HOME/.venvs/masdetective-q38/bin/python"

: "${S5_ROOT:?}"
: "${S5_ROUND:?}"

SELECTOR="artifacts/q38_selector_v41/best_adapter"

test -s "$SELECTOR/adapter_config.json"

echo "===== GPU ====="
nvidia-smi --query-gpu=name,memory.total \
  --format=csv,noheader

echo
echo "ROOT     = $S5_ROOT"
echo "ROUND    = $S5_ROUND"
echo "SELECTOR = $SELECTOR"

"$PY" -u \
  analysis/select_v14_candidates_q38_v41.py \
  --root "$S5_ROOT" \
  --round "$S5_ROUND" \
  --split dev \
  --adapter "$SELECTOR" \
  --max-length 1024 \
  --eval-perms 3
