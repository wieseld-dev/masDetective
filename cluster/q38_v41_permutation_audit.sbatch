#!/bin/bash
#SBATCH --job-name=q38-v41-audit
#SBATCH --partition=rtx_pro_6000
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=96G
#SBATCH --time=20:00:00
#SBATCH --output=logs/q38-v41-audit-%j.out
#SBATCH --error=logs/q38-v41-audit-%j.err

set -euo pipefail

cd "$HOME/mas_rl_detective"

export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PY="$HOME/.venvs/masdetective-q38/bin/python"

nvidia-smi \
  --query-gpu=name,memory.total,memory.free \
  --format=csv,noheader

"$PY" -u \
  analysis/q38_v41_permutation_stability_audit.py
