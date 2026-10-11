#!/bin/bash
#SBATCH --job-name=bt-s5-preflight
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx_4090:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=20:00:00
#SBATCH --output=logs/bt-s5-preflight-%j.out
#SBATCH --error=logs/bt-s5-preflight-%j.err

set -euo pipefail

cd "$HOME/mas_rl_detective"

export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1

PY="$HOME/.conda/envs/masdetective/bin/python"

nvidia-smi \
  --query-gpu=name,memory.total,memory.free \
  --format=csv,noheader

"$PY" -u \
  analysis/belief_targeted_simple5_preflight_v1.py
