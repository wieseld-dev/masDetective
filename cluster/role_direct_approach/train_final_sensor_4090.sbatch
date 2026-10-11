#!/bin/bash
#SBATCH --job-name=rd-final-sens
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx_4090:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=20:00:00
#SBATCH --output=logs/rd-final-sens-%j.out
#SBATCH --error=logs/rd-final-sens-%j.err

set -euo pipefail

PROJECT="$HOME/mas_rl_detective"
PY="$HOME/.conda/envs/masdetective/bin/python"

cd "$PROJECT"

export PYTHONPATH="$PROJECT:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

"$PY" -u \
  analysis/role_direct_approach/train_role_direct_sensor_final_v1.py
