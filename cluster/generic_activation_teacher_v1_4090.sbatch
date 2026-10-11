#!/bin/bash
#SBATCH --job-name=gen-act-v1
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --constraint=rtx_4090
#SBATCH --exclude=cs-4090-09
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:30:00
#SBATCH --output=logs/gen-act-v1-%j.out
#SBATCH --error=logs/gen-act-v1-%j.err

set -euo pipefail

cd "$HOME/mas_rl_detective"

export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export PATH="$HOME/.conda/envs/masdetective/bin:$PATH"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

"$HOME/.conda/envs/masdetective/bin/python" \
  -u analysis/generic_activation_teacher_v1.py
