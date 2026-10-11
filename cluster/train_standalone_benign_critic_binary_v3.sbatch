#!/bin/bash
#SBATCH --job-name=benbin-v3
#SBATCH --partition=gpu
#SBATCH --constraint=rtx_4090
#SBATCH --gres=gpu:1
#SBATCH --exclude=cs-4090-09
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --output=logs/benbin-v3-%j.out
#SBATCH --error=logs/benbin-v3-%j.err

set -euo pipefail

cd "$HOME/mas_rl_detective"

PY="/home/wieseld/.conda/envs/masdetective/bin/python"

export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${SLURM_GPU_INDEX:-0}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

"$PY" -u \
  analysis/train_standalone_benign_critic_binary_v3.py
