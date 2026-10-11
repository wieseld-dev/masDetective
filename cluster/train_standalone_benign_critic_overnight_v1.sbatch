#!/bin/bash
#SBATCH --job-name=ben7b-night
#SBATCH --partition=gpu
#SBATCH --constraint=rtx_4090
#SBATCH --gres=gpu:1
#SBATCH --exclude=cs-4090-09
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --output=logs/ben7b-night-%j.out
#SBATCH --error=logs/ben7b-night-%j.err

set -euo pipefail

cd "$HOME/mas_rl_detective"

PY="/home/wieseld/.conda/envs/masdetective/bin/python"

export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${SLURM_GPU_INDEX:-0}"

"$PY" -u \
  analysis/train_standalone_benign_critic_overnight_v1.py
