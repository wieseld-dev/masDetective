#!/bin/bash
#SBATCH --job-name=act-rwd
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --constraint=rtx_4090
#SBATCH --exclude=cs-4090-09
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=logs/act-rwd-%j.out
#SBATCH --error=logs/act-rwd-%j.err

set -euo pipefail

cd "$HOME/mas_rl_detective"

export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export PATH="$HOME/.conda/envs/masdetective/bin:$PATH"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export ACT_MODEL="Qwen/Qwen2.5-1.5B-Instruct"
export ACT_BATCH_PAIRS=8
export ACT_MAX_LEN=512

PY="$HOME/.conda/envs/masdetective/bin/python"

nvidia-smi

"$PY" -u analysis/activation_reward_layer_sweep_v1.py
