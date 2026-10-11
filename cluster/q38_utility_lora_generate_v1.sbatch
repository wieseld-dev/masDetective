#!/bin/bash
#SBATCH --job-name=q38lora-gen
#SBATCH --partition=rtx_pro_6000
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=10:00:00
#SBATCH --output=logs/q38lora-gen-%j.out
#SBATCH --error=logs/q38lora-gen-%j.err

set -euo pipefail

cd "$HOME/mas_rl_detective"

export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PY="$HOME/.venvs/masdetective-q38/bin/python"

nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader

"$PY" -u analysis/q38_utility_lora_generate_v1.py
