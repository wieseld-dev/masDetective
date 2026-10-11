#!/bin/bash
#SBATCH --job-name=hypergen-v2
#SBATCH --partition=gpu
#SBATCH --constraint=rtx_4090
#SBATCH --gres=gpu:1
#SBATCH --exclude=cs-4090-09
#SBATCH --mem=24G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/hypergen-v2-%j.out
#SBATCH --error=logs/hypergen-v2-%j.err

set -euo pipefail

cd "$HOME/mas_rl_detective"

PY="/home/wieseld/.conda/envs/masdetective/bin/python"

export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "===== NODE / GPU ====="
hostname
nvidia-smi

"$PY" -u analysis/train_hypersensor_general_v2.py
