#!/bin/bash
#SBATCH --job-name=gnr-v1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --output=logs/gnr-v1-%j.out
#SBATCH --error=logs/gnr-v1-%j.err

set -euo pipefail

cd "$HOME/mas_rl_detective"

PY="/home/wieseld/.conda/envs/masdetective/bin/python"

export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

"$PY" -u \
  analysis/generic_neural_ranker_v1.py
