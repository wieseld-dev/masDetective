#!/bin/bash
#SBATCH --job-name=v14a-native
#SBATCH --partition=gpu
#SBATCH --constraint=rtx_4090
#SBATCH --gres=gpu:1
#SBATCH --exclude=cs-4090-09
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/v14a-native-%j.out
#SBATCH --error=logs/v14a-native-%j.err

set -euo pipefail

cd "$HOME/mas_rl_detective"

PY=/home/wieseld/.conda/envs/masdetective/bin/python

export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false

"$PY" -u \
  analysis/v14_adaptive_native_smoke_v1.py
