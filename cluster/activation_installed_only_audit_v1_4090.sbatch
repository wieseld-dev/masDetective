#!/bin/bash
#SBATCH --job-name=act-route-v1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx_4090:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=20:00:00
#SBATCH --output=logs/act-route-v1-%j.out
#SBATCH --error=logs/act-route-v1-%j.err

set -euo pipefail

cd "$HOME/mas_rl_detective"

export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1

PY="$HOME/.conda/envs/masdetective/bin/python"

"$PY" -u \
  analysis/activation_installed_only_audit_v1.py
