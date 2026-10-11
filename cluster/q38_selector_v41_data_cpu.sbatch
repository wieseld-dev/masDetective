#!/bin/bash
#SBATCH --job-name=q38s41-data
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=20:00:00
#SBATCH --output=logs/q38s41-data-%j.out
#SBATCH --error=logs/q38s41-data-%j.err

set -euo pipefail

cd "$HOME/mas_rl_detective"

export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

PY="$HOME/.conda/envs/masdetective/bin/python"

"$PY" -u \
  analysis/build_q38_selector_v41_dataset.py
