#!/bin/bash
#SBATCH --job-name=selv3-data
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=20:00:00
#SBATCH --output=logs/selv3-data-%j.out
#SBATCH --error=logs/selv3-data-%j.err

set -euo pipefail

cd "$HOME/mas_rl_detective"

export PYTHONPATH="$HOME/mas_rl_detective:${PYTHONPATH:-}"
export PATH="$HOME/.conda/envs/masdetective/bin:$PATH"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1

PY="$HOME/.conda/envs/masdetective/bin/python"

"$PY" -u \
  analysis/build_llm_selector_v3_auc_dataset.py
