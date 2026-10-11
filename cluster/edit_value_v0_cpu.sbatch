#!/bin/bash
#SBATCH --job-name=edit-val
#SBATCH --partition=main
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=logs/edit-val-%j.out
#SBATCH --error=logs/edit-val-%j.err

set -euo pipefail

cd "$HOME/mas_rl_detective"

export PYTHONPATH="$HOME/mas_rl_detective:${PYTHONPATH:-}"
export PATH="$HOME/.conda/envs/masdetective/bin:$PATH"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1

"$HOME/.conda/envs/masdetective/bin/python" \
  -u scripts/edit_value_v0.py
