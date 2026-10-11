#!/bin/bash
#SBATCH --job-name=edit-tan
#SBATCH --partition=main
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/edit-tan-%j.out
#SBATCH --error=logs/edit-tan-%j.err

set -euo pipefail

cd "$HOME/mas_rl_detective"

export PYTHONPATH="$HOME/mas_rl_detective:${PYTHONPATH:-}"
export PATH="$HOME/.conda/envs/masdetective/bin:$PATH"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

"$HOME/.conda/envs/masdetective/bin/python" \
  -u scripts/edit_tangent_v0.py
