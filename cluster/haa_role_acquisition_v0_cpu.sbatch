#!/bin/bash
#SBATCH --job-name=haa-role-v0
#SBATCH --partition=main
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=logs/haa-role-v0-%j.out
#SBATCH --error=logs/haa-role-v0-%j.err

set -euo pipefail

cd "$HOME/mas_rl_detective"

export PYTHONPATH="$HOME/mas_rl_detective:${PYTHONPATH:-}"
export PATH="$HOME/.conda/envs/masdetective/bin:$PATH"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

"$HOME/.conda/envs/masdetective/bin/python" \
  -u scripts/haa_role_acquisition_v0.py
