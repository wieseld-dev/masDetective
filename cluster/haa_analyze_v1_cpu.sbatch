#!/bin/bash
#SBATCH --job-name=haa-analyze
#SBATCH --partition=main
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=logs/haa-analyze-%j.out
#SBATCH --error=logs/haa-analyze-%j.err

set -euo pipefail

cd "$HOME/mas_rl_detective"

export PYTHONPATH="$HOME/mas_rl_detective:${PYTHONPATH:-}"
export PATH="$HOME/.conda/envs/masdetective/bin:$PATH"
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false

"$HOME/.conda/envs/masdetective/bin/python" \
  -u scripts/haa_prospective_analyze_v1.py
