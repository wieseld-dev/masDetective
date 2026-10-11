#!/bin/bash
#SBATCH --job-name=la-gen
#SBATCH --partition=main
#SBATCH --gpus=1
#SBATCH --constraint=rtx_4090
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=logs/la-gen-%j.out
#SBATCH --error=logs/la-gen-%j.err

set -euo pipefail

cd "$HOME/mas_rl_detective"

export PYTHONPATH="$HOME/mas_rl_detective:${PYTHONPATH:-}"
export PATH="$HOME/.conda/envs/masdetective/bin:$PATH"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

"$HOME/.conda/envs/masdetective/bin/python" \
  -u scripts/local_ambiguity_generate_v1.py
