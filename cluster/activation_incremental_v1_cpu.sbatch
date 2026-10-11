#!/bin/bash
#SBATCH --job-name=act-inc
#SBATCH --partition=main
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/act-inc-%j.out
#SBATCH --error=logs/act-inc-%j.err

set -euo pipefail

cd "$HOME/mas_rl_detective"

export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export PATH="$HOME/.conda/envs/masdetective/bin:$PATH"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1

"$HOME/.conda/envs/masdetective/bin/python" \
  -u analysis/activation_incremental_v1.py
