#!/bin/bash
#SBATCH --job-name=v14a50-cmp
#SBATCH --partition=main
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/v14a50-cmp-%j.out
#SBATCH --error=logs/v14a50-cmp-%j.err

set -euo pipefail

cd "$HOME/mas_rl_detective"

export PYTHONPATH="$HOME/mas_rl_detective:${PYTHONPATH:-}"
export PATH="$HOME/.conda/envs/masdetective/bin:$PATH"

"$HOME/.conda/envs/masdetective/bin/python" \
    -u scripts/v14a_compare_g50_v2.py
