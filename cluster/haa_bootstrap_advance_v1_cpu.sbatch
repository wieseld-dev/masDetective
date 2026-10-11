#!/bin/bash
#SBATCH --job-name=haa-boot
#SBATCH --partition=main
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/haa-boot-%j.out
#SBATCH --error=logs/haa-boot-%j.err

set -euo pipefail

cd "$HOME/mas_rl_detective"

export PYTHONPATH="$HOME/mas_rl_detective:${PYTHONPATH:-}"
export PATH="$HOME/.conda/envs/masdetective/bin:$PATH"
export PYTHONNOUSERSITE=1

"$HOME/.conda/envs/masdetective/bin/python" \
  -u scripts/haa_bootstrap_advance_v1.py
