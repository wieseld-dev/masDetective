#!/bin/bash
#SBATCH --job-name=av11-analysis
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=logs/av11-robust-analysis-%j.out
#SBATCH --error=logs/av11-robust-analysis-%j.err

set -euo pipefail

cd "$HOME/mas_rl_detective"

"$HOME/.conda/envs/masdetective/bin/python" \
  scripts/analyze_av11_robust_3seed.py
