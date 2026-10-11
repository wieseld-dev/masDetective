#!/bin/bash
#SBATCH --job-name=av11-kdiag
#SBATCH --partition=main
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=00:05:00
#SBATCH --output=logs/av11-kdiag-%j.out
#SBATCH --error=logs/av11-kdiag-%j.err

set -euo pipefail

cd "$HOME/mas_rl_detective"

export PYTHONPATH="$HOME/mas_rl_detective:${PYTHONPATH:-}"
export PATH="$HOME/.conda/envs/masdetective/bin:$PATH"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1

PY="$HOME/.conda/envs/masdetective/bin/python"

"$PY" -u scripts/diagnose_av11_k_signal.py
