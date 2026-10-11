#!/bin/bash
#SBATCH --job-name=av11-man
#SBATCH --partition=main
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=logs/av11-man-%j.out
#SBATCH --error=logs/av11-man-%j.err
set -euo pipefail
cd "$HOME/mas_rl_detective"
export PYTHONPATH="$HOME/mas_rl_detective:${PYTHONPATH:-}"
export PATH="$HOME/.conda/envs/masdetective/bin:$PATH"
PY="$HOME/.conda/envs/masdetective/bin/python"
"$PY" -u scripts/adaptive_v11_make_manifest.py
