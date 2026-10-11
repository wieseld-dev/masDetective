#!/bin/bash
#SBATCH --job-name=av11-exec
#SBATCH --partition=main
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=logs/av11-exec-%j.out
#SBATCH --error=logs/av11-exec-%j.err
set -euo pipefail
cd "$HOME/mas_rl_detective"
export PYTHONPATH="$HOME/mas_rl_detective:${PYTHONPATH:-}"
export PATH="$HOME/.conda/envs/masdetective/bin:$PATH"
PY="$HOME/.conda/envs/masdetective/bin/python"
: "${AV11_ROUND:?set AV11_ROUND}"
"$PY" -u scripts/adaptive_v11_execute_round.py --round "$AV11_ROUND" --split "${AV11_SPLIT:-train}"
