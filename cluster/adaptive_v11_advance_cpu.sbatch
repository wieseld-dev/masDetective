#!/bin/bash
#SBATCH --job-name=av11-adv
#SBATCH --partition=main
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/av11-adv-%j.out
#SBATCH --error=logs/av11-adv-%j.err
set -euo pipefail
cd "$HOME/mas_rl_detective"
export PYTHONPATH="$HOME/mas_rl_detective:${PYTHONPATH:-}"
export PATH="$HOME/.conda/envs/masdetective/bin:$PATH"
PY="$HOME/.conda/envs/masdetective/bin/python"
: "${AV11_ROUND:?set AV11_ROUND}"
"$PY" -u scripts/adaptive_v11_advance_round.py --round "$AV11_ROUND" --split "${AV11_SPLIT:-train}"
