#!/bin/bash
#SBATCH --job-name=eig-seed
#SBATCH --partition=main
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=logs/eig-seed-%j.out
#SBATCH --error=logs/eig-seed-%j.err
set -euo pipefail
cd "$HOME/mas_rl_detective"; export PYTHONPATH="$HOME/mas_rl_detective:${PYTHONPATH:-}"; PY="$HOME/.conda/envs/masdetective/bin/python"
"$PY" -u scripts/av11_seed_three_arms.py --common-root "$EIG_EXP/common" --arms "$EIG_EXP/static" "$EIG_EXP/maxent" "$EIG_EXP/utility" --split dev
