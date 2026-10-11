#!/bin/bash
#SBATCH --job-name=eig-report
#SBATCH --partition=main
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/eig-report-%j.out
#SBATCH --error=logs/eig-report-%j.err
set -euo pipefail
cd "$HOME/mas_rl_detective"; export PYTHONPATH="$HOME/mas_rl_detective:${PYTHONPATH:-}"; export PATH="$HOME/.conda/envs/masdetective/bin:$PATH"; PY="$HOME/.conda/envs/masdetective/bin/python"
"$PY" -u scripts/av11_eig_report.py --root "$EIG_EXP" --rounds 6
