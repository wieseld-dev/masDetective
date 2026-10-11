#!/bin/bash
#SBATCH --job-name=q38on-report
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/q38on-report-%j.out
#SBATCH --error=logs/q38on-report-%j.err

set -euo pipefail

cd "$HOME/mas_rl_detective"

export PYTHONPATH="$PWD:${PYTHONPATH:-}"

PY="$HOME/.conda/envs/masdetective/bin/python"

"$PY" -u \
  analysis/q38_online_report_v1.py
