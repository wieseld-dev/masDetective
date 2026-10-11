#!/bin/bash
#SBATCH --job-name=rd-cf-report
#SBATCH --partition=main
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=20:00:00
#SBATCH --output=logs/rd-cf-report-%j.out
#SBATCH --error=logs/rd-cf-report-%j.err

set -euo pipefail

PROJECT="$HOME/mas_rl_detective"
PY="$HOME/.conda/envs/masdetective/bin/python"

cd "$PROJECT"

export PYTHONPATH="$PROJECT:${PYTHONPATH:-}"

"$PY" -u \
  analysis/role_direct_approach/report_matched_cf_v1.py
