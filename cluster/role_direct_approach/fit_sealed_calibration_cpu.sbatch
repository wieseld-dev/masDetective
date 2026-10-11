#!/bin/bash
#SBATCH --job-name=rd-seal-cal
#SBATCH --partition=main
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=20:00:00
#SBATCH --output=logs/rd-seal-cal-%j.out
#SBATCH --error=logs/rd-seal-cal-%j.err

set -euo pipefail

PROJECT="$HOME/mas_rl_detective"
PY="$HOME/.conda/envs/masdetective/bin/python"

cd "$PROJECT"

export PYTHONPATH="$PROJECT:${PYTHONPATH:-}"

"$PY" -u \
  analysis/role_direct_approach/fit_sealed_calibration_v1.py
