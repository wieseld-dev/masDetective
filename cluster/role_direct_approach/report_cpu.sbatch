#!/bin/bash
#SBATCH --job-name=rd-report
#SBATCH --partition=main
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=20:00:00
#SBATCH --output=logs/rd-report-%j.out
#SBATCH --error=logs/rd-report-%j.err

set -euo pipefail

cd "$HOME/mas_rl_detective"

export PYTHONPATH="$PWD:${PYTHONPATH:-}"

PY="$HOME/.conda/envs/masdetective/bin/python"

ROOT="${RD_ROOT:-artifacts/role_direct_approach_v1}"

"$PY" \
  analysis/role_direct_approach/report_role_direct_v1.py \
  --root "$ROOT" \
  --split dev
