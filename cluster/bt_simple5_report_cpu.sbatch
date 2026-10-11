#!/bin/bash
#SBATCH --job-name=bt-s5-report
#SBATCH --partition=main
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=20:00:00
#SBATCH --output=logs/bt-s5-report-%j.out
#SBATCH --error=logs/bt-s5-report-%j.err

set -euo pipefail

cd "$HOME/mas_rl_detective"

export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export PATH="$HOME/.conda/envs/masdetective/bin:$PATH"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1

PY="$HOME/.conda/envs/masdetective/bin/python"

: "${S5_ROOT:?}"

"$PY" -u analysis/bt_simple5_runtime_guard_v1.py \
  --root "$S5_ROOT" \
  --round 8 \
  --phase post_advance

"$PY" -u analysis/bt_simple5_report_v1.py \
  --root "$S5_ROOT" \
  --rounds 8
