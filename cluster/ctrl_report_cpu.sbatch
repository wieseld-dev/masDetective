#!/bin/bash
#SBATCH --job-name=ctrl-report
#SBATCH --partition=main
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/ctrl-report-%j.out
#SBATCH --error=logs/ctrl-report-%j.err
set -euo pipefail
cd "$HOME/mas_rl_detective"
export PYTHONPATH="$HOME/mas_rl_detective:${PYTHONPATH:-}"
export PATH="$HOME/.conda/envs/masdetective/bin:$PATH"
PY="$HOME/.conda/envs/masdetective/bin/python"
: "${CTRL_EXPERIMENT_ROOT:?set CTRL_EXPERIMENT_ROOT}"
"$PY" -u scripts/av11_controller_report.py \
  --static-root "$CTRL_EXPERIMENT_ROOT/static" \
  --adaptive-root "$CTRL_EXPERIMENT_ROOT/adaptive" \
  --split dev \
  --rounds 6 \
  --output-root "$CTRL_EXPERIMENT_ROOT/report"
