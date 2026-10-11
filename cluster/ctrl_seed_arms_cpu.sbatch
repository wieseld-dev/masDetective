#!/bin/bash
#SBATCH --job-name=ctrl-seed
#SBATCH --partition=main
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=logs/ctrl-seed-%j.out
#SBATCH --error=logs/ctrl-seed-%j.err
set -euo pipefail
cd "$HOME/mas_rl_detective"
export PYTHONPATH="$HOME/mas_rl_detective:${PYTHONPATH:-}"
export PATH="$HOME/.conda/envs/masdetective/bin:$PATH"
PY="$HOME/.conda/envs/masdetective/bin/python"
: "${CTRL_EXPERIMENT_ROOT:?set CTRL_EXPERIMENT_ROOT}"
"$PY" -u scripts/av11_controller_seed_arms.py \
  --common-root "$CTRL_EXPERIMENT_ROOT/common" \
  --static-root "$CTRL_EXPERIMENT_ROOT/static" \
  --adaptive-root "$CTRL_EXPERIMENT_ROOT/adaptive" \
  --split dev
