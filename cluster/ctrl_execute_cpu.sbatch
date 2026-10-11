#!/bin/bash
#SBATCH --job-name=ctrl-exec
#SBATCH --partition=main
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=logs/ctrl-exec-%j.out
#SBATCH --error=logs/ctrl-exec-%j.err
set -euo pipefail
cd "$HOME/mas_rl_detective"
export PYTHONPATH="$HOME/mas_rl_detective:${PYTHONPATH:-}"
export PATH="$HOME/.conda/envs/masdetective/bin:$PATH"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
PY="$HOME/.conda/envs/masdetective/bin/python"
: "${CTRL_ROOT:?set CTRL_ROOT}"
: "${CTRL_ROUND:?set CTRL_ROUND}"
: "${CTRL_SPLIT:?set CTRL_SPLIT}"
"$PY" -u scripts/adaptive_v11_execute_round.py \
  --round "$CTRL_ROUND" \
  --split "$CTRL_SPLIT" \
  --root "$CTRL_ROOT" \
  --manifest "$CTRL_ROOT/manifest_privileged.json" \
  --workers "${CTRL_WORKERS:-4}" \
  --retries 3
