#!/bin/bash
#SBATCH --job-name=hmux-v0
#SBATCH --partition=main
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/hmux-v0-%j.out
#SBATCH --error=logs/hmux-v0-%j.err
set -euo pipefail

cd "$HOME/mas_rl_detective"

export PYTHONPATH="$HOME/mas_rl_detective:${PYTHONPATH:-}"
export PATH="$HOME/.conda/envs/masdetective/bin:$PATH"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1

PY="$HOME/.conda/envs/masdetective/bin/python"

: "${HMUX_STAGE:?set HMUX_STAGE}"
: "${HMUX_ROOT:?set HMUX_ROOT}"

"$PY" -u scripts/hadamard_mux_v0.py \
  --stage "$HMUX_STAGE" \
  --root "$HMUX_ROOT"
