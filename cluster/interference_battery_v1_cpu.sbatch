#!/bin/bash
#SBATCH --job-name=interf-v1
#SBATCH --partition=main
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/interf-v1-%j.out
#SBATCH --error=logs/interf-v1-%j.err

set -euo pipefail

cd "$HOME/mas_rl_detective"

export PYTHONPATH="$HOME/mas_rl_detective:${PYTHONPATH:-}"
export PATH="$HOME/.conda/envs/masdetective/bin:$PATH"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1

PY="$HOME/.conda/envs/masdetective/bin/python"

: "${HMUX_ROOT:?set HMUX_ROOT}"

"$PY" -u scripts/interference_battery_v1.py \
  --root "$HMUX_ROOT" \
  --source-root artifacts/adaptive_v11_role_only_v1
