#!/bin/bash
#SBATCH --job-name=interf-v2
#SBATCH --partition=main
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --output=logs/interf-v2-%j.out
#SBATCH --error=logs/interf-v2-%j.err

set -euo pipefail

cd "$HOME/mas_rl_detective"

export PYTHONPATH="$HOME/mas_rl_detective:${PYTHONPATH:-}"
export PATH="$HOME/.conda/envs/masdetective/bin:$PATH"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1

PY="$HOME/.conda/envs/masdetective/bin/python"

: "${HMUX_ROOT:?set HMUX_ROOT}"

"$PY" -u scripts/interference_battery_v2.py \
  --root "$HMUX_ROOT" \
  --source-root artifacts/adaptive_v11_role_only_v1
