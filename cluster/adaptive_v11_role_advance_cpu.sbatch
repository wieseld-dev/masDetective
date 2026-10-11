#!/bin/bash
#SBATCH --job-name=av11-radv
#SBATCH --partition=main
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=logs/av11-radv-%j.out
#SBATCH --error=logs/av11-radv-%j.err
set -euo pipefail
cd "$HOME/mas_rl_detective"
export PYTHONPATH="$HOME/mas_rl_detective:${PYTHONPATH:-}"
export PATH="$HOME/.conda/envs/masdetective/bin:$PATH"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
PY="$HOME/.conda/envs/masdetective/bin/python"
: "${AV11_ROUND:?set AV11_ROUND}"
ROOT="${AV11_ROLE_ROOT:-artifacts/adaptive_v11_role_only_v1}"
"$PY" -u scripts/adaptive_v11_role_advance_round.py \
  --round "$AV11_ROUND" --split "${AV11_SPLIT:-train}" --root "$ROOT"
