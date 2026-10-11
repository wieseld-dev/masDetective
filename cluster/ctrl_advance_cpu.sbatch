#!/bin/bash
#SBATCH --job-name=ctrl-adv
#SBATCH --partition=main
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:30:00
#SBATCH --output=logs/ctrl-adv-%j.out
#SBATCH --error=logs/ctrl-adv-%j.err
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
"$PY" -u scripts/kboot_advance_role_only.py \
  --round "$CTRL_ROUND" \
  --split "$CTRL_SPLIT" \
  --root "$CTRL_ROOT" \
  --belief "${CTRL_BELIEF:-artifacts/adaptive_v11_active_v1/frozen_belief.joblib}" \
  --manifest "$CTRL_ROOT/manifest_privileged.json"
