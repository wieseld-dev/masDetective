#!/bin/bash
#SBATCH --job-name=kb-fitk
#SBATCH --partition=main
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=logs/kb-fitk-%j.out
#SBATCH --error=logs/kb-fitk-%j.err
set -euo pipefail
cd "$HOME/mas_rl_detective"
export PYTHONPATH="$HOME/mas_rl_detective:${PYTHONPATH:-}"
export PATH="$HOME/.conda/envs/masdetective/bin:$PATH"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
PY="$HOME/.conda/envs/masdetective/bin/python"
: "${KBOOT_ROOT:?set KBOOT_ROOT}"
"$PY" -u scripts/kboot_fit_k_head.py \
  --root "$KBOOT_ROOT" \
  --belief "${KBOOT_BELIEF:-artifacts/adaptive_v11_active_v1/frozen_belief.joblib}" \
  --manifest "$KBOOT_ROOT/manifest_privileged.json" \
  --rounds "${KBOOT_ROUNDS:-6}" \
  --output "$KBOOT_ROOT/k_head.joblib"
