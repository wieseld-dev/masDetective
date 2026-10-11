#!/bin/bash
#SBATCH --job-name=kb-adv
#SBATCH --partition=main
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:30:00
#SBATCH --output=logs/kb-adv-%j.out
#SBATCH --error=logs/kb-adv-%j.err
set -euo pipefail
cd "$HOME/mas_rl_detective"
export PYTHONPATH="$HOME/mas_rl_detective:${PYTHONPATH:-}"
export PATH="$HOME/.conda/envs/masdetective/bin:$PATH"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
PY="$HOME/.conda/envs/masdetective/bin/python"
: "${KBOOT_ROOT:?set KBOOT_ROOT}"
: "${KBOOT_ROUND:?set KBOOT_ROUND}"
: "${KBOOT_SPLIT:?set KBOOT_SPLIT}"
"$PY" -u scripts/kboot_advance_role_only.py \
  --round "$KBOOT_ROUND" \
  --split "$KBOOT_SPLIT" \
  --root "$KBOOT_ROOT" \
  --belief "${KBOOT_BELIEF:-artifacts/adaptive_v11_active_v1/frozen_belief.joblib}" \
  --manifest "$KBOOT_ROOT/manifest_privileged.json"
