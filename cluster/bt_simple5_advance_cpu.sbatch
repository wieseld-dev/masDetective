#!/bin/bash
#SBATCH --job-name=bt-s5-adv
#SBATCH --partition=main
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=20:00:00
#SBATCH --output=logs/bt-s5-adv-%j.out
#SBATCH --error=logs/bt-s5-adv-%j.err

set -euo pipefail

cd "$HOME/mas_rl_detective"

export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export PATH="$HOME/.conda/envs/masdetective/bin:$PATH"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1

PY="$HOME/.conda/envs/masdetective/bin/python"

: "${S5_ROOT:?}"
: "${S5_ROUND:?}"

"$PY" -u analysis/bt_simple5_runtime_guard_v1.py \
  --root "$S5_ROOT" \
  --round "$S5_ROUND" \
  --phase post_score

"$PY" -u scripts/kboot_advance_role_only.py \
  --round "$S5_ROUND" \
  --split dev \
  --root "$S5_ROOT" \
  --belief "$S5_ROOT/frozen_belief.joblib" \
  --manifest "$S5_ROOT/manifest_privileged.json"

"$PY" -u analysis/bt_simple5_runtime_guard_v1.py \
  --root "$S5_ROOT" \
  --round "$S5_ROUND" \
  --phase post_advance

echo "ADVANCE_PASS ROUND=$S5_ROUND"
