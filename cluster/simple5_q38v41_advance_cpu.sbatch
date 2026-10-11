#!/bin/bash
#SBATCH --job-name=s5-advance
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=20:00:00
#SBATCH --output=logs/s5-advance-%j.out
#SBATCH --error=logs/s5-advance-%j.err

set -euo pipefail

cd "$HOME/mas_rl_detective"

export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

: "${S5_ROOT:?}"
: "${S5_ROUND:?}"

PY="$HOME/.conda/envs/masdetective/bin/python"

"$PY" -u \
  scripts/kboot_advance_role_only.py \
  --round "$S5_ROUND" \
  --split dev \
  --root "$S5_ROOT" \
  --belief "$S5_ROOT/frozen_belief.joblib" \
  --manifest "$S5_ROOT/manifest_privileged.json"
