#!/bin/bash
#SBATCH --job-name=q38on-a50
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/q38on-a50-%j.out
#SBATCH --error=logs/q38on-a50-%j.err

set -euo pipefail

cd "$HOME/mas_rl_detective"

export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

: "${Q38_ROOT:?}"
: "${Q38_ROUND:?}"
: "${Q38_SPLIT:?}"

PY="$HOME/.conda/envs/masdetective/bin/python"

"$PY" -u \
  scripts/v14a_role_advance_g50_v2.py \
  --round "$Q38_ROUND" \
  --split "$Q38_SPLIT" \
  --root "$Q38_ROOT" \
  --belief "$Q38_ROOT/frozen_belief.joblib" \
  --manifest "$Q38_ROOT/manifest_privileged.json"
