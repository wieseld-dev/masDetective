#!/bin/bash
#SBATCH --job-name=s5-adv
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=20:00:00
#SBATCH --output=logs/s5-adv-%j.out
#SBATCH --error=logs/s5-adv-%j.err

set -euo pipefail

cd "$HOME/mas_rl_detective"

export PYTHONPATH="$HOME/mas_rl_detective:${PYTHONPATH:-}"
export PATH="$HOME/.conda/envs/masdetective/bin:$PATH"
export PYTHONUNBUFFERED=1

PY="$HOME/.conda/envs/masdetective/bin/python"

: "${S5_ROOT:?}"
: "${S5_ROUND:?}"

"$PY" scripts/adaptive_v11_role_advance_round.py \
  --round "$S5_ROUND" \
  --split dev \
  --root "$S5_ROOT" \
  --belief "$S5_ROOT/frozen_belief.joblib" \
  --manifest "$S5_ROOT/manifest_privileged.json"
