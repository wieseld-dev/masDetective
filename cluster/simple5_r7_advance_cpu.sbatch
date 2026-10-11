#!/bin/bash
#SBATCH --job-name=v14a50-adv
#SBATCH --partition=main
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=20:00:00
#SBATCH --output=logs/v14a50-adv-%j.out
#SBATCH --error=logs/v14a50-adv-%j.err

set -euo pipefail

cd "$HOME/mas_rl_detective"

export PYTHONPATH="$HOME/mas_rl_detective:${PYTHONPATH:-}"
export PATH="$HOME/.conda/envs/masdetective/bin:$PATH"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1

PY="$HOME/.conda/envs/masdetective/bin/python"

: "${V14A_ROOT:?}"
: "${V14A_ROUND:?}"
: "${V14A_SPLIT:?}"

SCRIPT="scripts/v14a_role_advance_g50_v2.py"

ARGS=(
  --round "$V14A_ROUND"
  --split "$V14A_SPLIT"
  --root "$V14A_ROOT"
)

HELP="$("$PY" "$SCRIPT" --help 2>&1)"

if grep -q -- '--manifest' <<<"$HELP"; then
    ARGS+=(--manifest "$V14A_ROOT/manifest_privileged.json")
fi

if grep -q -- '--belief' <<<"$HELP"; then
    ARGS+=(--belief "$V14A_ROOT/frozen_belief.joblib")
fi

"$PY" -u "$SCRIPT" "${ARGS[@]}"
