#!/bin/bash
#SBATCH --job-name=bt-s5-exec
#SBATCH --partition=main
#SBATCH --cpus-per-task=5
#SBATCH --mem=16G
#SBATCH --time=20:00:00
#SBATCH --output=logs/bt-s5-exec-%j.out
#SBATCH --error=logs/bt-s5-exec-%j.err

set -euo pipefail

cd "$HOME/mas_rl_detective"

export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export PATH="$HOME/.conda/envs/masdetective/bin:$PATH"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1

if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

PY="$HOME/.conda/envs/masdetective/bin/python"

: "${S5_ROOT:?}"
: "${S5_ROUND:?}"

"$PY" -u analysis/bt_simple5_runtime_guard_v1.py \
  --root "$S5_ROOT" \
  --round "$S5_ROUND" \
  --phase post_select

"$PY" -u scripts/adaptive_v11_execute_round.py \
  --round "$S5_ROUND" \
  --split dev \
  --root "$S5_ROOT" \
  --manifest "$S5_ROOT/manifest_privileged.json" \
  --workers 5

"$PY" -u analysis/bt_simple5_runtime_guard_v1.py \
  --root "$S5_ROOT" \
  --round "$S5_ROUND" \
  --phase post_exec

echo "EXECUTE_PASS ROUND=$S5_ROUND"
