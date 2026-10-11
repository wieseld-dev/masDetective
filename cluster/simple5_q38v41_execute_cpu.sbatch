#!/bin/bash
#SBATCH --job-name=s5-exec
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --time=20:00:00
#SBATCH --output=logs/s5-exec-%j.out
#SBATCH --error=logs/s5-exec-%j.err

set -euo pipefail

cd "$HOME/mas_rl_detective"

export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

if [ -f "$HOME/mas_rl_detective/.env" ]; then
    set -a
    source "$HOME/mas_rl_detective/.env"
    set +a
fi

: "${OPENAI_API_KEY:?OPENAI_API_KEY missing}"
: "${S5_ROOT:?}"
: "${S5_ROUND:?}"

PY="$HOME/.conda/envs/masdetective/bin/python"

"$PY" -u \
  scripts/adaptive_v11_execute_round.py \
  --round "$S5_ROUND" \
  --split dev \
  --root "$S5_ROOT" \
  --manifest "$S5_ROOT/manifest_privileged.json" \
  --workers 5
