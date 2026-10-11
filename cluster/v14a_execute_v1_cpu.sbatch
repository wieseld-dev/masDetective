#!/bin/bash
#SBATCH --job-name=v14a-exec
#SBATCH --partition=main
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --output=logs/v14a-exec-%j.out
#SBATCH --error=logs/v14a-exec-%j.err

set -euo pipefail

cd "$HOME/mas_rl_detective"

export PYTHONPATH="$HOME/mas_rl_detective:${PYTHONPATH:-}"
export PATH="$HOME/.conda/envs/masdetective/bin:$PATH"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1

if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

PY="$HOME/.conda/envs/masdetective/bin/python"

: "${V14A_ROOT:?}"
: "${V14A_ROUND:?}"
: "${V14A_SPLIT:?}"

"$PY" -u scripts/adaptive_v11_execute_round.py \
    --round "$V14A_ROUND" \
    --split "$V14A_SPLIT" \
    --root "$V14A_ROOT" \
    --manifest "$V14A_ROOT/manifest_privileged.json" \
    --workers "${AV11_API_WORKERS:-6}"
