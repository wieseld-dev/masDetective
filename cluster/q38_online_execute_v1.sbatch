#!/bin/bash
#SBATCH --job-name=q38on-exec
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/q38on-exec-%j.out
#SBATCH --error=logs/q38on-exec-%j.err

set -euo pipefail

cd "$HOME/mas_rl_detective"

if [[ -f .env ]]; then
    set -a
    source .env
    set +a
fi

: "${OPENAI_API_KEY:?OPENAI_API_KEY missing}"
: "${Q38_ROOT:?}"
: "${Q38_ROUND:?}"
: "${Q38_SPLIT:?}"

export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

PY="$HOME/.conda/envs/masdetective/bin/python"

"$PY" -u \
  scripts/adaptive_v11_execute_round.py \
  --round "$Q38_ROUND" \
  --split "$Q38_SPLIT" \
  --root "$Q38_ROOT" \
  --manifest "$Q38_ROOT/manifest_privileged.json" \
  --workers 6
