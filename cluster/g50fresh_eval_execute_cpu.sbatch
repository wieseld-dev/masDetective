#!/bin/bash
#SBATCH --job-name=g50f-exec
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/g50f-exec-%j.out
#SBATCH --error=logs/g50f-exec-%j.err

set -euo pipefail

cd "$HOME/mas_rl_detective"

export PYTHONPATH="$HOME/mas_rl_detective:${PYTHONPATH:-}"
export PATH="$HOME/.conda/envs/masdetective/bin:$PATH"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

if [[ -f .env ]]; then
    set -a
    source .env
    set +a
fi

: "${OPENAI_API_KEY:?OPENAI_API_KEY missing after loading .env}"
: "${G50_EVAL_ROOT:?}"
: "${G50_EVAL_ROUND:?}"

PY="$HOME/.conda/envs/masdetective/bin/python"

"$PY" -u scripts/adaptive_v11_execute_round.py \
    --round "$G50_EVAL_ROUND" \
    --split dev \
    --root "$G50_EVAL_ROOT" \
    --manifest "$G50_EVAL_ROOT/manifest_privileged.json" \
    --workers 2 \
    --retries 3
