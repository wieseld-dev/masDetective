#!/bin/bash
#SBATCH --job-name=cur-exec
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=0-04:00:00
#SBATCH --output=logs/cur-exec-%j.out
#SBATCH --error=logs/cur-exec-%j.err
set -euo pipefail
cd "$HOME/mas_rl_detective"
export PYTHONPATH="$HOME/mas_rl_detective:${PYTHONPATH:-}"
export PATH="$HOME/.conda/envs/masdetective/bin:$PATH"
if [[ -f .env ]]; then set -a; source .env; set +a; fi
: "${OPENAI_API_KEY:?OPENAI_API_KEY missing (.env not loaded?)}"
PY="$HOME/.conda/envs/masdetective/bin/python"
"$PY" -u scripts/adaptive_v11_execute_round.py \
  --round "$CURR_ROUND" --split "$CURR_SPLIT" --root "$CURR_ROOT" --manifest "$CURR_MANIFEST" \
  --model "${CURR_TARGET_MODEL:-gpt-4o-mini}" --workers "${CURR_WORKERS:-6}" --retries "${CURR_API_RETRIES:-3}"
