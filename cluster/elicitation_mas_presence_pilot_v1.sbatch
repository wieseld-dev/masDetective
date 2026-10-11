#!/bin/bash
#SBATCH --job-name=elicit-mas
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=6
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=logs/elicit-mas-%j.out
#SBATCH --error=logs/elicit-mas-%j.err

set -euo pipefail

cd "$HOME/mas_rl_detective"

PY="/home/wieseld/.conda/envs/masdetective/bin/python"

export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

if [[ -f .env ]]; then
    set -a
    source .env
    set +a
fi

: "${OPENAI_API_KEY:?OPENAI_API_KEY missing}"

export MAS_MODEL="${MAS_MODEL:-gpt-4o-mini}"
export PILOT_API_WORKERS="${PILOT_API_WORKERS:-6}"

"$PY" -u \
  analysis/elicitation_mas_presence_pilot_v1.py
