#!/bin/bash
#SBATCH --job-name=probe-route
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=logs/probe-route-%j.out
#SBATCH --error=logs/probe-route-%j.err

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
export ROUTER_TOURNAMENT_WORKERS="${ROUTER_TOURNAMENT_WORKERS:-10}"

"$PY" -u \
  analysis/probe_router_tournament_v1.py
