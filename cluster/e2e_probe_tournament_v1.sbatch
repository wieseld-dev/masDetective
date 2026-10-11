#!/bin/bash
#SBATCH --job-name=e2e-probes
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/e2e-probes-%j.out
#SBATCH --error=logs/e2e-probes-%j.err

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
export E2E_TOURNAMENT_WORKERS="${E2E_TOURNAMENT_WORKERS:-8}"

"$PY" -u \
  analysis/e2e_probe_tournament_v1.py
