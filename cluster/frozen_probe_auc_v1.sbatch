#!/bin/bash
#SBATCH --job-name=probe-auc
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/probe-auc-%j.out
#SBATCH --error=logs/probe-auc-%j.err

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
export AUC_API_WORKERS="${AUC_API_WORKERS:-6}"

"$PY" -u \
  analysis/frozen_probe_auc_v1.py
