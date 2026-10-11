#!/bin/bash
#SBATCH --job-name=fp-v3
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=10:00:00
#SBATCH --output=logs/fp-v3-%j.out
#SBATCH --error=logs/fp-v3-%j.err

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

export PROFILE_MODEL="${PROFILE_MODEL:-gpt-4o-mini}"
export PROFILE_WORKERS="${PROFILE_WORKERS:-6}"

"$PY" -u \
  analysis/explainable_fingerprint_v3.py
