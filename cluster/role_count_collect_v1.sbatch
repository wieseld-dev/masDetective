#!/bin/bash
#SBATCH --job-name=rc-data
#SBATCH --partition=main
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=10:00:00
#SBATCH --output=logs/rc-data-%j.out
#SBATCH --error=logs/rc-data-%j.err

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

"$PY" -m py_compile scripts/role_count_spectral_v1.py

"$PY" -u scripts/role_count_spectral_v1.py \
    --collect \
    --n-probes "${ROLECOUNT_N_PROBES:-40}" \
    --concurrency "${ROLECOUNT_CONCURRENCY:-10}"
