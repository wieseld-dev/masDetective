#!/bin/bash
#SBATCH --job-name=v9-holdout-all
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --output=logs/v9-holdout-all-%j.out

set -euo pipefail

cd "$HOME/mas_rl_detective"
mkdir -p logs artifacts

# Use the existing project environment directly; no interactive conda activation needed.
PY="$HOME/.conda/envs/masdetective/bin/python"
if [[ ! -x "$PY" ]]; then
  echo "Missing Python environment: $PY" >&2
  exit 2
fi

set -a
source .env
set +a

export PYTHONPATH="$PWD"

"$PY" -u scripts/evaluate_v9full_exhaustive_holdout.py --execute
