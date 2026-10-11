#!/bin/bash
#SBATCH --job-name=v10-elite-diag
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --output=logs/v10-elite-diag-%j.out

set -euo pipefail
cd "$HOME/mas_rl_detective"
mkdir -p logs artifacts
PY="$HOME/.conda/envs/masdetective/bin/python"
if [[ ! -x "$PY" ]]; then
  echo "Missing Python environment: $PY" >&2
  exit 2
fi
if [[ -f .env ]]; then set -a; source .env; set +a; fi
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

if [[ ! -f artifacts/v10_final_summary.json ]]; then
  echo "Missing consumed final summary; refusing diagnostic." >&2
  exit 2
fi

"$PY" -u scripts/diagnose_v10_final_elites.py --execute
