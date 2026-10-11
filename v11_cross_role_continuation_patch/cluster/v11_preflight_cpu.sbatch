#!/usr/bin/env bash
#SBATCH --job-name=v11-preflight
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=logs/v11-preflight-%j.out
set -euo pipefail
cd "$HOME/mas_rl_detective"
mkdir -p logs artifacts
PY="$HOME/.conda/envs/masdetective/bin/python"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

"$PY" -m pytest -q tests/test_v11_role_holdout.py tests/test_v11_hashcheck.py
"$PY" scripts/verify_v11_preflight.py

echo "V11 PREFLIGHT COMPLETE: no API calls were made."
