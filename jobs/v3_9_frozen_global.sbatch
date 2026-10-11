#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:10:00
#SBATCH --job-name=v39-frozen
#SBATCH --output=logs/v39-frozen-%J.out
#SBATCH --error=logs/v39-frozen-%J.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

PKG_DIR="$(
    find "$PWD" \
      -maxdepth 5 \
      -type d \
      -name adaptive_v11_active \
      -print \
      -quit
)"

if [ -n "$PKG_DIR" ]; then
    export PYTHONPATH="$(dirname "$PKG_DIR"):$PWD:${PYTHONPATH:-}"
else
    export PYTHONPATH="$PWD:${PYTHONPATH:-}"
fi

PYTHON="$HOME/.conda/envs/masdetective/bin/python"

echo "START=$(date)"
echo "NO OPENAI"
echo "NO MAS EXECUTIONS"
echo "FROZEN GLOBAL = V3.1 QUERY-ONLY ENSEMBLE"
echo

"$PYTHON" -u \
    scripts/v3_9_frozen_global.py

echo
echo "END=$(date)"
