#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --job-name=v36-signal
#SBATCH --output=logs/v36-signal-%J.out
#SBATCH --error=logs/v36-signal-%J.err

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
echo "COUNTERFACTUAL REPLAY ONLY"
echo

"$PYTHON" -u \
  scripts/v3_6_counterfactual_signal.py

echo
echo "END=$(date)"
