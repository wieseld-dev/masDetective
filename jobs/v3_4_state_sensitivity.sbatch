#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --job-name=v34-state
#SBATCH --output=logs/v34-state-%J.out
#SBATCH --error=logs/v34-state-%J.err

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
echo "PURE OFFLINE SELECTOR AUDIT"
echo

"$PYTHON" -u \
  scripts/v3_4_state_sensitivity.py

echo
echo "END=$(date)"
