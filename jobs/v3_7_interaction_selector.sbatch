#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --job-name=v37-inter
#SBATCH --output=logs/v37-inter-%J.out
#SBATCH --error=logs/v37-inter-%J.err

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
echo "COUNTERFACTUAL STATE x QUERY TRAINING"
echo

"$PYTHON" -u \
    scripts/v3_7_interaction_selector.py

echo
echo "END=$(date)"
