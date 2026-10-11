#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:05:00
#SBATCH --job-name=v311-panel
#SBATCH --output=logs/v311-panel-%J.out
#SBATCH --error=logs/v311-panel-%J.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

export PYTHONPATH="$PWD:${PYTHONPATH:-}"

PYTHON="$HOME/.conda/envs/masdetective/bin/python"

echo "START=$(date)"
echo "NO OPENAI"
echo "NO MAS EXECUTIONS"
echo

"$PYTHON" -u \
  scripts/v3_11_build_shared_panel.py

echo
echo "END=$(date)"
