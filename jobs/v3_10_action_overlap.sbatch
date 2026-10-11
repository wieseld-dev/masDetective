#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:05:00
#SBATCH --job-name=v310-overlap
#SBATCH --output=logs/v310-overlap-%J.out
#SBATCH --error=logs/v310-overlap-%J.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

PYTHON="$HOME/.conda/envs/masdetective/bin/python"

echo "START=$(date)"
echo "NO OPENAI"
echo "NO MAS EXECUTIONS"
echo

"$PYTHON" -u \
  scripts/v3_10_action_overlap.py

echo
echo "END=$(date)"
