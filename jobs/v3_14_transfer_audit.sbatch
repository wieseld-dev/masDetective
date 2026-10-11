#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=20:00:00
#SBATCH --job-name=v314-transfer
#SBATCH --output=logs/v314-transfer-%J.out
#SBATCH --error=logs/v314-transfer-%J.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

export PYTHONPATH="$PWD:${PYTHONPATH:-}"

PYTHON="$HOME/.conda/envs/masdetective/bin/python"

echo "START=$(date)"
echo "NO GPU"
echo "NO API"
echo "NO MAS CALLS"
echo "TIME LIMIT=20 HOURS"

"$PYTHON" -u \
  scripts/v3_14_transfer_audit.py

echo "END=$(date)"
