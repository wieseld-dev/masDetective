#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=20:00:00
#SBATCH --job-name=v315-roleaudit
#SBATCH --output=logs/v315-roleaudit-%J.out
#SBATCH --error=logs/v315-roleaudit-%J.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

export PYTHONPATH="$PWD:${PYTHONPATH:-}"

PYTHON="$HOME/.conda/envs/masdetective/bin/python"

echo "START=$(date)"
echo "PRIVILEGED DIAGNOSTIC ONLY"
echo "NO API"
echo "NO MAS CALLS"
echo "TIME LIMIT=20 HOURS"

"$PYTHON" -u \
  scripts/v3_15_privileged_role_transfer.py

echo "END=$(date)"
