#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=20:00:00
#SBATCH --job-name=v316-seenstate
#SBATCH --output=logs/v316-seenstate-%J.out
#SBATCH --error=logs/v316-seenstate-%J.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

export PYTHONPATH="$PWD:${PYTHONPATH:-}"

PYTHON="$HOME/.conda/envs/masdetective/bin/python"

echo "START=$(date)"
echo "DIAGNOSTIC ONLY"
echo "NO API"
echo "NO MAS CALLS"
echo "TIME LIMIT=20 HOURS"

"$PYTHON" -u \
  scripts/v3_16_seen_system_state_transfer.py

echo "END=$(date)"
