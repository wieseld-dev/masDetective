#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=20:00:00
#SBATCH --job-name=v317-lcurve
#SBATCH --output=logs/v317-lcurve-%J.out
#SBATCH --error=logs/v317-lcurve-%J.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

export PYTHONPATH="$PWD:${PYTHONPATH:-}"

PYTHON="$HOME/.conda/envs/masdetective/bin/python"

echo "START=$(date)"
echo "STRICT UNSEEN-SYSTEM LEARNING CURVE"
echo "NO API"
echo "NO MAS CALLS"
echo "TIME LIMIT=20 HOURS"

"$PYTHON" -u \
  scripts/v3_17_system_learning_curve.py

echo "END=$(date)"
