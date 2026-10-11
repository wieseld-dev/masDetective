#!/bin/bash
#SBATCH --partition=main
#SBATCH --time=0-00:10:00
#SBATCH --job-name=v32-preflight
#SBATCH --output=logs/v32-preflight-%J.out
#SBATCH --mem=2G
#SBATCH --cpus-per-task=1

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

mkdir -p logs artifacts/eqs_v3_2_closed_loop

PYTHON="$HOME/.conda/envs/masdetective/bin/python"
if [ ! -x "$PYTHON" ]; then
  PYTHON="$HOME/.conda/envs/eqs/bin/python"
fi

echo "V3.2 CLOSED-LOOP PREFLIGHT"
echo "JOB=$SLURM_JOB_ID"
echo "NODE=$(hostname)"
echo "START=$(date)"
echo "PYTHON=$PYTHON"
echo

"$PYTHON" -u scripts/v32_closed_loop_preflight.py

echo
echo "END=$(date)"
