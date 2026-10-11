#!/bin/bash
#SBATCH --partition=main
#SBATCH --time=0-00:15:00
#SBATCH --job-name=v31-cluster
#SBATCH --output=logs/v31-cluster-%J.out
#SBATCH --mem=6G
#SBATCH --cpus-per-task=2

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

# Make local research packages importable.
PKG_DIR="$(find "$PWD" -maxdepth 5 -type d -name adaptive_v11_active -print -quit)"
if [ -z "$PKG_DIR" ]; then
    echo "ERROR: adaptive_v11_active not found"
    exit 2
fi
export PYTHONPATH="$(dirname "$PKG_DIR"):$PWD/scripts:$PWD:${PYTHONPATH:-}"

PYTHON="$HOME/.conda/envs/masdetective/bin/python"

echo "============================================================"
echo "V3.1 CLUSTERED STATISTICAL AUDIT"
echo "JOB=$SLURM_JOB_ID"
echo "START=$(date)"
echo "PYTHON=$PYTHON"
echo "============================================================"

"$PYTHON" -u scripts/v31_cluster_stats.py

echo
cat artifacts/eqs_v3_1_mas_auc/clustered_stats.json

echo "END=$(date)"
