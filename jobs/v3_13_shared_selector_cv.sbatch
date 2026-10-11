#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=20:00:00
#SBATCH --job-name=v313-sharedcv
#SBATCH --output=logs/v313-sharedcv-%J.out
#SBATCH --error=logs/v313-sharedcv-%J.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

PYTHON="$HOME/.conda/envs/masdetective/bin/python"

export PYTHONPATH="$PWD:${PYTHONPATH:-}"

echo "START=$(date)"
echo "HOST=$(hostname)"
echo "V3.13 STRICT HELD-OUT-SYSTEM CV"
echo "NO DEV SYSTEMS"
echo "NO MAS/API CALLS"
echo "TIME LIMIT=20 HOURS"

"$PYTHON" -u \
  scripts/v3_13_shared_selector_cv.py

echo "END=$(date)"
