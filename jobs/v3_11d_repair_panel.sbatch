#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:15:00
#SBATCH --job-name=v311d-repair
#SBATCH --output=logs/v311d-repair-%J.out
#SBATCH --error=logs/v311d-repair-%J.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

PYTHON="$HOME/.conda/envs/masdetective/bin/python"

echo "START=$(date)"
echo "LOCAL GENERATOR ONLY"
echo "NO OPENAI"
echo "NO MAS EXECUTIONS"
echo

"$PYTHON" -u \
  scripts/v3_11d_repair_panel.py

echo
echo "END=$(date)"
