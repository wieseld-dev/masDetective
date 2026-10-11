#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:15:00
#SBATCH --job-name=v311b-panel
#SBATCH --output=logs/v311b-panel-%J.out
#SBATCH --error=logs/v311b-panel-%J.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

export PYTHONPATH="$PWD:${PYTHONPATH:-}"

PYTHON="$HOME/.conda/envs/masdetective/bin/python"

echo "START=$(date)"
echo "NO OPENAI"
echo "NO MAS EXECUTIONS"
echo "BUILDING ROLE-BALANCED SHARED PANEL"
echo

"$PYTHON" -u \
  scripts/v3_11b_build_balanced_panel.py

echo
echo "END=$(date)"
