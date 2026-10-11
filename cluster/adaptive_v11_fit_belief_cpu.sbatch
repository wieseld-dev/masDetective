#!/bin/bash
#SBATCH --job-name=av11-belief
#SBATCH --partition=main
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=logs/av11-belief-%j.out
#SBATCH --error=logs/av11-belief-%j.err
set -euo pipefail
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
cd "$HOME/mas_rl_detective"
export PYTHONPATH="$HOME/mas_rl_detective:${PYTHONPATH:-}"
export PATH="$HOME/.conda/envs/masdetective/bin:$PATH"
PY="$HOME/.conda/envs/masdetective/bin/python"
ARGS=()
if [ -n "${AV11_BASELINE_BANK:-}" ]; then ARGS+=(--bank "$AV11_BASELINE_BANK"); fi
"$PY" -u scripts/adaptive_v11_fit_belief.py "${ARGS[@]}"
