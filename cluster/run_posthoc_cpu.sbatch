#!/bin/bash
#SBATCH --job-name=av11-kpost
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=01:00:00
#SBATCH --output=logs/av11-kpost-%j.out
#SBATCH --error=logs/av11-kpost-%j.err

set -euo pipefail
cd "$HOME/mas_rl_detective"
export PYTHONPATH="$HOME/mas_rl_detective:${PYTHONPATH:-}"
export PATH="$HOME/.conda/envs/masdetective/bin:$PATH"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1

PY="$HOME/.conda/envs/masdetective/bin/python"
"$PY" -u scripts/av11_posthoc_k_mechanism.py \
  --root artifacts/adaptive_v11_factorial_heldout32 \
  --belief artifacts/adaptive_v11_active_v1/frozen_belief.joblib \
  --out artifacts/adaptive_v11_factorial_heldout32/posthoc_k_mechanism \
  --cv-C 0.20
