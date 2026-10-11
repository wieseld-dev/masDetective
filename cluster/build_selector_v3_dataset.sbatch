#!/bin/bash
#SBATCH --job-name=selv3-data
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=10:00:00
#SBATCH --output=logs/selv3_data_%j.out
#SBATCH --error=logs/selv3_data_%j.err

cd "$HOME/mas_rl_detective" || exit 1
mkdir -p logs

export PYTHONPATH="$PWD"

PY="$HOME/.conda/envs/masdetective/bin/python"

"$PY" analysis/build_selector_v3_multiround_dataset.py \
  --base artifacts/adaptive_v11_curriculum_v2 \
  --belief artifacts/adaptive_v11_active_v1/frozen_belief.joblib \
  --out-dir artifacts/selector_v3_multiround_g8_dataset_v1
