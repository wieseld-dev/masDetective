#!/bin/bash
#SBATCH --job-name=hca-hyb-sel
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --constraint=rtx_4090
#SBATCH --exclude=cs-4090-09
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=10:00:00
#SBATCH --output=logs/hca-hyb-sel-%j.out
#SBATCH --error=logs/hca-hyb-sel-%j.err

set -euo pipefail

cd "$HOME/mas_rl_detective"

export PYTHONPATH="$HOME/mas_rl_detective:${PYTHONPATH:-}"
export PATH="$HOME/.conda/envs/masdetective/bin:$PATH"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

PY="$HOME/.conda/envs/masdetective/bin/python"

ROOT="artifacts/HIGHEST_CLEAN_AUC/experiments/01_GENERATOR_HYBRID_V2/data/r1_selector_test"
SELECTOR="artifacts/HIGHEST_CLEAN_AUC/models/selector_v2"

"$PY" -u analysis/select_v14_candidates_llmv2.py \
  --root "$ROOT" \
  --round 1 \
  --split dev \
  --adapter "$SELECTOR"
