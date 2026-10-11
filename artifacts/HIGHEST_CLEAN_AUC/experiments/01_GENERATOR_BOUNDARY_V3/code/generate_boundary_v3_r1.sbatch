#!/bin/bash
#SBATCH --job-name=hca-bnd-v3
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --constraint=rtx_4090
#SBATCH --exclude=cs-4090-09
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=10:00:00
#SBATCH --output=logs/hca-bnd-v3-%j.out
#SBATCH --error=logs/hca-bnd-v3-%j.err

set -euo pipefail

cd "$HOME/mas_rl_detective"

export PYTHONPATH="$HOME/mas_rl_detective:${PYTHONPATH:-}"
export PATH="$HOME/.conda/envs/masdetective/bin:$PATH"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

PY="$HOME/.conda/envs/masdetective/bin/python"

HCA="artifacts/HIGHEST_CLEAN_AUC"
V3="$HCA/experiments/01_GENERATOR_BOUNDARY_V3"
ROOT="$V3/data/r1_test"

"$PY" -u \
  "$V3/code/generate_boundary_v3.py" \
  --round 1 \
  --split dev \
  --group-size 4 \
  --root "$ROOT" \
  --adapter "$HCA/models/generator_clean_r7" \
  --v11-adapter "$HCA/models/generator_v11_base" \
  --temperature 0.90 \
  --top-p 0.95 \
  --max-new-tokens 64 \
  --max-attempts 8
