#!/bin/bash
#SBATCH --job-name=ck-gen
#SBATCH --gpus=1
#SBATCH --constraint=rtx_4090
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/ck-gen-%j.out
#SBATCH --error=logs/ck-gen-%j.err

set -euo pipefail

cd "$HOME/mas_rl_detective"

export PYTHONPATH="$HOME/mas_rl_detective:${PYTHONPATH:-}"
export PATH="$HOME/.conda/envs/masdetective/bin:$PATH"
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false

PY="$HOME/.conda/envs/masdetective/bin/python"

: "${CK_ROOT:?}"
: "${CK_ROUND:?}"
: "${CK_ADAPTER:?}"
: "${CK_SELECTOR:?}"

"$PY" -u scripts/curriculum_v2_generate.py \
  --round "$CK_ROUND" \
  --split dev \
  --root "$CK_ROOT" \
  --adapter "$CK_ADAPTER" \
  --v11-reference artifacts/v11_policy \
  --group-size 8 \
  --selector-model "$CK_SELECTOR" \
  --selector-epsilon 0.0 \
  --selector-min-samples 96 \
  --stage 10
