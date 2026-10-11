#!/bin/bash
#SBATCH --job-name=ck-adv
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/ck-adv-%j.out
#SBATCH --error=logs/ck-adv-%j.err

set -euo pipefail

cd "$HOME/mas_rl_detective"

export PYTHONPATH="$HOME/mas_rl_detective:${PYTHONPATH:-}"
export PATH="$HOME/.conda/envs/masdetective/bin:$PATH"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1

PY="$HOME/.conda/envs/masdetective/bin/python"

: "${CK_ROOT:?}"
: "${CK_MANIFEST:?}"
: "${CK_BELIEF:?}"
: "${CK_SELECTOR:?}"
: "${CK_ROUND:?}"

"$PY" -u scripts/curriculum_v2_advance.py \
  --round "$CK_ROUND" \
  --split dev \
  --root "$CK_ROOT" \
  --manifest "$CK_MANIFEST" \
  --belief "$CK_BELIEF" \
  --selector-model "$CK_SELECTOR" \
  --stage 10
