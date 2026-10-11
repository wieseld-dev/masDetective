#!/bin/bash
#SBATCH --job-name=rd-score
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx_4090:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=20:00:00
#SBATCH --output=logs/rd-score-%j.out
#SBATCH --error=logs/rd-score-%j.err

set -euo pipefail

cd "$HOME/mas_rl_detective"

export PYTHONPATH="$PWD:${PYTHONPATH:-}"

PY="$HOME/.conda/envs/masdetective/bin/python"

ROOT="${RD_ROOT:-artifacts/role_direct_approach_v1}"

nvidia-smi \
  --query-gpu=name,memory.total,memory.free \
  --format=csv,noheader

"$PY" \
  scripts/adaptive_v11_score_round.py \
  --round 1 \
  --split dev \
  --root "$ROOT"
