#!/bin/bash
#SBATCH --job-name=rd-exec
#SBATCH --partition=main
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --time=20:00:00
#SBATCH --output=logs/rd-exec-%j.out
#SBATCH --error=logs/rd-exec-%j.err

set -euo pipefail

cd "$HOME/mas_rl_detective"

export PYTHONPATH="$PWD:${PYTHONPATH:-}"

set -a
source .env
set +a

PY="$HOME/.conda/envs/masdetective/bin/python"

ROOT="${RD_ROOT:-artifacts/role_direct_approach_v1}"

"$PY" \
  scripts/adaptive_v11_execute_round.py \
  --round 1 \
  --split dev \
  --root "$ROOT" \
  --manifest "$ROOT/manifest_privileged.json" \
  --workers 5
