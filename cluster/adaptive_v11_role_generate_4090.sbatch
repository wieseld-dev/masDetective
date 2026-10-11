#!/bin/bash
#SBATCH --job-name=av11-rgen
#SBATCH --partition=main
#SBATCH --gpus=1
#SBATCH --constraint=rtx_4090
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/av11-rgen-%j.out
#SBATCH --error=logs/av11-rgen-%j.err
set -euo pipefail
cd "$HOME/mas_rl_detective"
export PYTHONPATH="$HOME/mas_rl_detective:${PYTHONPATH:-}"
export PATH="$HOME/.conda/envs/masdetective/bin:$PATH"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
PY="$HOME/.conda/envs/masdetective/bin/python"
: "${AV11_ROUND:?set AV11_ROUND}"
ROOT="${AV11_ROLE_ROOT:-artifacts/adaptive_v11_role_only_v1}"
SPLIT="${AV11_SPLIT:-train}"
G="${AV11_GROUP_SIZE:-4}"
ARGS=(--round "$AV11_ROUND" --split "$SPLIT" --group-size "$G" --root "$ROOT" --no-4bit)
if [ -n "${AV11_ADAPTER:-}" ]; then ARGS+=(--adapter "$AV11_ADAPTER"); fi
"$PY" -u scripts/adaptive_v11_role_generate_round.py "${ARGS[@]}"
