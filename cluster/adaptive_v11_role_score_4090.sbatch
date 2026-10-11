#!/bin/bash
#SBATCH --job-name=av11-rscore
#SBATCH --partition=main
#SBATCH --gpus=1
#SBATCH --constraint=rtx_4090
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/av11-rscore-%j.out
#SBATCH --error=logs/av11-rscore-%j.err
set -euo pipefail
cd "$HOME/mas_rl_detective"
export PYTHONPATH="$HOME/mas_rl_detective:${PYTHONPATH:-}"
export PATH="$HOME/.conda/envs/masdetective/bin:$PATH"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
PY="$HOME/.conda/envs/masdetective/bin/python"
: "${AV11_ROUND:?set AV11_ROUND}"
ROOT="${AV11_ROLE_ROOT:-artifacts/adaptive_v11_role_only_v1}"
"$PY" -u scripts/adaptive_v11_score_round.py --round "$AV11_ROUND" --split "${AV11_SPLIT:-train}" --root "$ROOT"
