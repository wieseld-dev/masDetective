#!/bin/bash
#SBATCH --job-name=av11-rexec
#SBATCH --partition=main
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=logs/av11-rexec-%j.out
#SBATCH --error=logs/av11-rexec-%j.err
set -euo pipefail
cd "$HOME/mas_rl_detective"
export PYTHONPATH="$HOME/mas_rl_detective:${PYTHONPATH:-}"
export PATH="$HOME/.conda/envs/masdetective/bin:$PATH"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
PY="$HOME/.conda/envs/masdetective/bin/python"
: "${AV11_ROUND:?set AV11_ROUND}"
ROOT="${AV11_ROLE_ROOT:-artifacts/adaptive_v11_role_only_v1}"
"$PY" -u scripts/adaptive_v11_execute_round.py \
  --round "$AV11_ROUND" --split "${AV11_SPLIT:-train}" --root "$ROOT" \
  --manifest artifacts/adaptive_v11_active_v1/manifest_privileged.json --workers "${AV11_API_WORKERS:-4}"
