#!/bin/bash
#SBATCH --job-name=av11-pre
#SBATCH --partition=main
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=00:10:00
#SBATCH --output=logs/av11-pre-%j.out
#SBATCH --error=logs/av11-pre-%j.err

set -euo pipefail

cd "$HOME/mas_rl_detective"

export PYTHONPATH="$HOME/mas_rl_detective:${PYTHONPATH:-}"
export PATH="$HOME/.conda/envs/masdetective/bin:$PATH"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1

PY="$HOME/.conda/envs/masdetective/bin/python"

ARGS=()
if [ -n "${AV11_REQUIRE_BELIEF:-}" ]; then
    ARGS+=(--require-belief)
fi

"$PY" -u scripts/adaptive_v11_preflight.py "${ARGS[@]}"
