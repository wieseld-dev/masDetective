#!/bin/bash
#SBATCH --job-name=cur-v2-boot
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=0-00:45:00
#SBATCH --output=logs/cur-v2-boot-%j.out
#SBATCH --error=logs/cur-v2-boot-%j.err
set -euo pipefail
cd "$HOME/mas_rl_detective"
export PYTHONPATH="$HOME/mas_rl_detective:${PYTHONPATH:-}"
PY="$HOME/.conda/envs/masdetective/bin/python"
"$PY" -u scripts/curriculum_v2_bootstrap_selector.py --source-root "${CURR_BOOTSTRAP_SOURCE:-artifacts/adaptive_v11_curriculum_v1}" --selector-model "$CURR_SELECTOR_MODEL" --max-stage "${CURR_BOOTSTRAP_MAX_STAGE:-2}" --reset
