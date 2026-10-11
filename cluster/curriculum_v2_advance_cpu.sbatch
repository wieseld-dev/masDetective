#!/bin/bash
#SBATCH --job-name=cur-v2-adv
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=0-00:45:00
#SBATCH --output=logs/cur-v2-adv-%j.out
#SBATCH --error=logs/cur-v2-adv-%j.err
set -euo pipefail
cd "$HOME/mas_rl_detective"
export PYTHONPATH="$HOME/mas_rl_detective:${PYTHONPATH:-}"
PY="$HOME/.conda/envs/masdetective/bin/python"
"$PY" -u scripts/curriculum_v2_advance.py --round "$CURR_ROUND" --split "$CURR_SPLIT" --root "$CURR_ROOT" --manifest "$CURR_MANIFEST" --belief "$CURR_BELIEF" --selector-model "$CURR_SELECTOR_MODEL" --stage "${CURR_STAGE:-1}"
