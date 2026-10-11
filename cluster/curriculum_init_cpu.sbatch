#!/bin/bash
#SBATCH --job-name=cur-init
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=0-00:20:00
#SBATCH --output=logs/cur-init-%j.out
#SBATCH --error=logs/cur-init-%j.err
set -euo pipefail
cd "$HOME/mas_rl_detective"
export PYTHONPATH="$HOME/mas_rl_detective:${PYTHONPATH:-}"
PY="$HOME/.conda/envs/masdetective/bin/python"
"$PY" -u scripts/curriculum_init_states.py --manifest "$CURR_MANIFEST" --belief "$CURR_BELIEF" --root "$CURR_ROOT" --split "$CURR_SPLIT"
