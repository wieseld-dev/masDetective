#!/bin/bash
#SBATCH --job-name=cur-eval
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0-00:30:00
#SBATCH --output=logs/cur-eval-%j.out
#SBATCH --error=logs/cur-eval-%j.err
set -euo pipefail
cd "$HOME/mas_rl_detective"
export PYTHONPATH="$HOME/mas_rl_detective:${PYTHONPATH:-}"
PY="$HOME/.conda/envs/masdetective/bin/python"
"$PY" -u scripts/curriculum_eval_report.py --root "$CURR_ROOT" --horizon "$CURR_HORIZON" --output "$CURR_EVAL_OUTPUT"
