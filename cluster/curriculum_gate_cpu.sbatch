#!/bin/bash
#SBATCH --job-name=cur-gate
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=0-00:20:00
#SBATCH --output=logs/cur-gate-%j.out
#SBATCH --error=logs/cur-gate-%j.err
set -euo pipefail
cd "$HOME/mas_rl_detective"
export PYTHONPATH="$HOME/mas_rl_detective:${PYTHONPATH:-}"
PY="$HOME/.conda/envs/masdetective/bin/python"
"$PY" -u scripts/curriculum_gate.py --update-summary "$CURR_UPDATE_SUMMARY" --eval-report "$CURR_EVAL_REPORT" --strict "${CURR_STRICT_GATE:-1}"
