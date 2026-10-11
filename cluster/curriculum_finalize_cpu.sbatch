#!/bin/bash
#SBATCH --job-name=cur-final
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=0-00:30:00
#SBATCH --output=logs/cur-final-%j.out
#SBATCH --error=logs/cur-final-%j.err
set -euo pipefail
cd "$HOME/mas_rl_detective"
export PYTHONPATH="$HOME/mas_rl_detective:${PYTHONPATH:-}"
PY="$HOME/.conda/envs/masdetective/bin/python"
"$PY" -u scripts/curriculum_finalize.py --root "$CURR_EXP_ROOT" --source "$CURR_FINAL_SOURCE" --v11 "${CURR_V11:-artifacts/v11_policy}" --hash-file "$CURR_V11_HASH_FILE"
