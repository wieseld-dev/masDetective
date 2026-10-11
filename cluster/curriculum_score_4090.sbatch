#!/bin/bash
#SBATCH --job-name=cur-score
#SBATCH --gpus=1
#SBATCH --constraint=rtx_4090
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --time=0-03:00:00
#SBATCH --output=logs/cur-score-%j.out
#SBATCH --error=logs/cur-score-%j.err
set -euo pipefail
cd "$HOME/mas_rl_detective"
export PYTHONPATH="$HOME/mas_rl_detective:${PYTHONPATH:-}"
export PATH="$HOME/.conda/envs/masdetective/bin:$PATH"
PY="$HOME/.conda/envs/masdetective/bin/python"
"$PY" -u scripts/adaptive_v11_score_round.py --round "$CURR_ROUND" --split "$CURR_SPLIT" --root "$CURR_ROOT" --scorer "${CURR_SCORER:-adaptive_score_v135.py}"
