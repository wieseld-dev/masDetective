#!/bin/bash
#SBATCH --job-name=count-adv
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=0-00:10:00
#SBATCH --output=logs/count-adv-%j.out
#SBATCH --error=logs/count-adv-%j.err

set -euo pipefail
cd "$HOME/mas_rl_detective"

export PYTHONPATH="$HOME/mas_rl_detective:${PYTHONPATH:-}"
PY="$HOME/.conda/envs/masdetective/bin/python"

"$PY" -u presence_count_sanity/run.py \
  --phase advance \
  --root "$COUNT_ROOT" \
  --belief "$COUNT_BELIEF"
