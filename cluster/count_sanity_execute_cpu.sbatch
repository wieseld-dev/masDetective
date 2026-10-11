#!/bin/bash
#SBATCH --job-name=count-exec
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=0-00:30:00
#SBATCH --output=logs/count-exec-%j.out
#SBATCH --error=logs/count-exec-%j.err

set -euo pipefail
cd "$HOME/mas_rl_detective"

export PYTHONPATH="$HOME/mas_rl_detective:${PYTHONPATH:-}"

if [[ -f .env ]]; then
    set -a
    source .env
    set +a
fi

PY="$HOME/.conda/envs/masdetective/bin/python"

"$PY" -u presence_count_sanity/run.py \
  --phase execute \
  --root "$COUNT_ROOT" \
  --belief "$COUNT_BELIEF"
