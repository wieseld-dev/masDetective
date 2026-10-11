#!/bin/bash
#SBATCH --job-name=p310-adv
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=0-00:10:00
#SBATCH --output=logs/p310-adv-%j.out
#SBATCH --error=logs/p310-adv-%j.err

set -euo pipefail
cd "$HOME/mas_rl_detective"

export PYTHONPATH="$HOME/mas_rl_detective:${PYTHONPATH:-}"
PY="$HOME/.conda/envs/masdetective/bin/python"

"$PY" -u presence_3of10/advance_round.py \
    --root "$P310_ROOT" \
    --round "$P310_ROUND" \
    --belief "$P310_BELIEF"
