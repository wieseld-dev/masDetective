#!/bin/bash
#SBATCH --job-name=p310-report
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=0-00:10:00
#SBATCH --output=logs/p310-report-%j.out
#SBATCH --error=logs/p310-report-%j.err

set -euo pipefail
cd "$HOME/mas_rl_detective"

export PYTHONPATH="$HOME/mas_rl_detective:${PYTHONPATH:-}"
PY="$HOME/.conda/envs/masdetective/bin/python"

"$PY" -u presence_3of10/report.py \
    --root "$P310_ROOT" \
    --horizon "${P310_HORIZON:-7}"
