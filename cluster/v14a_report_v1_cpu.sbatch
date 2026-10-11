#!/bin/bash
#SBATCH --job-name=v14a-report
#SBATCH --partition=main
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/v14a-report-%j.out
#SBATCH --error=logs/v14a-report-%j.err

set -euo pipefail

cd "$HOME/mas_rl_detective"

export PYTHONPATH="$HOME/mas_rl_detective:${PYTHONPATH:-}"
export PATH="$HOME/.conda/envs/masdetective/bin:$PATH"

: "${V14A_ROOT:?}"
: "${V14A_TOTAL_ROUNDS:?}"

"$HOME/.conda/envs/masdetective/bin/python" \
    -u scripts/v14a_report_v1.py \
    --root "$V14A_ROOT" \
    --rounds "$V14A_TOTAL_ROUNDS"
