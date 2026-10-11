#!/bin/bash
#SBATCH --job-name=v14a-cksweep
#SBATCH --partition=main
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=logs/v14a-cksweep-%j.out
#SBATCH --error=logs/v14a-cksweep-%j.err

set -euo pipefail

cd "$HOME/mas_rl_detective"

export PYTHONPATH="$HOME/mas_rl_detective:${PYTHONPATH:-}"
export PATH="$HOME/.conda/envs/masdetective/bin:$PATH"

"$HOME/.conda/envs/masdetective/bin/python" \
    -u analysis/v14a_checkpoint_sweep_report.py
