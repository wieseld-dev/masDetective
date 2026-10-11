#!/bin/bash
#SBATCH --job-name=v112-ev-r
#SBATCH --partition=main
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --output=logs/v112-ev-r-%j.out
#SBATCH --error=logs/v112-ev-r-%j.err
set -euo pipefail
cd "$HOME/mas_rl_detective"
export PYTHONPATH="$HOME/mas_rl_detective:${PYTHONPATH:-}"
source /storage/modules/packages/anaconda/etc/profile.d/conda.sh
conda activate masdetective
python -u evaluation/v11_2_strong_report.py
