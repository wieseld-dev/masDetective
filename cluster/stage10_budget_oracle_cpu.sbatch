#!/bin/bash
#SBATCH --job-name=s10-budget
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=0-00:10:00
#SBATCH --output=logs/s10-budget-%j.out
#SBATCH --error=logs/s10-budget-%j.err

set -euo pipefail

cd "$HOME/mas_rl_detective"

source /storage/modules/packages/anaconda/etc/profile.d/conda.sh
conda activate masdetective

python -u \
  analysis/stage10_budget_matched_oracle_v1.py
