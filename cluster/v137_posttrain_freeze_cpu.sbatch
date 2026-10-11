#!/bin/bash
#SBATCH --job-name=v137-freeze
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0-00:30:00
#SBATCH --output=logs/v137-freeze-%j.out
#SBATCH --error=logs/v137-freeze-%j.err

set -euo pipefail

cd "$HOME/mas_rl_detective"

source /storage/modules/packages/anaconda/etc/profile.d/conda.sh
conda activate masdetective

export PYTHONPATH="$HOME/mas_rl_detective:${PYTHONPATH:-}"

python -u \
  analysis/v137_posttrain_audit_freeze_v1.py
