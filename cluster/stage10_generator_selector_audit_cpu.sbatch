#!/bin/bash
#SBATCH --job-name=s10-audit
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=0-00:10:00
#SBATCH --output=logs/s10-audit-%j.out
#SBATCH --error=logs/s10-audit-%j.err

set -euo pipefail

cd "$HOME/mas_rl_detective"

source /storage/modules/packages/anaconda/etc/profile.d/conda.sh
conda activate masdetective

python -u \
  analysis/stage10_generator_selector_audit_v1.py
