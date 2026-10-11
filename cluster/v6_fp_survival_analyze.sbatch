#!/bin/bash
#SBATCH --job-name=v6fp-an
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/v6fp-an-%j.out
#SBATCH --error=logs/v6fp-an-%j.err

set -euo pipefail

cd "$HOME/mas_rl_detective"

source /storage/modules/packages/anaconda/etc/profile.d/conda.sh
conda activate masdetective

export OMP_NUM_THREADS=8

python -u \
  analysis/v6_fp_survival_multisys_analyze.py
