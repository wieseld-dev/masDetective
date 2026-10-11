#!/bin/bash
#SBATCH --job-name=adapt-smoke
#SBATCH --partition=main
#SBATCH --gpus=0
#SBATCH --mem=12G
#SBATCH --output=logs/adapt-smoke-%j.out
#SBATCH --error=logs/adapt-smoke-%j.err
set -euo pipefail
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
cd "$HOME/mas_rl_detective"
source /storage/modules/packages/anaconda/etc/profile.d/conda.sh
conda activate masdetective
python -u adaptive_legacy_smoke.py
