#!/bin/bash
#SBATCH --job-name=adapt-collect
#SBATCH --partition=main
#SBATCH --gpus=0
#SBATCH --mem=16G
#SBATCH --output=logs/adapt-collect-%j.out
#SBATCH --error=logs/adapt-collect-%j.err
set -euo pipefail
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
cd "$HOME/mas_rl_detective"
source /storage/modules/packages/anaconda/etc/profile.d/conda.sh
conda activate masdetective
set -a
[ -f .env ] && source .env
set +a
python -u adaptive_collect_cpu.py
