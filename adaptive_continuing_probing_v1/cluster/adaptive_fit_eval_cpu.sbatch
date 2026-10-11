#!/bin/bash
#SBATCH --job-name=adapt-eval
#SBATCH --partition=main
#SBATCH --gpus=0
#SBATCH --mem=16G
#SBATCH --output=logs/adapt-eval-%j.out
#SBATCH --error=logs/adapt-eval-%j.err
set -euo pipefail
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
cd "$HOME/mas_rl_detective"
source /storage/modules/packages/anaconda/etc/profile.d/conda.sh
conda activate masdetective
python -u adaptive_fit_evaluate.py
python -u adaptive_train_imitation.py
