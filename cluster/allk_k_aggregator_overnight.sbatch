#!/bin/bash
#SBATCH --job-name=allk-k-v1
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-12:00:00
#SBATCH --output=logs/allk-k-v1-%j.out
#SBATCH --error=logs/allk-k-v1-%j.err

set -euo pipefail

cd "$HOME/mas_rl_detective"

source \
  /storage/modules/packages/anaconda/etc/profile.d/conda.sh

conda activate masdetective

export PYTHONPATH="$HOME/mas_rl_detective:${PYTHONPATH:-}"

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false

python -u \
  analysis/train_allk_k_aggregator_v1.py
