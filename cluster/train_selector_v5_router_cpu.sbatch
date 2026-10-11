#!/bin/bash
#SBATCH --job-name=sel-v5
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=0-00:30:00
#SBATCH --output=logs/sel-v5-%j.out
#SBATCH --error=logs/sel-v5-%j.err

set -euo pipefail

cd "$HOME/mas_rl_detective"

source /storage/modules/packages/anaconda/etc/profile.d/conda.sh
conda activate masdetective

export PYTHONPATH="$HOME/mas_rl_detective:${PYTHONPATH:-}"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

python -u \
  scripts/train_selector_v5_router.py
