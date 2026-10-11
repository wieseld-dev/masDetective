#!/bin/bash
#SBATCH --job-name=adapt-kfirst-v2
#SBATCH --partition=main
#SBATCH --gpus=0
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=logs/adapt-kfirst-v2-%j.out
#SBATCH --error=logs/adapt-kfirst-v2-%j.err

set -euo pipefail

cd "$HOME/mas_rl_detective"
export PYTHONPATH="$HOME/mas_rl_detective:${PYTHONPATH:-}"

source /storage/modules/packages/anaconda/etc/profile.d/conda.sh
conda activate masdetective

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

mkdir -p logs artifacts/adaptive_kfirst_v2

python -u adaptive_kfirst_fit_eval.py \
  --output-dir artifacts/adaptive_kfirst_v2 \
  --budgets 1,3,5,8,12,20 \
  --random-repeats 20 \
  --mc-samples 16 \
  --max-candidate-actions 12
