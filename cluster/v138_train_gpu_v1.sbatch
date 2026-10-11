#!/bin/bash
#SBATCH --job-name=v138-pair
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=0-06:00:00
#SBATCH --output=logs/v138-pair-%j.out
#SBATCH --error=logs/v138-pair-%j.err

set -euo pipefail

cd "$HOME/mas_rl_detective"

source /storage/modules/packages/anaconda/etc/profile.d/conda.sh
conda activate masdetective

export PYTHONPATH="$HOME/mas_rl_detective:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false

echo "===== GPU ====="
nvidia-smi

echo
echo "===== V13.8 ====="

python -u \
  analysis/train_v138_causal_pairwise_v1.py
