#!/bin/bash
#SBATCH --job-name=v13-1-strong
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/v13-1-strong-%j.out

set -euo pipefail
cd "$HOME/mas_rl_detective"
mkdir -p logs artifacts/v13_1_strong_judge

source /storage/modules/packages/anaconda/etc/profile.d/conda.sh
conda activate masdetective

export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false

python analysis/train_v13_1_strong_judge.py \
  --model auto \
  --output-dir artifacts/v13_1_strong_judge \
  --epochs 8 \
  --patience 2 \
  --batch-size 2 \
  --grad-accum 8 \
  --eval-batch-size 4 \
  --max-length 1400 \
  --lr 1e-4 \
  --margin 0.75 \
  --bce-weight 0.50 \
  --seed 1317 \
  --fold 0 \
  --n-splits 5
