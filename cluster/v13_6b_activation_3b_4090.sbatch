#!/bin/bash
#SBATCH --job-name=v136b-3b
#SBATCH --partition=main
#SBATCH --gpus=1
#SBATCH --constraint=rtx_4090
#SBATCH --mem=48G
#SBATCH --cpus-per-task=6
#SBATCH --time=08:00:00
#SBATCH --output=logs/v136b-3b-%j.out
#SBATCH --error=logs/v136b-3b-%j.err

set -euo pipefail

cd "$HOME/mas_rl_detective"
mkdir -p logs artifacts/v13_6b_activation_judge

source /storage/modules/packages/anaconda/etc/profile.d/conda.sh
conda activate masdetective
export PYTHONPATH="$HOME/mas_rl_detective:${PYTHONPATH:-}"

python -u analysis/train_v13_6b_activation_judge.py \
  --output-dir artifacts/v13_6b_activation_judge \
  --base-model Qwen/Qwen2.5-3B-Instruct \
  --epochs 7 \
  --lr 6e-5 \
  --neg-ratio 3.0 \
  --self-target-neg-weight 5.0 \
  --hardneg-weight 2.5 \
  --pair-loss-weight 0.75 \
  --pair-margin 1.25 \
  --lora-r 32 \
  --lora-alpha 64 \
  --batch-size 4 \
  --pair-batch-size 2 \
  --grad-accum 4 \
  --early-stop-patience 2
