#!/bin/bash
#SBATCH --job-name=v13-6-act
#SBATCH --partition=main
#SBATCH --gpus=1
#SBATCH --constraint=rtx_4090
#SBATCH --mem=48G
#SBATCH --cpus-per-task=6
#SBATCH --time=08:00:00
#SBATCH --output=logs/v13-6-act-%j.out
#SBATCH --error=logs/v13-6-act-%j.err

set -euo pipefail

cd "$HOME/mas_rl_detective"
mkdir -p logs artifacts/v13_6_activation_judge

source /storage/modules/packages/anaconda/etc/profile.d/conda.sh
conda activate masdetective

export PYTHONPATH="$HOME/mas_rl_detective:${PYTHONPATH:-}"

python -u analysis/train_v13_6_activation_judge.py \
  --output-dir artifacts/v13_6_activation_judge \
  --epochs 5 \
  --lr 8e-5 \
  --neg-ratio 3.0 \
  --self-target-neg-weight 4.0 \
  --hardneg-weight 2.0 \
  --pair-loss-weight 0.50 \
  --pair-margin 1.0 \
  --batch-size 4 \
  --pair-batch-size 2 \
  --grad-accum 4 \
  --early-stop-patience 2
