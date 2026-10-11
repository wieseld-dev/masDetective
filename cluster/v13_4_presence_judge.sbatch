#!/bin/bash
#SBATCH --job-name=v13-4-pres
#SBATCH --output=logs/v13-4-pres-%j.out
#SBATCH --error=logs/v13-4-pres-%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G

set -euo pipefail

cd "$HOME/mas_rl_detective"

mkdir -p logs
mkdir -p artifacts/v13_4_presence_judge

source /storage/modules/packages/anaconda/etc/profile.d/conda.sh
conda activate masdetective

echo "========================================"
echo "V13.4 PRESENCE JUDGE"
echo "host=$(hostname)"
echo "date=$(date)"
echo "python=$(which python)"
python --version
nvidia-smi || true
echo "========================================"

python -u analysis/train_v13_4_presence_judge.py \
  --output-dir artifacts/v13_4_presence_judge \
  --epochs 4 \
  --max-length 1024 \
  --batch-size 4 \
  --pair-batch-size 2 \
  --eval-batch-size 8 \
  --eval-pair-batch-size 4 \
  --grad-accum 4 \
  --neg-ratio 2.0 \
  --self-target-neg-weight 2.5 \
  --hardneg-weight 1.5 \
  --pair-loss-weight 0.25 \
  --pair-margin 0.75

echo "========================================"
echo "FINISHED"
echo "date=$(date)"
echo "========================================"
