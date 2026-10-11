#!/bin/bash
#SBATCH --partition=main
#SBATCH --time=0-02:00:00
#SBATCH --job-name=specsel
#SBATCH --output=logs/specsel-%J.out
#SBATCH --gpus=0
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

set -euo pipefail

cd ~/mas_rl_detective
mkdir -p logs artifacts/spectral_selector_v1

module load anaconda
source activate masdetective

python spectral_probe_selector.py \
  --train artifacts/YOUR_TRAIN.jsonl \
  --eval artifacts/YOUR_EVAL.jsonl \
  --out artifacts/spectral_selector_v1 \
  --top-r 4 \
  --epochs 25 \
  --batch-size 128 \
  --device cpu
