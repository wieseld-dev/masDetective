#!/bin/bash
#SBATCH --partition=main
#SBATCH --time=0-01:00:00
#SBATCH --job-name=postgain-v1
#SBATCH --output=logs/postgain-v1-%J.out
#SBATCH --error=logs/postgain-v1-%J.err
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --gpus=0

set -euo pipefail
cd ~/mas_rl_detective
mkdir -p logs artifacts/realized_posterior_gain_v1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate masdetective

python analysis/realized_posterior_gain_diag_v1.py \
  --root artifacts/adaptive_v11_curriculum_v2 \
  --obs-dir artifacts/observation_model_v1 \
  --out artifacts/realized_posterior_gain_v1 \
  --particles-per-k 64 \
  --device cpu
