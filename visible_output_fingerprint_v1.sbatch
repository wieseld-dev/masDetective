#!/bin/bash
#SBATCH --partition=main
#SBATCH --time=0-01:00:00
#SBATCH --job-name=visfp-v1
#SBATCH --output=logs/visfp-v1-%J.out
#SBATCH --error=logs/visfp-v1-%J.err
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --gpus=0

set -euo pipefail
cd ~/mas_rl_detective
mkdir -p logs artifacts/visible_output_fingerprint_v1

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate masdetective

python analysis/visible_output_fingerprint_v1.py \
  --root artifacts/adaptive_v11_curriculum_v2 \
  --obs-dir artifacts/observation_model_v1 \
  --out artifacts/visible_output_fingerprint_v1 \
  --device cpu \
  --embed-batch-size 128
