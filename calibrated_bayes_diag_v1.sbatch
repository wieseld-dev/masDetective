#!/bin/bash
#SBATCH --partition=main
#SBATCH --time=0-01:00:00
#SBATCH --job-name=calbayes-v1
#SBATCH --output=logs/calbayes-v1-%J.out
#SBATCH --error=logs/calbayes-v1-%J.err
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --gpus=0

set -euo pipefail
cd ~/mas_rl_detective
mkdir -p logs artifacts/calibrated_bayes_diag_v1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate masdetective

python analysis/calibrated_bayes_diag_v1.py \
  --root artifacts/adaptive_v11_curriculum_v2 \
  --obs-dir artifacts/observation_model_v1 \
  --out artifacts/calibrated_bayes_diag_v1 \
  --particles-per-k 32 \
  --device cpu
