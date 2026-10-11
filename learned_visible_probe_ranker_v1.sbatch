#!/bin/bash
#SBATCH --partition=main
#SBATCH --time=0-00:40:00
#SBATCH --job-name=visrank
#SBATCH --output=logs/visrank-%J.out
#SBATCH --error=logs/visrank-%J.err
#SBATCH --mem=24G
#SBATCH --cpus-per-task=8
#SBATCH --gpus=0

set -euo pipefail
cd ~/mas_rl_detective
mkdir -p logs artifacts/learned_visible_probe_ranker_v1

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate masdetective

python -u analysis/learned_visible_probe_ranker_v1.py \
  --root artifacts/adaptive_v11_curriculum_v2 \
  --obs-dir artifacts/observation_model_v1 \
  --vis-dir artifacts/visible_output_fingerprint_v1 \
  --cal-dir artifacts/calibrated_bayes_diag_v1 \
  --out artifacts/learned_visible_probe_ranker_v1 \
  --particles-per-k 8 \
  --pca-dim 24 \
  --device cpu
