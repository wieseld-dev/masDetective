#!/bin/bash
#SBATCH --partition=main
#SBATCH --time=0-00:30:00
#SBATCH --job-name=visbridge
#SBATCH --output=logs/visbridge-%J.out
#SBATCH --error=logs/visbridge-%J.err
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8
#SBATCH --gpus=0

set -euo pipefail
cd ~/mas_rl_detective
mkdir -p logs artifacts/visible_adaptive_bridge_v1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate masdetective

python -u analysis/visible_adaptive_bridge_v1.py \
  --root artifacts/adaptive_v11_curriculum_v2 \
  --obs-dir artifacts/observation_model_v1 \
  --vis-dir artifacts/visible_output_fingerprint_v1 \
  --cal-dir artifacts/calibrated_bayes_diag_v1 \
  --out artifacts/visible_adaptive_bridge_v1 \
  --particles-per-k 16 \
  --device cpu
