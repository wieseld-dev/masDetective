#!/bin/bash
#SBATCH --partition=main
#SBATCH --time=0-04:00:00
#SBATCH --job-name=obsmdl-v1
#SBATCH --output=logs/obsmdl-v1-%J.out
#SBATCH --error=logs/obsmdl-v1-%J.err
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --gpus=0

set -euo pipefail

cd ~/mas_rl_detective
mkdir -p logs artifacts/observation_model_v1

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate masdetective

echo "===== ENV CHECK ====="
which python
python -c "import sys, torch; print('PYTHON=', sys.executable); print('TORCH=', torch.__version__)"

echo
echo "===== RUN ====="

python analysis/observation_model_v1.py \
  --root artifacts/adaptive_v11_curriculum_v2 \
  --out artifacts/observation_model_v1 \
  --device auto \
  --epochs 35 \
  --patience 6 \
  --batch-size 256 \
  --embed-batch-size 128
