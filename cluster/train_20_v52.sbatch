#!/bin/bash
#SBATCH --partition main
#SBATCH --time 0-01:30:00
#SBATCH --job-name mas-rl-v52-20
#SBATCH --output logs/mas-rl-v52-20-%J.out
#SBATCH --gpus=gtx_1080:1

set -euo pipefail

cd "$HOME/mas_rl_detective"
mkdir -p logs artifacts outputs

PY="$HOME/.conda/envs/masdetective/bin/python"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

set -a
source .env
set +a

"$PY" -u -m training.train_ppo \
  --estimator artifacts/k_estimator.joblib \
  --episodes 20 \
  --budget 3 \
  --rollout-batch 2 \
  --ppo-epochs 1 \
  --seed 73 \
  --output-dir outputs/detective_v5_2_run20 \
  --log artifacts/train_v5_2_run20.jsonl \
  --execute
