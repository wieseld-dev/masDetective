#!/bin/bash
#SBATCH --partition main
#SBATCH --time 0-06:00:00
#SBATCH --job-name v63-fanout48
#SBATCH --output logs/v63-fanout48-%J.out
#SBATCH --gpus=gtx_1080:1

set -euo pipefail
cd "$HOME/mas_rl_detective"
mkdir -p logs artifacts outputs
PY="$HOME/.conda/envs/masdetective/bin/python"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set -a
source .env
set +a

"$PY" -u -m training.train_role_ppo_v63 \
  --estimator artifacts/role_presence_estimator_v63.joblib \
  --episodes 48 \
  --budget 4 \
  --rollout-batch 8 \
  --ppo-epochs 2 \
  --lr 1e-5 \
  --clip 0.20 \
  --gamma 0.95 \
  --entropy-coef 0.01 \
  --seed 73 \
  --output-dir outputs/detective_v6_3_fanout48 \
  --log artifacts/train_v6_3_fanout48.jsonl \
  --execute
