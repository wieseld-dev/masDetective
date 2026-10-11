#!/bin/bash
#SBATCH --partition main
#SBATCH --time 0-04:00:00
#SBATCH --job-name mas-detective
#SBATCH --output logs/mas-detective-%J.out
#SBATCH --gpus=gtx_1080:1

set -euo pipefail
cd "$HOME/mas_rl_detective"
mkdir -p logs artifacts outputs
module load anaconda
source activate masdetective
set -a
source .env
set +a

nvidia-smi
python -u -m scripts.check_gpu_stack
python -u -m training.train_ppo \
  --estimator artifacts/k_estimator.joblib \
  --episodes 12 \
  --budget 4 \
  --rollout-batch 4 \
  --ppo-epochs 2 \
  --execute
