#!/bin/bash
#SBATCH --partition main
#SBATCH --time 0-01:00:00
#SBATCH --job-name mas-ppo-smoke
#SBATCH --output logs/mas-ppo-smoke-%J.out
#SBATCH --gpus=gtx_1080:1

set -euo pipefail

cd "$HOME/mas_rl_detective"
mkdir -p logs artifacts outputs

PY="$HOME/.conda/envs/masdetective/bin/python"

echo "Python:"
"$PY" --version
echo "Interpreter:"
"$PY" -c 'import sys; print(sys.executable)'

echo "Torch check:"
"$PY" - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
PY

set -a
source .env
set +a

"$PY" -u -m training.train_ppo \
  --estimator artifacts/k_estimator.joblib \
  --episodes 2 \
  --budget 2 \
  --rollout-batch 2 \
  --ppo-epochs 2 \
  --execute
