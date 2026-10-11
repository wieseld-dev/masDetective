#!/bin/bash
#SBATCH --partition=main
#SBATCH --gpus=rtx_3090:1
#SBATCH --time=0-00:30:00
#SBATCH --job-name=belief5r
#SBATCH --output=logs/belief5r-%J.out
#SBATCH --error=logs/belief5r-%J.err

set -euo pipefail

cd "$HOME/mas_rl_detective"

PY="/home/wieseld/.conda/envs/masdetective/bin/python"

export PYTHONPATH="$PWD:${PYTHONPATH:-}"

if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

if [ -z "${OPENAI_API_KEY:-}" ]; then
    echo "OPENAI_API_KEY missing"
    exit 1
fi

echo "===== ENV ====="

"$PY" - <<'PY'
import sys
import torch

print("python:", sys.executable)
print("torch:", torch.__version__)
print("CUDA:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY

echo
echo "===== RUN FIVE-SYSTEM MATCHED REPRO ====="

"$PY" -u \
  analysis/five_system_belief_auc_control_repro_v2.py

echo
echo "===== FINAL SUMMARY ====="

cat \
  artifacts/five_system_belief_auc_control_repro_v2/summary.json
