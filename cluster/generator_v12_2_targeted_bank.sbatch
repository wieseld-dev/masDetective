#!/bin/bash
#SBATCH --partition=main
#SBATCH --gpus=rtx_3090:1
#SBATCH --time=0-00:12:00
#SBATCH --job-name=genv122
#SBATCH --output=logs/genv122-%J.out
#SBATCH --error=logs/genv122-%J.err

set -euo pipefail

cd "$HOME/mas_rl_detective"

PY="/home/wieseld/.conda/envs/masdetective/bin/python"

export PYTHONPATH="$PWD:${PYTHONPATH:-}"

echo "===== ENV ====="
echo "PY=$PY"
"$PY" --version

"$PY" - <<'PY'
import sys
import torch

print("sys.executable:", sys.executable)
print("torch:", torch.__version__)
print("CUDA:", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise SystemExit("CUDA NOT AVAILABLE")

print("GPU:", torch.cuda.get_device_name(0))
PY

rm -f \
  artifacts/generator_v12_2/targeted_probe_bank.json

"$PY" -u \
  analysis/generator_v12_2_targeted_bank.py \
  --adapter artifacts/robust_visible_generator_v1_epoch2/policy \
  --temperature 0.85 \
  --top-p 0.95 \
  --max-new-tokens 224 \
  --retries 2

echo
echo "===== TARGETED BANK SAVED ====="

test -f artifacts/generator_v12_2/targeted_probe_bank.json

echo
echo "===== DONE ====="
