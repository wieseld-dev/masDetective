#!/bin/bash
#SBATCH --partition=main
#SBATCH --gpus=rtx_3090:1
#SBATCH --time=0-00:10:00
#SBATCH --job-name=genv125
#SBATCH --output=logs/genv125-%J.out
#SBATCH --error=logs/genv125-%J.err

set -euo pipefail

cd "$HOME/mas_rl_detective"

PY="/home/wieseld/.conda/envs/masdetective/bin/python"

export PYTHONPATH="$PWD:${PYTHONPATH:-}"

echo "===== ENV ====="

"$PY" - <<'PY'
import torch
import sys

print(
    "python:",
    sys.executable,
)

print(
    "CUDA:",
    torch.cuda.is_available(),
)

if not torch.cuda.is_available():
    raise SystemExit(
        "CUDA unavailable"
    )

print(
    "GPU:",
    torch.cuda.get_device_name(0),
)
PY

rm -rf \
  artifacts/generator_v12_5/*

"$PY" -u \
  analysis/generator_v12_5_elicitation_bank.py
