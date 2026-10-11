#!/bin/bash
#SBATCH --partition=main
#SBATCH --gpus=rtx_3090:1
#SBATCH --time=0-00:12:00
#SBATCH --job-name=genv126
#SBATCH --output=logs/genv126-%J.out
#SBATCH --error=logs/genv126-%J.err

set -euo pipefail

cd "$HOME/mas_rl_detective"

PY="/home/wieseld/.conda/envs/masdetective/bin/python"

export PYTHONPATH="$PWD:${PYTHONPATH:-}"

echo "===== ENV ====="

"$PY" - <<'PY'
import sys
import torch

print(
    "python:",
    sys.executable,
)

print(
    "torch:",
    torch.__version__,
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

rm -f \
  artifacts/generator_v12_6/all_candidates.jsonl \
  artifacts/generator_v12_6/selected_bank.jsonl

"$PY" -u \
  analysis/generator_v12_6_elicitation_only.py
