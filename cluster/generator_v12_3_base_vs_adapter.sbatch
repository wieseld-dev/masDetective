#!/bin/bash
#SBATCH --partition=main
#SBATCH --gpus=rtx_3090:1
#SBATCH --time=0-00:10:00
#SBATCH --job-name=genv123
#SBATCH --output=logs/genv123-%J.out
#SBATCH --error=logs/genv123-%J.err

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

rm -rf \
  artifacts/generator_v12_3/base_vs_adapter

"$PY" -u \
  analysis/generator_v12_3_base_vs_adapter.py \
  --replicates 2 \
  --temperature 0.85 \
  --top-p 0.95 \
  --max-new-tokens 224

echo
echo "===== SUMMARY FILE ====="

cat \
  artifacts/generator_v12_3/base_vs_adapter/summary.json
