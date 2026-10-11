#!/bin/bash
#SBATCH --job-name=q38-sel-v4
#SBATCH --partition=rtx_pro_6000
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=20:00:00
#SBATCH --output=logs/q38-sel-v4-%j.out
#SBATCH --error=logs/q38-sel-v4-%j.err

set -euo pipefail

cd "$HOME/mas_rl_detective"

export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export DATA_ROOT="artifacts/q38_selector_v4_listwise_data_v1"
export OUT_DIR="artifacts/q38_selector_v4_listwise_v1"

# Serious overnight run.
export MAX_LEN=1024
export EPOCHS=4
export MAX_MICRO=15000
export GRAD_ACC=8
export LR=8e-6
export PATIENCE=3
export EVAL_BATCH=2

PY="$HOME/.venvs/masdetective-q38/bin/python"

echo "===== BLACKWELL PREFLIGHT ====="
nvidia-smi

"$PY" - <<'PY'
import torch

name = torch.cuda.get_device_name(0)
cc = torch.cuda.get_device_capability(0)
mem = (
    torch.cuda.get_device_properties(0)
    .total_memory
    / 1024**3
)

print("GPU =", name)
print("CC  =", cc)
print("VRAM GiB =", round(mem, 2))

if cc != (12, 0) or mem < 80:
    raise SystemExit(
        "REFUSING: Q38 selector needs 96GB Blackwell"
    )

print("BLACKWELL_RESOURCE_OK")
PY

"$PY" -u \
  analysis/train_q38_selector_v4_listwise.py
