#!/bin/bash
#SBATCH --job-name=q38-sel41
#SBATCH --partition=rtx_pro_6000
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=20:00:00
#SBATCH --output=logs/q38-sel41-%j.out
#SBATCH --error=logs/q38-sel41-%j.err

set -euo pipefail

cd "$HOME/mas_rl_detective"

export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export DATA_ROOT="artifacts/q38_selector_v41_data"
export OUT_DIR="artifacts/q38_selector_v41"

export MAX_LEN=1024

# ~12-16h expected from today's measured throughput.
export EPOCHS=3
export MAX_MICRO=18000

export PERMS_PER_STATE=4
export EVAL_PERMS=3

export GRAD_ACC=8
export LR=6e-6

export TARGET_TEMP=2.0
export PAIR_WEIGHT=0.50

export PATIENCE=2

PY="$HOME/.venvs/masdetective-q38/bin/python"

echo "===== BLACKWELL PREFLIGHT ====="

nvidia-smi

"$PY" - <<'PY'
import torch

name = torch.cuda.get_device_name(0)

cc = torch.cuda.get_device_capability(0)

mem = (
    torch.cuda
    .get_device_properties(0)
    .total_memory
    / 1024**3
)

print("GPU =", name)
print("CAPABILITY =", cc)
print("VRAM GiB =", round(mem, 2))

if cc != (12, 0):
    raise SystemExit(
        "REFUSING: expected Blackwell CC 12.0"
    )

if mem < 80:
    raise SystemExit(
        "REFUSING: expected ~96GB GPU"
    )

print(
    "BLACKWELL_RESOURCE_OK"
)
PY

"$PY" -u \
  analysis/train_q38_selector_v41.py
