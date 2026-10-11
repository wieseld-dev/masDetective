#!/bin/bash
#SBATCH --job-name=q38-lora
#SBATCH --partition=rtx_pro_6000
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=10:00:00
#SBATCH --output=logs/q38-lora-%j.out
#SBATCH --error=logs/q38-lora-%j.err

set -euo pipefail

cd "$HOME/mas_rl_detective"

export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export MAX_LEN=1024
export MAX_STEPS=0
export GRAD_ACC=8
export EPOCHS=1
export LR=2e-5

export OUT_DIR="artifacts/q38_utility_lora_v1/policy"

PY="$HOME/.venvs/masdetective-q38/bin/python"

echo "===== GPU PREFLIGHT ====="
nvidia-smi

"$PY" - <<'PY'
import torch

name = torch.cuda.get_device_name(0)
cc = torch.cuda.get_device_capability(0)
mem = torch.cuda.get_device_properties(0).total_memory / 1024**3

print("GPU        =", name)
print("capability =", cc)
print("VRAM GiB   =", round(mem, 2))

if cc != (12, 0) or mem < 80:
    raise SystemExit("REFUSING TRAINING: not a 96GB Blackwell")

print("BLACKWELL_RESOURCE_OK")
PY

"$PY" -u analysis/train_q38_utility_lora_v1.py
