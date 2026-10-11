#!/bin/bash
#SBATCH --job-name=q38-env
#SBATCH --partition=main
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/q38-env-%j.out
#SBATCH --error=logs/q38-env-%j.err

set -euo pipefail

BASEPY="$HOME/.conda/envs/masdetective/bin/python"
ENV="$HOME/.venvs/masdetective-q38"

rm -rf "$ENV"
mkdir -p "$HOME/.venvs"

"$BASEPY" -m venv "$ENV"

PY="$ENV/bin/python"

"$PY" -m pip install -U pip setuptools wheel

# Blackwell-compatible PyTorch
"$PY" -m pip install \
    torch==2.14.0+cu132 \
    torchvision==0.29.0+cu132 \
    --index-url https://download.pytorch.org/whl/cu132 \
    --extra-index-url https://pypi.org/simple

# Minimal Qwen3.8 / PEFT stack
"$PY" -m pip install \
    transformers==5.16.1 \
    peft==0.20.0 \
    accelerate==1.14.0 \
    pillow \
    safetensors

echo
echo "===== Q38 ENV READY ====="

"$PY" - <<'PY'
import torch, transformers, peft, accelerate
print("torch        =", torch.__version__)
print("torch CUDA   =", torch.version.cuda)
print("transformers =", transformers.__version__)
print("peft         =", peft.__version__)
print("accelerate   =", accelerate.__version__)
PY
