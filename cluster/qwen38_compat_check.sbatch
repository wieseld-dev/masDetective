#!/bin/bash
#SBATCH --job-name=q38-compat
#SBATCH --partition=main
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=logs/q38-compat-%j.out
#SBATCH --error=logs/q38-compat-%j.err

set -euo pipefail

PYTHON=/home/wieseld/.conda/envs/masdetective/bin/python

echo "===== PACKAGE VERSIONS ====="
$PYTHON - <<'PY'
import torch, transformers, peft, accelerate
print("torch       =", torch.__version__)
print("transformers=", transformers.__version__)
print("peft        =", peft.__version__)
print("accelerate  =", accelerate.__version__)
PY

echo
echo "===== QWEN3.8 CONFIG TEST ====="

$PYTHON - <<'PY'
from transformers import AutoConfig

name = "Qwen/Qwen3.8-27B"
cfg = AutoConfig.from_pretrained(name)

print("model_type    =", cfg.model_type)
print("architectures =", cfg.architectures)
print("dtype         =", getattr(cfg, "dtype", None))
print("CONFIG_OK")
PY
