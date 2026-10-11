#!/bin/bash
#SBATCH --job-name=q38-r7-ab
#SBATCH --partition=rtx_pro_6000
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=00:45:00
#SBATCH --output=logs/q38-r7-ab-%j.out
#SBATCH --error=logs/q38-r7-ab-%j.err

set -euo pipefail

cd "$HOME/mas_rl_detective"

export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false

PY="$HOME/.venvs/masdetective-q38/bin/python"

echo "===== GPU ====="
nvidia-smi --query-gpu=name,memory.total,memory.free \
    --format=csv,noheader

echo
echo "===== ENV ====="
"$PY" - <<'PY'
import torch, transformers, peft
print("torch =", torch.__version__)
print("CUDA =", torch.version.cuda)
print("transformers =", transformers.__version__)
print("peft =", peft.__version__)
print("GPU =", torch.cuda.get_device_name(0))
PY

echo
echo "===== A/B ====="

"$PY" -u analysis/q38_vs_r7_generator_ab_v1.py

echo
echo "===== DONE ====="
