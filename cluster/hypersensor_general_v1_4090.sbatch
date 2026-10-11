#!/bin/bash
#SBATCH --job-name=hypergen-v1
#SBATCH --partition=gpu
#SBATCH --constraint=rtx_4090
#SBATCH --gres=gpu:1
#SBATCH --exclude=cs-4090-09
#SBATCH --mem=24G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/hypergen-v1-%j.out
#SBATCH --error=logs/hypergen-v1-%j.err

set -euo pipefail

cd "$HOME/mas_rl_detective"

PY="/home/wieseld/.conda/envs/masdetective/bin/python"

export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "===== NODE ====="
hostname

echo
echo "===== GPU ====="
nvidia-smi

echo
echo "===== PYTHON ====="
"$PY" - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
    print(
        "VRAM_GB:",
        torch.cuda.get_device_properties(0).total_memory / 1024**3
    )
PY

echo
echo "===== RUN ====="

"$PY" -u analysis/train_hypersensor_general_v1.py
