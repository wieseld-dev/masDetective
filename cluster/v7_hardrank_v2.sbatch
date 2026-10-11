#!/bin/bash
#SBATCH --partition=main
#SBATCH --gpus=rtx_4090:1
#SBATCH --time=0-11:00:00
#SBATCH --job-name=v7hard-v2
#SBATCH --output=logs/v7hard-v2-%J.out
#SBATCH --error=logs/v7hard-v2-%J.err
#SBATCH --signal=B:USR1@180

set -euo pipefail

# Reduce allocator fragmentation and avoid unnecessary tokenizer threads.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

cd "$HOME/mas_rl_detective"

echo "============================================================"
echo "V7 HARD-RANK NIGHT TRAINING"
echo "============================================================"
date
hostname
echo "JOB=$SLURM_JOB_ID"

module load anaconda
source activate masdetective

echo "PYTHON=$(which python)"
python --version

if [[ "$(which python)" != "$HOME/.conda/envs/masdetective/bin/python" ]]; then
    echo "ERROR: wrong Python environment"
    exit 20
fi

echo "NOTE: this training job does NOT use the OpenAI API."

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("CUDA unavailable on allocated node")
print("gpu:", torch.cuda.get_device_name(0))
PY

echo
echo "===== PY_COMPILE ====="
python -m py_compile \
    analysis/train_v7_hardrank_night.py

echo
echo "===== TRAINING SMOKE ====="
rm -rf artifacts/system_v2_rankfirst/v7_hardrank_v2_smoke

python analysis/train_v7_hardrank_night.py \
    --smoke \
    --max-length 384 \
    --negatives-per-row 4 \
    --out artifacts/system_v2_rankfirst/v7_hardrank_v2_smoke

echo
echo "===== FULL TRAINING ====="
python -u analysis/train_v7_hardrank_night.py \
    --epochs 1 \
    --train-rows-per-epoch 5000 \
    --negatives-per-row 4 \
    --max-length 384 \
    --lr 2e-6 \
    --grad-accum 8 \
    --margin 1.0 \
    --bce-weight 0.02 \
    --worst-weight 0.25 \
    --dev-limit 512 \
    --eval-batch-size 8 \
    --checkpoint-every 150 \
    --log-every 20 \
    --seed 20260926 \
    --out artifacts/system_v2_rankfirst/v7_hardrank_v2

echo
echo "===== TRAINING SUMMARY ====="
cat artifacts/system_v2_rankfirst/v7_hardrank_v2/summary.json

date
