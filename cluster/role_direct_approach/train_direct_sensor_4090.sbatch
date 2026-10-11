#!/bin/bash
#SBATCH --job-name=rd-sens
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx_4090:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=20:00:00
#SBATCH --output=logs/rd-sens-%A-%a.out
#SBATCH --error=logs/rd-sens-%A-%a.err

set -euo pipefail

PROJECT="$HOME/mas_rl_detective"
PY="$HOME/.conda/envs/masdetective/bin/python"

cd "$PROJECT"

export PYTHONPATH="$PROJECT:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

FOLD="${SLURM_ARRAY_TASK_ID}"

echo "FOLD=$FOLD"

nvidia-smi \
  --query-gpu=name,memory.total,memory.free \
  --format=csv,noheader

"$PY" -u \
  analysis/role_direct_approach/train_role_direct_sensor_v1.py \
  --fold "$FOLD" \
  --epochs 2 \
  --lr 1e-4 \
  --batch-pairs 2 \
  --grad-accum 4 \
  --max-length 1024 \
  --pair-weight 0.5
