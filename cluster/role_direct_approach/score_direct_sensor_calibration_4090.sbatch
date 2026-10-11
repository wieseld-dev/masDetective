#!/bin/bash
#SBATCH --job-name=rd-cal
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx_4090:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=40G
#SBATCH --time=20:00:00
#SBATCH --output=logs/rd-cal-%A-%a.out
#SBATCH --error=logs/rd-cal-%A-%a.err

set -euo pipefail

PROJECT="$HOME/mas_rl_detective"
PY="$HOME/.conda/envs/masdetective/bin/python"

cd "$PROJECT"

export PYTHONPATH="$PROJECT:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

FOLD="$SLURM_ARRAY_TASK_ID"

echo "CALIBRATION FOLD=$FOLD"

"$PY" -u \
  analysis/role_direct_approach/score_direct_sensor_trainfold_v1.py \
  --fold "$FOLD" \
  --max-length 1024
