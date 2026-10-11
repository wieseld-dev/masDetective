#!/bin/bash
#SBATCH --job-name=gensens-v1
#SBATCH --partition=gpu
#SBATCH --constraint=rtx_4090
#SBATCH --gres=gpu:1
#SBATCH --exclude=cs-4090-09
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/gensens-v1-%j.out
#SBATCH --error=logs/gensens-v1-%j.err

set -euo pipefail

cd "$HOME/mas_rl_detective"

PY=/home/wieseld/.conda/envs/masdetective/bin/python

export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false

echo "======================================================================"
echo "GENERAL ROLE SENSOR V1"
echo "======================================================================"
echo "TRAIN: 3240 rows / 1620 PLUS-MINUS pairs / 60 roles"
echo "VAL  :  648 rows /  324 PLUS-MINUS pairs / 12 unseen roles"
echo "MODEL: Qwen2.5-3B-Instruct + QLoRA"
echo "LOSS : BCE + pairwise PLUS > MINUS"
echo "INTERNAL TEST: NOT USED"
echo "======================================================================"

"$PY" -u \
  analysis/train_general_role_sensor_v1.py
