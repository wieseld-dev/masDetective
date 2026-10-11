#!/bin/bash
#SBATCH --partition=main
#SBATCH --gres=gpu:rtx_4090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --output=logs/v4s-%x-%j.out
#SBATCH --error=logs/v4s-%x-%j.err
set -euo pipefail
cd /home/wieseld/mas_rl_detective
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
"$HOME/.conda/envs/masdetective/bin/python" -u analysis/role_direct_within_v4/pipeline.py "$V4_STAGE" --root "$V4_ROOT"
