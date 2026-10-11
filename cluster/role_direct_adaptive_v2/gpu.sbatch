#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx_4090:1
#SBATCH --job-name=rda2-gpu
#SBATCH --cpus-per-task=6
#SBATCH --mem=40G
#SBATCH --time=20:00:00
#SBATCH --output=logs/rda2-%x-%j.out
#SBATCH --error=logs/rda2-%x-%j.err
set -euo pipefail
cd /home/wieseld/mas_rl_detective
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
exec /home/wieseld/.conda/envs/masdetective/bin/python -u \
 analysis/role_direct_adaptive_v2/attack.py "$ROLE_STAGE" --round "$ROLE_ROUND" --root "$ROLE_ROOT" --manifest "$ROLE_MANIFEST"
