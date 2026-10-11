#!/usr/bin/env bash
#SBATCH --job-name=mas5-v53-r2-sft
#SBATCH --partition=main
#SBATCH --gres=gpu:rtx_4090:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=40G
#SBATCH --time=20:00:00
#SBATCH --output=logs/v53-%x-%j.out
#SBATCH --error=logs/v53-%x-%j.err
set -euo pipefail
cd /home/wieseld/mas_rl_detective
export PYTHONPATH="$PWD:${PYTHONPATH:-}" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
exec "$HOME/.conda/envs/masdetective/bin/python" -u \
  -m analysis.role_direct_within_v4.v53_r2_sft train --root "$V53_ROOT" --epochs 2
