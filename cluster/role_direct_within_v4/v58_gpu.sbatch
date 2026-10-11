#!/usr/bin/env bash
#SBATCH --partition=main
#SBATCH --gres=gpu:rtx_4090:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=40G
#SBATCH --time=20:00:00
#SBATCH --output=logs/v58-%x-%j.out
#SBATCH --error=logs/v58-%x-%j.err
set -euo pipefail
cd /home/wieseld/mas_rl_detective
export PYTHONPATH="$PWD:${PYTHONPATH:-}" PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
: "${V58_ROOT:?missing run root}"
: "${V58_STAGE:?missing stage}"
: "${V58_ROUND:?missing round}"
exec "$HOME/.conda/envs/masdetective/bin/python" -u -m analysis.role_direct_within_v4.v58_adaptive_eight \
  "$V58_STAGE" --root "$V58_ROOT" --round "$V58_ROUND"
