#!/usr/bin/env bash
#SBATCH --job-name=mas5-v52-r2
#SBATCH --partition=main
#SBATCH --gres=gpu:rtx_4090:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=40G
#SBATCH --time=20:00:00
#SBATCH --output=logs/v52-%x-%j.out
#SBATCH --error=logs/v52-%x-%j.err
set -euo pipefail
cd /home/wieseld/mas_rl_detective
export PYTHONPATH="$PWD:${PYTHONPATH:-}" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
PY="$HOME/.conda/envs/masdetective/bin/python"
"$PY" -u -m analysis.role_direct_within_v4.v52_adaptive_short_r2 generate --root "$V51_ROOT" --tries "${V52_TRIES:-8}"
"$PY" -u -m analysis.role_direct_within_v4.v52_adaptive_short_r2 audit --root "$V51_ROOT"
