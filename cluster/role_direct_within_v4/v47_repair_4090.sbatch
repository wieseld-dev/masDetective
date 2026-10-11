#!/usr/bin/env bash
#SBATCH --job-name=mas5-v47-repair
#SBATCH --partition=main
#SBATCH --gres=gpu:rtx_4090:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=40G
#SBATCH --time=20:00:00
#SBATCH --output=logs/v47-%x-%j.out
#SBATCH --error=logs/v47-%x-%j.err
set -euo pipefail
cd /home/wieseld/mas_rl_detective
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
ARGS=()
if [[ "${V47_RETRY:-0}" == 1 ]]; then ARGS=(--retry); fi
exec "$HOME/.conda/envs/masdetective/bin/python" -u \
    analysis/role_direct_within_v4/v47_repair.py --root "$V46_ROOT" "${ARGS[@]}"
