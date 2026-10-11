#!/bin/bash
#SBATCH --job-name=cur-ret
#SBATCH --gpus=1
#SBATCH --constraint=rtx_4090
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=0-02:00:00
#SBATCH --output=logs/cur-ret-%j.out
#SBATCH --error=logs/cur-ret-%j.err
set -euo pipefail
cd "$HOME/mas_rl_detective"
export PYTHONPATH="$HOME/mas_rl_detective:${PYTHONPATH:-}"
export PATH="$HOME/.conda/envs/masdetective/bin:$PATH"
PY="$HOME/.conda/envs/masdetective/bin/python"
"$PY" -u scripts/curriculum_build_retention.py \
  --adapter "${CURR_V11:-artifacts/v11_policy}" \
  --output "${CURR_RETENTION:-artifacts/adaptive_v11_curriculum_v1/v11_retention.jsonl}" \
  --variants "${CURR_RETENTION_VARIANTS:-3}"
