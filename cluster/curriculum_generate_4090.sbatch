#!/bin/bash
#SBATCH --job-name=cur-gen
#SBATCH --gpus=1
#SBATCH --constraint=rtx_4090
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=0-03:00:00
#SBATCH --output=logs/cur-gen-%j.out
#SBATCH --error=logs/cur-gen-%j.err
set -euo pipefail
cd "$HOME/mas_rl_detective"
export PYTHONPATH="$HOME/mas_rl_detective:${PYTHONPATH:-}"
export PATH="$HOME/.conda/envs/masdetective/bin:$PATH"
PY="$HOME/.conda/envs/masdetective/bin/python"
"$PY" -u scripts/curriculum_generate.py \
  --round "$CURR_ROUND" --split "$CURR_SPLIT" --root "$CURR_ROOT" \
  --adapter "$CURR_ADAPTER" --v11-reference "${CURR_V11:-artifacts/v11_policy}" \
  --group-size "${CURR_GROUP_SIZE:-6}" --max-seq-length "${CURR_MAX_SEQ_LENGTH:-4096}" \
  --max-new-tokens "${CURR_MAX_NEW_TOKENS:-192}" --local-retries "${CURR_LOCAL_RETRIES:-12}" \
  --temperature "${CURR_TEMPERATURE:-0.85}" --top-p "${CURR_TOP_P:-0.95}"
