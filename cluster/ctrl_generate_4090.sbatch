#!/bin/bash
#SBATCH --job-name=ctrl-gen
#SBATCH --partition=main
#SBATCH --gpus=1
#SBATCH --constraint=rtx_4090
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/ctrl-gen-%j.out
#SBATCH --error=logs/ctrl-gen-%j.err
set -euo pipefail
cd "$HOME/mas_rl_detective"
export PYTHONPATH="$HOME/mas_rl_detective:${PYTHONPATH:-}"
export PATH="$HOME/.conda/envs/masdetective/bin:$PATH"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
PY="$HOME/.conda/envs/masdetective/bin/python"
: "${CTRL_ROOT:?set CTRL_ROOT}"
: "${CTRL_ROUND:?set CTRL_ROUND}"
: "${CTRL_SPLIT:?set CTRL_SPLIT}"
: "${CTRL_MODE:?set CTRL_MODE}"
"$PY" -u scripts/av11_controller_generate.py \
  --round "$CTRL_ROUND" \
  --split "$CTRL_SPLIT" \
  --root "$CTRL_ROOT" \
  --controller "$CTRL_MODE" \
  --adapter "${CTRL_ADAPTER:-artifacts/v11_policy}" \
  --max-seq-length "${CTRL_MAX_SEQ_LENGTH:-4096}" \
  --max-new-tokens "${CTRL_MAX_NEW_TOKENS:-192}" \
  --local-retries "${CTRL_LOCAL_RETRIES:-20}" \
  --no-4bit
