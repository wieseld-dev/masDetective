#!/bin/bash
#SBATCH --job-name=kb-gen
#SBATCH --partition=main
#SBATCH --gpus=1
#SBATCH --constraint=rtx_4090
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/kb-gen-%j.out
#SBATCH --error=logs/kb-gen-%j.err
set -euo pipefail
cd "$HOME/mas_rl_detective"
export PYTHONPATH="$HOME/mas_rl_detective:${PYTHONPATH:-}"
export PATH="$HOME/.conda/envs/masdetective/bin:$PATH"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
PY="$HOME/.conda/envs/masdetective/bin/python"
: "${KBOOT_ROOT:?set KBOOT_ROOT}"
: "${KBOOT_ROUND:?set KBOOT_ROUND}"
: "${KBOOT_SPLIT:?set KBOOT_SPLIT}"
"$PY" -u scripts/kboot_generate_round.py \
  --round "$KBOOT_ROUND" \
  --split "$KBOOT_SPLIT" \
  --root "$KBOOT_ROOT" \
  --adapter "${KBOOT_ADAPTER:-artifacts/v11_policy}" \
  --v11-adapter artifacts/v11_policy \
  --max-seq-length "${KBOOT_MAX_SEQ_LENGTH:-2048}" \
  --max-new-tokens "${KBOOT_MAX_NEW_TOKENS:-96}" \
  --local-retries "${KBOOT_LOCAL_RETRIES:-6}" \
  --no-4bit
