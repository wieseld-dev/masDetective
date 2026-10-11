#!/bin/bash
#SBATCH --job-name=cur-v2-gen
#SBATCH --gpus=1
#SBATCH --constraint=rtx_4090
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=0-03:00:00
#SBATCH --output=logs/cur-v2-gen-%j.out
#SBATCH --error=logs/cur-v2-gen-%j.err
set -euo pipefail
cd "$HOME/mas_rl_detective"
export PYTHONPATH="$HOME/mas_rl_detective:${PYTHONPATH:-}"
export PATH="$HOME/.conda/envs/masdetective/bin:$PATH"
PY="$HOME/.conda/envs/masdetective/bin/python"
"$PY" -u scripts/curriculum_v2_generate.py --round "$CURR_ROUND" --split "$CURR_SPLIT" --root "$CURR_ROOT" --adapter "$CURR_ADAPTER" --v11-reference "${CURR_V11:-artifacts/v11_policy}" --group-size "${CURR_GROUP_SIZE:-6}" --local-retries "${CURR_LOCAL_RETRIES:-12}" --selector-model "$CURR_SELECTOR_MODEL" --selector-epsilon "${CURR_SELECTOR_EPS:-0.20}" --selector-min-samples "${CURR_SELECTOR_MIN_SAMPLES:-96}" --stage "${CURR_STAGE:-1}"
