#!/bin/bash
#SBATCH --job-name=cur-upd
#SBATCH --gpus=1
#SBATCH --constraint=rtx_4090
#SBATCH --mem=48G
#SBATCH --cpus-per-task=6
#SBATCH --time=0-05:00:00
#SBATCH --output=logs/cur-upd-%j.out
#SBATCH --error=logs/cur-upd-%j.err
set -euo pipefail
cd "$HOME/mas_rl_detective"
export PYTHONPATH="$HOME/mas_rl_detective:${PYTHONPATH:-}"
export PATH="$HOME/.conda/envs/masdetective/bin:$PATH"
PY="$HOME/.conda/envs/masdetective/bin/python"
ARGS=(
  --root "$CURR_ROOT" --horizon "$CURR_HORIZON" --input-adapter "$CURR_INPUT_ADAPTER" --output-dir "$CURR_OUTPUT_ADAPTER"
  --v11-reference "${CURR_V11:-artifacts/v11_policy}" --v11-hash-file "$CURR_V11_HASH_FILE" --retention "$CURR_RETENTION"
  --lr "$CURR_LR" --kl-beta "$CURR_KL_BETA" --retention-weight "$CURR_RETENTION_WEIGHT" --mc-weight "$CURR_MC_WEIGHT"
  --gamma "${CURR_GAMMA:-1.0}" --epochs "${CURR_EPOCHS:-1}" --grad-accum "${CURR_GRAD_ACCUM:-8}"
)
if [[ -n "${CURR_OPTIMIZER_IN:-}" && -f "$CURR_OPTIMIZER_IN" ]]; then ARGS+=(--optimizer-in "$CURR_OPTIMIZER_IN"); fi
"$PY" -u scripts/curriculum_update.py "${ARGS[@]}"
