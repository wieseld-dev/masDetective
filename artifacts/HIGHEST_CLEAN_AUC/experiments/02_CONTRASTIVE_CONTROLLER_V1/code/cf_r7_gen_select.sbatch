#!/bin/bash
#SBATCH --job-name=contrast-cf-r7
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --constraint=rtx_4090
#SBATCH --exclude=cs-4090-09
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=10:00:00
#SBATCH --output=logs/contrast-cf-r7-%j.out
#SBATCH --error=logs/contrast-cf-r7-%j.err

set -euo pipefail

cd "$HOME/mas_rl_detective"

export PYTHONPATH="$HOME/mas_rl_detective:${PYTHONPATH:-}"
export PATH="$HOME/.conda/envs/masdetective/bin:$PATH"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

PY="$HOME/.conda/envs/masdetective/bin/python"

HCA="artifacts/HIGHEST_CLEAN_AUC"
C1="$HCA/experiments/02_CONTRASTIVE_CONTROLLER_V1"
ROOT="$C1/data/cf_from_hybrid_r6"

echo "===== CONTRASTIVE GENERATOR ====="

"$PY" -u \
  "$C1/code/generate_contrastive_v1.py" \
  --round 7 \
  --split dev \
  --group-size 4 \
  --root "$ROOT" \
  --adapter "$HCA/models/generator_clean_r7" \
  --v11-adapter "$HCA/models/generator_v11_base" \
  --temperature 0.90 \
  --top-p 0.95 \
  --max-new-tokens 64 \
  --max-attempts 8

echo
echo "===== FROZEN SELECTOR V2 ====="

"$PY" -u \
  analysis/select_v14_candidates_llmv2.py \
  --root "$ROOT" \
  --round 7 \
  --split dev \
  --adapter "$HCA/models/selector_v2"

echo
echo "DONE"
