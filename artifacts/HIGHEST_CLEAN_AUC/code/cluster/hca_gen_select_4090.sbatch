#!/bin/bash
#SBATCH --job-name=s5-r7-sel
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --constraint=rtx_4090
#SBATCH --exclude=cs-4090-09
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=10:00:00
#SBATCH --output=logs/s5-r7-sel-%j.out
#SBATCH --error=logs/s5-r7-sel-%j.err

set -euo pipefail

cd "$HOME/mas_rl_detective"

export PYTHONPATH="$HOME/mas_rl_detective:${PYTHONPATH:-}"
export PATH="$HOME/.conda/envs/masdetective/bin:$PATH"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

PY="$HOME/.conda/envs/masdetective/bin/python"

: "${V14A_ROOT:?}"
: "${V14A_ROUND:?}"

CLEAN_GEN="artifacts/HIGHEST_CLEAN_AUC/models/generator_clean_r7"
SELECTOR="artifacts/HIGHEST_CLEAN_AUC/models/selector_v2"

test -s "$CLEAN_GEN/adapter_config.json"
test -s "$SELECTOR/adapter_config.json"

echo "===== GENERATOR ====="
echo "$CLEAN_GEN"

"$PY" scripts/v14_adaptive_generate_round_v1.py \
  --round "$V14A_ROUND" \
  --split dev \
  --group-size 4 \
  --root "$V14A_ROOT" \
  --adapter "$CLEAN_GEN" \
  --v11-adapter artifacts/HIGHEST_CLEAN_AUC/models/generator_v11_base \
  --temperature 0.85 \
  --top-p 0.95 \
  --max-new-tokens 128 \
  --max-seq-length 3072

echo
echo "===== SELECTOR ====="

"$PY" analysis/select_v14_candidates_llmv2.py \
  --root "$V14A_ROOT" \
  --round "$V14A_ROUND" \
  --split dev \
  --adapter "$SELECTOR"

echo
echo "GENERATOR+SELECTOR DONE"
