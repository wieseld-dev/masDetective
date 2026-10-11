#!/bin/bash
#SBATCH --job-name=hyb-v2-gensel
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --constraint=rtx_4090
#SBATCH --exclude=cs-4090-09
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=10:00:00
#SBATCH --output=logs/hyb-v2-gensel-%j.out
#SBATCH --error=logs/hyb-v2-gensel-%j.err

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

HCA="artifacts/HIGHEST_CLEAN_AUC"
H2="$HCA/experiments/01_GENERATOR_HYBRID_V2"

echo "===== HYBRID GENERATOR ====="

"$PY" -u \
  "$H2/code/generate_hybrid_v2.py" \
  --round "$V14A_ROUND" \
  --split dev \
  --group-size 4 \
  --root "$V14A_ROOT" \
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
  --root "$V14A_ROOT" \
  --round "$V14A_ROUND" \
  --split dev \
  --adapter "$HCA/models/selector_v2"

echo
echo "HYBRID GENERATOR+SELECTOR DONE"
