#!/bin/bash
#SBATCH --job-name=s5-r7-gen
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=rtx_4090
#SBATCH --cpus-per-task=6
#SBATCH --mem=40G
#SBATCH --time=20:00:00
#SBATCH --output=logs/s5-r7-gen-%j.out
#SBATCH --error=logs/s5-r7-gen-%j.err

set -euo pipefail

cd "$HOME/mas_rl_detective"

export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

PY="$HOME/.conda/envs/masdetective/bin/python"

: "${S5_ROOT:?}"
: "${S5_ROUND:?}"

GEN="artifacts/v14_adaptive_role_v1/checkpoints/round_07/policy"

test -s "$GEN/adapter_config.json"

echo "ROOT  = $S5_ROOT"
echo "ROUND = $S5_ROUND"
echo "GEN   = $GEN"
echo "G     = 8"

"$PY" -u \
  scripts/v14_adaptive_generate_round_v1.py \
  --round "$S5_ROUND" \
  --split dev \
  --group-size 8 \
  --root "$S5_ROOT" \
  --adapter "$GEN" \
  --v11-adapter artifacts/v14_benign_policy \
  --temperature 0.85 \
  --top-p 0.95 \
  --max-new-tokens 128 \
  --max-seq-length 3072
