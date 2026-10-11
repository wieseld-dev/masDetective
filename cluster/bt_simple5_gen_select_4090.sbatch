#!/bin/bash
#SBATCH --job-name=bt-s5-gen
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx_4090:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=20:00:00
#SBATCH --output=logs/bt-s5-gen-%j.out
#SBATCH --error=logs/bt-s5-gen-%j.err

set -euo pipefail

cd "$HOME/mas_rl_detective"

export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export PATH="$HOME/.conda/envs/masdetective/bin:$PATH"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

PY="$HOME/.conda/envs/masdetective/bin/python"

: "${S5_ROOT:?}"
: "${S5_ROUND:?}"

GEN="artifacts/v14_adaptive_role_v1/checkpoints/round_07/policy"
SEL="artifacts/belief_targeted_selector_v1"

"$PY" -u analysis/bt_simple5_runtime_guard_v1.py \
  --root "$S5_ROOT" \
  --round "$S5_ROUND" \
  --phase pre

test ! -e \
  "$S5_ROOT/candidate_pool_dev_round_$(printf '%02d' "$S5_ROUND").jsonl"

test ! -e \
  "$S5_ROOT/responses_dev_round_$(printf '%02d' "$S5_ROUND").jsonl"

echo "===== GENERATE ROUND $S5_ROUND ====="

"$PY" -u scripts/v14_adaptive_generate_round_v1.py \
  --round "$S5_ROUND" \
  --split dev \
  --group-size 8 \
  --root "$S5_ROOT" \
  --adapter "$GEN" \
  --v11-adapter artifacts/v14_benign_policy \
  --temperature 0.85 \
  --top-p 0.95 \
  --max-new-tokens 128 \
  --max-seq-length 3072 \
  --seed 722031

"$PY" -u analysis/bt_simple5_runtime_guard_v1.py \
  --root "$S5_ROOT" \
  --round "$S5_ROUND" \
  --phase post_gen

R=$(printf '%02d' "$S5_ROUND")

mv \
  "$S5_ROOT/candidates_dev_round_${R}.jsonl" \
  "$S5_ROOT/candidate_pool_dev_round_${R}.jsonl"

echo "===== SELECT ROUND $S5_ROUND ====="

"$PY" -u analysis/select_belief_targeted_v1_online.py \
  --round "$S5_ROUND" \
  --split dev \
  --root "$S5_ROOT" \
  --model-root "$SEL"

"$PY" -u analysis/bt_simple5_runtime_guard_v1.py \
  --root "$S5_ROOT" \
  --round "$S5_ROUND" \
  --phase post_select

echo "GEN_SELECT_PASS ROUND=$S5_ROUND"
