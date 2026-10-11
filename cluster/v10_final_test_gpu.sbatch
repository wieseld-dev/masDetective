#!/usr/bin/env bash
#SBATCH --job-name=v10-final-test
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/v10-final-test-%j.out
set -euo pipefail
cd "$HOME/mas_rl_detective"
mkdir -p logs artifacts
if [ -f .env ]; then set -a; source .env; set +a; fi
PY="$HOME/.conda/envs/masdetective/bin/python"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

# Compute-node preflight only. No API calls and no training.
PYTHONPATH="$PWD" "$PY" -m pytest -q tests/test_v10_final_design.py

if [ ! -d artifacts/v10_policy ]; then
  echo "Missing frozen V10.1 policy: artifacts/v10_policy"
  exit 1
fi
if [ ! -f artifacts/v10_summary.json ]; then
  echo "Missing V10.1 summary: artifacts/v10_summary.json"
  exit 1
fi

"$PY" -u scripts/evaluate_v10_final_untouched.py \
  --policy-dir artifacts/v10_policy \
  --v9-summary artifacts/v9_full_summary.json \
  --v10-summary artifacts/v10_summary.json \
  --probe-manifest artifacts/v10_final_probe_manifest.json \
  --rows artifacts/v10_final_rows.jsonl \
  --summary artifacts/v10_final_summary.json \
  --response-public-cache artifacts/v10_final_response_public.jsonl \
  --response-truth-cache artifacts/v10_final_response_truth.jsonl \
  --probes-per-role 3 \
  --generation-retries 8 \
  --max-similarity 0.94 \
  --gate-mode hybrid \
  --guard-model gpt-4o-mini \
  --seed 20260909 \
  --min-activation-rate 0.75 \
  --execute
