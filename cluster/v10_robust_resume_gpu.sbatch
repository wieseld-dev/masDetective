#!/usr/bin/env bash
#SBATCH --job-name=v10-1-resume
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/v10-1-resume-%j.out
set -euo pipefail
cd "$HOME/mas_rl_detective"
mkdir -p logs artifacts
if [ -f .env ]; then set -a; source .env; set +a; fi
PY="$HOME/.conda/envs/masdetective/bin/python"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Fast compute-node preflight; no API calls.
PYTHONPATH="$PWD" "$PY" -m pytest -q tests/test_v10_reward.py tests/test_v10_checkpoint.py

STATE=artifacts/v10_checkpoint.json
if [ ! -f "$STATE" ]; then
  echo "No committed V10.1 checkpoint: $STATE"
  exit 1
fi

"$PY" -u -m training.train_robust_probe_discovery_v10 \
  --rounds-per-role 3 \
  --group-size 6 \
  --bundle-size 4 \
  --confirmation-size 2 \
  --confirm-top-k 2 \
  --balanced-minibatches 3 \
  --grpo-epochs 2 \
  --elite-nll-weight 0.06 \
  --lr 1e-5 \
  --kl-beta 0.0 \
  --syntax-retries 3 \
  --holdout-generation-retries 6 \
  --gate-mode hybrid \
  --min-activation-rate 0.75 \
  --novelty-free-similarity 0.90 \
  --novelty-weight 0.25 \
  --semantic-hard-duplicate 0.97 \
  --holdout-max-similarity 0.94 \
  --resume-from-checkpoint \
  --initial-elite-summary artifacts/v9_full_summary.json \
  --baseline-summary artifacts/v9_full_exhaustive_holdout_summary.json \
  --holdout-eval \
  --holdout-probes-per-role 3 \
  --output-dir artifacts/v10_policy \
  --log artifacts/v10_probe_discovery.jsonl \
  --summary artifacts/v10_summary.json \
  --response-public-cache artifacts/v10_response_public.jsonl \
  --response-truth-cache artifacts/v10_response_truth.jsonl \
  --seed-response-public-cache artifacts/v9_full_response_public.jsonl \
  --seed-response-truth-cache artifacts/v9_full_response_truth.jsonl \
  --checkpoint-state artifacts/v10_checkpoint.json \
  --checkpoint-root artifacts/v10_checkpoints \
  --execute
