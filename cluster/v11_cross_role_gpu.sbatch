#!/usr/bin/env bash
#SBATCH --job-name=v11-xrole
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=16:00:00
#SBATCH --output=logs/v11-xrole-%j.out
set -euo pipefail
cd "$HOME/mas_rl_detective"
mkdir -p logs artifacts
if [ -f .env ]; then set -a; source .env; set +a; fi
PY="$HOME/.conda/envs/masdetective/bin/python"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Compute-node-only preflight. No API calls.
"$PY" -m pytest -q \
  tests/test_v11_role_holdout.py \
  tests/test_v11_hashcheck.py \
  tests/test_v10_reward.py \
  tests/test_v10_checkpoint.py
"$PY" scripts/verify_v11_preflight.py

if [ -e artifacts/v11_checkpoint.json ] || [ -e artifacts/v11_policy ]; then
  echo "Existing V11 artifacts found. Use the resume job or archive V11 artifacts first."
  exit 1
fi

"$PY" -u -m training.train_cross_role_probe_discovery_v11 \
  --rounds-per-role 2 \
  --group-size 5 \
  --bundle-size 4 \
  --confirmation-size 2 \
  --confirm-top-k 2 \
  --balanced-minibatches 4 \
  --grpo-epochs 2 \
  --elite-nll-weight 0.06 \
  --lr 8e-6 \
  --kl-beta 0.0 \
  --syntax-retries 3 \
  --holdout-generation-retries 8 \
  --gate-mode hybrid \
  --min-activation-rate 0.75 \
  --novelty-free-similarity 0.90 \
  --novelty-weight 0.25 \
  --semantic-hard-duplicate 0.97 \
  --holdout-max-similarity 0.94 \
  --resume-adapter artifacts/v10_policy \
  --v10-frozen-summary artifacts/v10_summary.json \
  --initial-elite-summary artifacts/v10_summary.json \
  --baseline-summary artifacts/v10_summary.json \
  --holdout-eval \
  --holdout-probes-per-role 3 \
  --output-dir artifacts/v11_policy \
  --log artifacts/v11_probe_discovery.jsonl \
  --summary artifacts/v11_summary.json \
  --response-public-cache artifacts/v11_response_public.jsonl \
  --response-truth-cache artifacts/v11_response_truth.jsonl \
  --checkpoint-state artifacts/v11_checkpoint.json \
  --checkpoint-root artifacts/v11_checkpoints \
  --optimizer-mirror artifacts/v11_optimizer.pt \
  --execute

"$PY" scripts/diagnose_v11.py --summary artifacts/v11_summary.json
