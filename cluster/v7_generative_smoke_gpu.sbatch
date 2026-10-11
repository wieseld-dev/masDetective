#!/usr/bin/env bash
#SBATCH --job-name=v7-smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=logs/v7-smoke-%j.out

set -euo pipefail
cd "$HOME/mas_rl_detective"
set -a
source .env
set +a
mkdir -p logs artifacts
PY="$HOME/.conda/envs/masdetective/bin/python"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

"$PY" -u -m training.train_generative_probe_ppo_v7 \
  --episodes 16 \
  --max-followups 2 \
  --max-attempts 4 \
  --rollout-episodes 4 \
  --ppo-epochs 1 \
  --lr 1e-5 \
  --gate-mode hybrid \
  --output-dir artifacts/v7_generative_policy_smoke \
  --log artifacts/v7_train_smoke.jsonl \
  --response-public-cache artifacts/v7_smoke_response_public.jsonl \
  --response-truth-cache artifacts/v7_smoke_response_truth.jsonl \
  --execute
