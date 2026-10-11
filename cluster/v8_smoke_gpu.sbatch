#!/usr/bin/env bash
#SBATCH --job-name=v8-smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=logs/v8-smoke-%j.out
set -euo pipefail
cd "$HOME/mas_rl_detective"
mkdir -p logs artifacts
if [ -f .env ]; then set -a; source .env; set +a; fi
PY="$HOME/.conda/envs/masdetective/bin/python"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
"$PY" -u -m training.train_counterfactual_grpo_v8 \
  --states 8 \
  --group-size 4 \
  --groups-per-update 2 \
  --grpo-epochs 1 \
  --lr 1.5e-5 \
  --kl-beta 0.015 \
  --gate-mode hybrid \
  --output-dir artifacts/v8_grpo_policy_smoke \
  --log artifacts/v8_grpo_smoke.jsonl \
  --response-public-cache artifacts/v8_smoke_response_public.jsonl \
  --response-truth-cache artifacts/v8_smoke_response_truth.jsonl \
  --execute
