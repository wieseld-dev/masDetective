#!/usr/bin/env bash
#SBATCH --job-name=v9-discovery
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/v9-discovery-%j.out
set -euo pipefail
cd "$HOME/mas_rl_detective"
mkdir -p logs artifacts
if [ -f .env ]; then set -a; source .env; set +a; fi
PY="$HOME/.conda/envs/masdetective/bin/python"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
"$PY" -u -m training.train_seeded_probe_discovery_v9 \
  --rounds-per-role 2 \
  --group-size 5 \
  --bundle-size 2 \
  --groups-per-update 2 \
  --grpo-epochs 2 \
  --elite-nll-weight 0.06 \
  --lr 1.5e-5 \
  --kl-beta 0.02 \
  --gate-mode hybrid \
  --holdout-eval \
  --holdout-probes-per-role 1 \
  --output-dir artifacts/v9_probe_policy \
  --log artifacts/v9_probe_discovery.jsonl \
  --summary artifacts/v9_probe_summary.json \
  --response-public-cache artifacts/v9_response_public.jsonl \
  --response-truth-cache artifacts/v9_response_truth.jsonl \
  --execute
