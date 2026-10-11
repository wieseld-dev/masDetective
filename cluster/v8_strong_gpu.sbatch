#!/usr/bin/env bash
#SBATCH --job-name=v8-grpo
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=18:00:00
#SBATCH --output=logs/v8-grpo-%j.out
set -euo pipefail
cd "$HOME/mas_rl_detective"
mkdir -p logs artifacts
if [ -f .env ]; then set -a; source .env; set +a; fi
PY="$HOME/.conda/envs/masdetective/bin/python"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
"$PY" -u -m training.train_counterfactual_grpo_v8 \
  --states 64 \
  --group-size 4 \
  --groups-per-update 4 \
  --grpo-epochs 2 \
  --lr 1.5e-5 \
  --kl-beta 0.015 \
  --gate-mode hybrid \
  --include-sham \
  --execute
