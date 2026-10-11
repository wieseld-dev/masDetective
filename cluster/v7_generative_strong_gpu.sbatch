#!/usr/bin/env bash
#SBATCH --job-name=v7-genrl
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/v7-genrl-%j.out

set -euo pipefail
cd "$HOME/mas_rl_detective"
mkdir -p logs artifacts
PY="$HOME/.conda/envs/masdetective/bin/python"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Strong single-seed run. Seed interactions are replayed from V6.5; only generated follow-ups are live.
"$PY" -u -m training.train_generative_probe_ppo_v7 \
  --episodes 256 \
  --max-followups 3 \
  --max-attempts 6 \
  --rollout-episodes 8 \
  --ppo-epochs 2 \
  --lr 1e-5 \
  --clip 0.20 \
  --gamma 0.95 \
  --temperature-start 0.95 \
  --temperature-end 0.65 \
  --top-p 0.95 \
  --gate-mode hybrid \
  --execute
