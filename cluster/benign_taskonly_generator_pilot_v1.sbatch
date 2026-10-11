#!/bin/bash
#SBATCH --job-name=bentaskgen
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=10:00:00
#SBATCH --output=logs/bentaskgen-%j.out
#SBATCH --error=logs/bentaskgen-%j.err

set -euo pipefail

cd "$HOME/mas_rl_detective"

PY="/home/wieseld/.conda/envs/masdetective/bin/python"

export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

"$PY" -u scripts/curriculum_generate.py \
  --round 1 \
  --split train \
  --root artifacts/benign_taskonly_generator_pilot_v1 \
  --adapter artifacts/adaptive_v11_curriculum_v2/checkpoints/stage_10_cycle_02/policy \
  --v11-reference artifacts/v11_policy \
  --group-size 5 \
  --temperature 0.85 \
  --top-p 0.95 \
  --max-new-tokens 160 \
  --max-seq-length 4096 \
  --local-retries 6
