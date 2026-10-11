#!/bin/bash
#SBATCH --job-name=rd-gen
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx_4090:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=20:00:00
#SBATCH --output=logs/rd-gen-%j.out
#SBATCH --error=logs/rd-gen-%j.err

set -euo pipefail

cd "$HOME/mas_rl_detective"

export PYTHONPATH="$PWD:${PYTHONPATH:-}"

PY="$HOME/.conda/envs/masdetective/bin/python"

ROOT="${RD_ROOT:-artifacts/role_direct_approach_v1}"
N="${RD_PROBES_PER_ROLE:-6}"

nvidia-smi \
  --query-gpu=name,memory.total,memory.free \
  --format=csv,noheader

"$PY" \
  analysis/role_direct_approach/generate_role_direct_v1.py \
  --root "$ROOT" \
  --manifest "$ROOT/manifest_privileged.json" \
  --state-source "$ROOT/states_dev_round_00.jsonl" \
  --adapter \
    artifacts/v14_adaptive_role_v1/checkpoints/round_07/policy \
  --split dev \
  --probes-per-role "$N" \
  --temperature 0.85 \
  --top-p 0.95 \
  --max-new-tokens 64 \
  --max-seq-length 3072
