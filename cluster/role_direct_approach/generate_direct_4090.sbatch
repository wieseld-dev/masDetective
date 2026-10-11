#!/bin/bash
#SBATCH --job-name=rd-direct-gen
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx_4090:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=20:00:00
#SBATCH --output=logs/rd-direct-gen-%j.out
#SBATCH --error=logs/rd-direct-gen-%j.err

set -euo pipefail

cd "$HOME/mas_rl_detective"

export PYTHONPATH="$PWD:${PYTHONPATH:-}"

PY="$HOME/.conda/envs/masdetective/bin/python"

ROOT="${RD_ROOT:-artifacts/role_direct_approach_directprompt_v1}"
N="${RD_PROBES_PER_ROLE:-8}"

nvidia-smi \
  --query-gpu=name,memory.total,memory.free \
  --format=csv,noheader

"$PY" -u \
  analysis/role_direct_approach/generate_direct_panel_v1.py \
  --root "$ROOT" \
  --model-name Qwen/Qwen2.5-1.5B-Instruct \
  --probes-per-role "$N" \
  --temperature 0.90 \
  --top-p 0.95 \
  --max-new-tokens 96

"$PY" -u \
  analysis/role_direct_approach/expand_shared_panel_v1.py \
  --root "$ROOT" \
  --manifest "$ROOT/manifest_privileged.json" \
  --split dev
