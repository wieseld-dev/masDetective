#!/usr/bin/env bash
#SBATCH --job-name=v8-final
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --output=logs/v8-final-%j.out
set -euo pipefail
cd "$HOME/mas_rl_detective"
mkdir -p logs artifacts
if [ -f .env ]; then set -a; source .env; set +a; fi
PY="$HOME/.conda/envs/masdetective/bin/python"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
"$PY" -u -m evaluation.eval_v8_end_to_end --execute
