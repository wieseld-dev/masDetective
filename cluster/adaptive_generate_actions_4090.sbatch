#!/bin/bash
#SBATCH --job-name=adapt-gen-v11
#SBATCH --partition=main
#SBATCH --gpus=1
#SBATCH --constraint=rtx_4090
#SBATCH --mem=32G
#SBATCH --output=logs/adapt-gen-v11-%j.out
#SBATCH --error=logs/adapt-gen-v11-%j.err

set -euo pipefail

cd "$HOME/mas_rl_detective"
export PYTHONPATH="$HOME/mas_rl_detective:${PYTHONPATH:-}"

source /storage/modules/packages/anaconda/etc/profile.d/conda.sh
conda activate masdetective

mkdir -p logs artifacts/adaptive_v1

python -u adaptive_generate_action_library.py \
  --adapter artifacts/v11_policy \
  --v11-frozen-summary artifacts/v11_summary.json \
  --initial-elite-summary artifacts/v11_summary.json \
  --variants 1 \
  --max-attempts-per-variant 4 \
  --no-4bit
