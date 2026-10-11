#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=0-02:00:00
#SBATCH --job-name=dagop-eval
#SBATCH --output=logs/dagop-eval-%J.out
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --tmp=10G

set -euo pipefail
cd "$HOME/mas_rl_detective"
mkdir -p logs artifacts/dag_operator_active_poc

source ~/.bashrc || true
conda activate masdetective

python -u dag_operator_active_poc.py \
  --n-prompts 60 \
  --analysis-only \
  --steps 3000 \
  --dim 16 \
  --random-repeats 200 \
  --seed 73
