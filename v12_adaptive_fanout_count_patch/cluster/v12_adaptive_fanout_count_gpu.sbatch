#!/usr/bin/env bash
#SBATCH --job-name=v12-fanout-count
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/v12-fanout-count-%j.out

set -euo pipefail
cd "$HOME/mas_rl_detective"
mkdir -p logs

if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/anaconda3/etc/profile.d/conda.sh"
fi
conda activate masdetective

if [ -f .env ]; then
  set -a
  source .env
  set +a
fi

export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false

python analysis/adaptive_fanout_count_v12.py \
  --execute \
  --candidate-universe train \
  --systems-per-cell 1 \
  --min-probes-per-role 3 \
  --max-probes-per-role 8 \
  --max-total-queries 120 \
  --confidence 0.92 \
  --evidence-discount 0.55 \
  --evidence-clip 2.0 \
  --gate-mode hard
