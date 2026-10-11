#!/bin/bash
#SBATCH --job-name=v13-count
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/v13-count-%j.out

set -euo pipefail
cd "$HOME/mas_rl_detective"
mkdir -p logs

source /storage/modules/packages/anaconda/etc/profile.d/conda.sh
conda activate masdetective

if [ -f .env ]; then
  set -a
  source .env
  set +a
fi

export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false

python analysis/adaptive_fanout_count_v13.py \
  --execute \
  --systems-per-cell 1 \
  --min-probes-per-role 3 \
  --max-probes-per-role 7 \
  --max-total-queries 120 \
  --confidence 0.90 \
  --evidence-discount 0.45 \
  --evidence-clip 1.5 \
  --neighbor-k 5 \
  --gate-mode hard
