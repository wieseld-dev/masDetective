#!/bin/bash
#SBATCH --job-name=v13-pilot1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=logs/v13-pilot1-%j.out

set -euo pipefail
cd "$HOME/mas_rl_detective"
mkdir -p logs

source /storage/modules/packages/anaconda/etc/profile.d/conda.sh
conda activate masdetective

set -a
source .env
set +a

export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false

python analysis/adaptive_fanout_count_v13_pilot1.py \
  --output-dir artifacts/v13_fanout_count_pilot1 \
  --systems-per-cell 1 \
  --execute
