#!/bin/bash
#SBATCH --job-name=v13-2-dual
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/v13-2-dual-%j.out

set -euo pipefail
cd "$HOME/mas_rl_detective"
mkdir -p logs artifacts/v13_2_dual_judge artifacts/v13_2_pilot_replay

source /storage/modules/packages/anaconda/etc/profile.d/conda.sh
conda activate masdetective

export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -u analysis/train_v13_2_dual_judge.py \
  --model auto \
  --output-dir artifacts/v13_2_dual_judge \
  --epochs 8 \
  --patience 2 \
  --batch-size 2 \
  --grad-accum 8 \
  --eval-batch-size 6 \
  --max-length 1400 \
  --lr 8e-5 \
  --margin 0.75 \
  --bce-weight 0.35 \
  --v11-source-weight 0.70 \
  --attr-target-weight 0.75 \
  --attr-hard-weight 1.00 \
  --minus-flip-weight 2.00 \
  --seed 1322 \
  --fold 2 \
  --n-splits 5

# Offline only: reuses the 120 public answers from the already-consumed pilot.
# This makes zero OpenAI/API calls.
python -u analysis/replay_v13_2_pilot.py \
  --judge-dir artifacts/v13_2_dual_judge \
  --sessions artifacts/v13_fanout_count_pilot1/sessions.jsonl \
  --output-dir artifacts/v13_2_pilot_replay \
  --eval-batch-size 6
