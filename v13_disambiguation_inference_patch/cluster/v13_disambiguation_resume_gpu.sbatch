#!/bin/bash
#SBATCH --job-name=v13-resume
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/v13-resume-%j.out

set -euo pipefail
cd "$HOME/mas_rl_detective"
mkdir -p logs
source /storage/modules/packages/anaconda/etc/profile.d/conda.sh
conda activate masdetective
if [ -f .env ]; then set -a; source .env; set +a; fi
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false

python training/train_disambiguation_v13.py \
  --execute \
  --resume-from-checkpoint \
  --rounds-per-role 2 \
  --group-size 4 \
  --bundle-size 3 \
  --confirmation-size 1 \
  --confirm-top-k 2 \
  --systems-per-cell 4 \
  --neighbor-k 5 \
  --balanced-minibatches 3 \
  --grpo-epochs 2 \
  --lr 8e-6 \
  --gate-mode hard
