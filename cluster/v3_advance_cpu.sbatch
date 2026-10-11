#!/bin/bash
#SBATCH --job-name=v3-adv
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:55:00
#SBATCH --output=logs/v3-adv-%j.out
#SBATCH --error=logs/v3-adv-%j.err

set -euo pipefail
cd "$HOME/mas_rl_detective"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate masdetective
export PYTHONPATH="$HOME/mas_rl_detective:$HOME/mas_rl_detective/analysis:${PYTHONPATH:-}"

python -u analysis/v3_advance.py \
  --round "$V3_ROUND" \
  --split "${V3_SPLIT:-train}" \
  --root "$V3_ROOT" \
  --manifest "$V3_MANIFEST" \
  --batch-size "${V3_PARTICLE_BATCH:-4096}"
