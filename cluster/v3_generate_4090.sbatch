#!/bin/bash
#SBATCH --job-name=v3-gen
#SBATCH --partition=main
#SBATCH --gpus=1
#SBATCH --constraint=rtx_4090
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:40:00
#SBATCH --output=logs/v3-gen-%j.out
#SBATCH --error=logs/v3-gen-%j.err

set -euo pipefail
cd "$HOME/mas_rl_detective"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate masdetective
export PYTHONPATH="$HOME/mas_rl_detective:$HOME/mas_rl_detective/analysis:${PYTHONPATH:-}"
python -u analysis/v3_generate.py \
  --round "$V3_ROUND" \
  --split "${V3_SPLIT:-train}" \
  --root "$V3_ROOT" \
  --adapter "$V3_ADAPTER" \
  --v11-reference "${V3_V11:-artifacts/v11_policy}" \
  --group-size "${V3_GROUP_SIZE:-4}"
