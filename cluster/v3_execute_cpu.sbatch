#!/bin/bash
#SBATCH --job-name=v3-exec
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --output=logs/v3-exec-%j.out
#SBATCH --error=logs/v3-exec-%j.err

set -euo pipefail
cd "$HOME/mas_rl_detective"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate masdetective
export PYTHONPATH="$HOME/mas_rl_detective:${PYTHONPATH:-}"

if [ -f .env ]; then
  set -a
  source .env
  set +a
fi

python -u analysis/v3_execute.py \
  --round "$V3_ROUND" \
  --split "${V3_SPLIT:-train}" \
  --root "$V3_ROOT" \
  --manifest "$V3_MANIFEST" \
  --workers "${V3_WORKERS:-6}"
