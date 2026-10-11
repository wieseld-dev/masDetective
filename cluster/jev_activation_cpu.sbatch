#!/bin/bash
#SBATCH --job-name=jev-act
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=02:00:00
#SBATCH --output=logs/jev-act-%j.out
#SBATCH --error=logs/jev-act-%j.err

set -euo pipefail

cd "$HOME/mas_rl_detective"
mkdir -p logs artifacts/jev_activation_v1

source /storage/modules/packages/anaconda/etc/profile.d/conda.sh
conda activate masdetective

export PYTHONPATH="$HOME/mas_rl_detective:${PYTHONPATH:-}"

source "$HOME/.config/typesafe/env"

test -n "${TYPESAFE_API_KEY:-}" || {
    echo "ERROR: TYPESAFE_API_KEY missing"
    exit 1
}

python -u scripts/jev_activation_benchmark.py \
  --output-dir artifacts/jev_activation_v1 \
  --historical-limit 100 \
  --seed 20260921
