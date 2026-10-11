#!/bin/bash
#SBATCH --job-name=jev-max
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --time=04:00:00
#SBATCH --output=logs/jev-max-%j.out
#SBATCH --error=logs/jev-max-%j.err

set -euo pipefail

cd "$HOME/mas_rl_detective"

mkdir -p \
  logs \
  artifacts/jev_max_v2

source /storage/modules/packages/anaconda/etc/profile.d/conda.sh
conda activate masdetective

export PYTHONPATH="$HOME/mas_rl_detective:${PYTHONPATH:-}"

source "$HOME/.config/typesafe/env"

test -n "${TYPESAFE_API_KEY:-}" || {
    echo "ERROR: TYPESAFE_API_KEY missing"
    exit 1
}

python -u scripts/jev_max_benchmark.py \
  --output-dir artifacts/jev_max_v2 \
  --calibration-n 8
