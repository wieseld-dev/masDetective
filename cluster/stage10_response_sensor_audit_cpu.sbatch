#!/bin/bash
#SBATCH --job-name=s10-audit
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=0-01:00:00
#SBATCH --output=logs/s10-audit-%j.out
#SBATCH --error=logs/s10-audit-%j.err

set -euo pipefail

cd "$HOME/mas_rl_detective"

source \
  /storage/modules/packages/anaconda/etc/profile.d/conda.sh

conda activate masdetective

export PYTHONPATH="$HOME/mas_rl_detective:${PYTHONPATH:-}"
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false

python -u \
  scripts/stage10_response_sensor_audit_v1.py
