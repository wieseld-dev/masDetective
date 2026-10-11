#!/bin/bash
#SBATCH --job-name=s10-sensor
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=0-00:20:00
#SBATCH --output=logs/s10-sensor-%j.out
#SBATCH --error=logs/s10-sensor-%j.err

set -euo pipefail

cd "$HOME/mas_rl_detective"

source \
  /storage/modules/packages/anaconda/etc/profile.d/conda.sh

conda activate masdetective

export PYTHONPATH="$HOME/mas_rl_detective:${PYTHONPATH:-}"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

python -u \
  scripts/train_stage10_response_sensor_v1.py
