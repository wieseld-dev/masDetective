#!/bin/bash
#SBATCH --job-name=v137-oofcal
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=0-00:30:00
#SBATCH --output=logs/v137-oofcal-%j.out
#SBATCH --error=logs/v137-oofcal-%j.err

set -euo pipefail

cd "$HOME/mas_rl_detective"

source /storage/modules/packages/anaconda/etc/profile.d/conda.sh
conda activate masdetective

export PYTHONPATH="$HOME/mas_rl_detective:${PYTHONPATH:-}"

python -u \
  analysis/v137_oof_role_calibration_v1.py
