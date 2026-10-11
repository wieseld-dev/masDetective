#!/bin/bash
#SBATCH --job-name=v3-prep
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/v3-prep-%j.out
#SBATCH --error=logs/v3-prep-%j.err

set -euo pipefail
cd "$HOME/mas_rl_detective"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate masdetective
export PYTHONPATH="$HOME/mas_rl_detective:$HOME/mas_rl_detective/analysis:${PYTHONPATH:-}"
python -u analysis/v3_prepare.py --root "${V3_ROOT:-artifacts/v3_coverage_elimination_pilot}"
