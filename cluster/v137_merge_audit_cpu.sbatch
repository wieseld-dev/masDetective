#!/bin/bash
#SBATCH --job-name=v137-merge
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0-00:30:00
#SBATCH --output=logs/v137-merge-%j.out
#SBATCH --error=logs/v137-merge-%j.err

set -euo pipefail
cd "$HOME/mas_rl_detective"

source /storage/modules/packages/anaconda/etc/profile.d/conda.sh
conda activate masdetective

export PYTHONPATH="$HOME/mas_rl_detective:${PYTHONPATH:-}"

python -u analysis/v137_merge_audit_build_dataset_v1.py
