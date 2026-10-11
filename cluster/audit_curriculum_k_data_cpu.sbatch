#!/bin/bash
#SBATCH --job-name=k-audit
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=logs/k-audit-%j.out
#SBATCH --error=logs/k-audit-%j.err

set -euo pipefail

cd "$HOME/mas_rl_detective"

source /storage/modules/packages/anaconda/etc/profile.d/conda.sh
conda activate masdetective

export PYTHONPATH="$HOME/mas_rl_detective:${PYTHONPATH:-}"

python -u analysis/audit_curriculum_k_data.py
