#!/bin/bash
#SBATCH --job-name=r1-audit2
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=logs/r1-audit2-%j.out
#SBATCH --error=logs/r1-audit2-%j.err

set -euo pipefail

cd "$HOME/mas_rl_detective"

source /storage/modules/packages/anaconda/etc/profile.d/conda.sh
conda activate masdetective

export PYTHONPATH="$HOME/mas_rl_detective:${PYTHONPATH:-}"

python -u analysis/audit_r1_identifiability_v2.py
