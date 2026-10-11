#!/bin/bash
#SBATCH --job-name=k3-oracle
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=00:20:00
#SBATCH --output=logs/k3-oracle-%j.out
#SBATCH --error=logs/k3-oracle-%j.err

set -euo pipefail

cd "$HOME/mas_rl_detective"

export PYTHONPATH="$HOME/mas_rl_detective:${PYTHONPATH:-}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate masdetective

python -u analysis/oracle_activation_exact_replay.py
