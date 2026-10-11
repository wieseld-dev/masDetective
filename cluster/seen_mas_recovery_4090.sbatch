#!/bin/bash
#SBATCH --job-name=seen-rec
#SBATCH --partition=main
#SBATCH --gpus=1
#SBATCH --constraint=rtx_4090
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=logs/seen-rec-%j.out
#SBATCH --error=logs/seen-rec-%j.err

set -euo pipefail

cd "$HOME/mas_rl_detective"

source /storage/modules/packages/anaconda/etc/profile.d/conda.sh
conda activate masdetective

export PYTHONPATH="$HOME/mas_rl_detective:${PYTHONPATH:-}"

python -u analysis/seen_mas_recovery_replay.py
