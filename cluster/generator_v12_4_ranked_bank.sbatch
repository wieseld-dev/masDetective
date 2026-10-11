#!/bin/bash
#SBATCH --partition=main
#SBATCH --gpus=rtx_3090:1
#SBATCH --time=0-00:10:00
#SBATCH --job-name=genv124
#SBATCH --output=logs/genv124-%J.out
#SBATCH --error=logs/genv124-%J.err

set -euo pipefail

cd "$HOME/mas_rl_detective"

PY="/home/wieseld/.conda/envs/masdetective/bin/python"

export PYTHONPATH="$PWD:${PYTHONPATH:-}"

"$PY" -u \
  analysis/generator_v12_4_ranked_bank.py
