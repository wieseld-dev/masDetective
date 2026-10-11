#!/bin/bash
#SBATCH --job-name=fp-scale1
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=10:00:00
#SBATCH --output=logs/fp-scale1-%j.out
#SBATCH --error=logs/fp-scale1-%j.err

set -euo pipefail

cd "$HOME/mas_rl_detective"

PY="/home/wieseld/.conda/envs/masdetective/bin/python"

export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

"$PY" -u \
  analysis/generic_fingerprint_scaling_v1.py
