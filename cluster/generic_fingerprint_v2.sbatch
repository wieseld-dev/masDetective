#!/bin/bash
#SBATCH --job-name=fp-v2
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=10:00:00
#SBATCH --output=logs/fp-v2-%j.out
#SBATCH --error=logs/fp-v2-%j.err

set -euo pipefail

cd "$HOME/mas_rl_detective"

PY="/home/wieseld/.conda/envs/masdetective/bin/python"

export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

"$PY" -u \
  analysis/generic_fingerprint_v2.py
