#!/bin/bash
#SBATCH --job-name=fp-explain
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=logs/fp-explain-%j.out
#SBATCH --error=logs/fp-explain-%j.err

set -euo pipefail

cd "$HOME/mas_rl_detective"

PY="/home/wieseld/.conda/envs/masdetective/bin/python"

export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

"$PY" -u \
  analysis/explainable_fingerprint_v1.py
