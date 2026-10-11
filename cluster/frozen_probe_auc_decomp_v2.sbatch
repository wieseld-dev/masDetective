#!/bin/bash
#SBATCH --job-name=auc-dec2
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --time=00:08:00
#SBATCH --output=logs/auc-dec2-%j.out
#SBATCH --error=logs/auc-dec2-%j.err

set -euo pipefail

cd "$HOME/mas_rl_detective"

PY="/home/wieseld/.conda/envs/masdetective/bin/python"

export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

"$PY" -u \
  analysis/frozen_probe_auc_decomp_v2.py
