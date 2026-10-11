#!/bin/bash
#SBATCH --job-name=auc-decomp
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --time=00:08:00
#SBATCH --output=logs/auc-decomp-%j.out
#SBATCH --error=logs/auc-decomp-%j.err

set -euo pipefail

cd "$HOME/mas_rl_detective"

PY="/home/wieseld/.conda/envs/masdetective/bin/python"

export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

"$PY" -u \
  analysis/frozen_probe_auc_decomp_v1.py
