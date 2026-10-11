#!/bin/bash
#SBATCH --job-name=v4-stage0
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --time=20:00:00
#SBATCH --output=logs/v4-stage0-%j.out
#SBATCH --error=logs/v4-stage0-%j.err

set -euo pipefail

cd /home/wieseld/mas_rl_detective

export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=6

PY="$HOME/.conda/envs/masdetective/bin/python"

"$PY" -u \
    analysis/role_direct_within_v4/stage0.py
