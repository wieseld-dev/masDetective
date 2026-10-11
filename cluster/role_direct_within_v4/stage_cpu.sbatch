#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=20:00:00
#SBATCH --output=logs/v4s-%x-%j.out
#SBATCH --error=logs/v4s-%x-%j.err
set -euo pipefail
cd /home/wieseld/mas_rl_detective
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export OMP_NUM_THREADS=6
export PYTHONUNBUFFERED=1
"$HOME/.conda/envs/masdetective/bin/python" -u analysis/role_direct_within_v4/pipeline.py "$V4_STAGE" --root "$V4_ROOT"
