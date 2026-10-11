#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=20:00:00
#SBATCH --output=logs/v42-%x-%j.out
#SBATCH --error=logs/v42-%x-%j.err
set -euo pipefail
cd /home/wieseld/mas_rl_detective
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export OMP_NUM_THREADS=6
export OPENBLAS_NUM_THREADS=6
export MKL_NUM_THREADS=6
export PYTHONUNBUFFERED=1
"$HOME/.conda/envs/masdetective/bin/python" -u analysis/role_direct_within_v4/v42_sensor.py benchmark --root "$V42_ROOT"
