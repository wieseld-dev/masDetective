#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=20:00:00
#SBATCH --output=logs/v4u-%x-%j.out
#SBATCH --error=logs/v4u-%x-%j.err
set -euo pipefail
cd /home/wieseld/mas_rl_detective
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export OMP_NUM_THREADS=2
"$HOME/.conda/envs/masdetective/bin/python" -u analysis/role_direct_within_v4/role_upgrade_v1.py selector --root "$V4_ROOT"
