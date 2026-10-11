#!/usr/bin/env bash
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=6
#SBATCH --mem=20G
#SBATCH --time=20:00:00
#SBATCH --output=logs/v5112-%x-%j.out
#SBATCH --error=logs/v5112-%x-%j.err
set -euo pipefail
cd "$HOME/mas_rl_detective"
export PYTHONPATH="$PWD:${PYTHONPATH:-}" PYTHONUNBUFFERED=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
: "${V5112_RUN:?V5112_RUN missing}"
exec "$HOME/.conda/envs/masdetective/bin/python" -u -m analysis.role_direct_within_v4.v5112_overlap_safe run --out "$V5112_RUN"
