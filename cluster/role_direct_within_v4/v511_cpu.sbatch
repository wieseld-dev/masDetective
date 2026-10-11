#!/usr/bin/env bash
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=6
#SBATCH --mem=20G
#SBATCH --time=20:00:00
#SBATCH --output=logs/v511-%x-%j.out
#SBATCH --error=logs/v511-%x-%j.err
set -euo pipefail
cd /home/wieseld/mas_rl_detective
export PYTHONPATH="$PWD:${PYTHONPATH:-}" PYTHONUNBUFFERED=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
: "${V511_RUN:?missing V511_RUN}"
exec "$HOME/.conda/envs/masdetective/bin/python" -u -m analysis.role_direct_within_v4.v511_rolewise_verifier run --out "$V511_RUN"
