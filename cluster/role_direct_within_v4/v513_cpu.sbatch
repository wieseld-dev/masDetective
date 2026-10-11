#!/usr/bin/env bash
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=20:00:00
#SBATCH --output=logs/v513-%x-%j.out
#SBATCH --error=logs/v513-%x-%j.err
set -euo pipefail
cd "$HOME/mas_rl_detective"
export PYTHONPATH="$PWD:${PYTHONPATH:-}" PYTHONUNBUFFERED=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
: "${V513_RUN:?Missing V513_RUN}"
: "${V513_SOURCE:?Missing V513_SOURCE}"
"$HOME/.conda/envs/masdetective/bin/python" -u -m \
  analysis.role_direct_within_v4.v513_role_error_attribution run \
  --source "$V513_SOURCE" --out "$V513_RUN"
