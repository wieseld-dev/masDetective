#!/usr/bin/env bash
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/v512-%x-%j.out
#SBATCH --error=logs/v512-%x-%j.err
set -euo pipefail
cd "$HOME/mas_rl_detective"
export PYTHONPATH="$PWD:${PYTHONPATH:-}" PYTHONUNBUFFERED=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
: "${V512_RUN:?Missing V512_RUN}"
: "${V512_SOURCE:?Missing V512_SOURCE}"
"$HOME/.conda/envs/masdetective/bin/python" -u -m \
  analysis.role_direct_within_v4.v512_calibration_audit run \
  --source "$V512_SOURCE" --out "$V512_RUN"
