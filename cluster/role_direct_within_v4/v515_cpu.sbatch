#!/usr/bin/env bash
#SBATCH --job-name=mas5-v515-mbg
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=20:00:00
#SBATCH --output=logs/v515-%x-%j.out
#SBATCH --error=logs/v515-%x-%j.err
set -euo pipefail
cd "$HOME/mas_rl_detective"
export PYTHONPATH="$PWD:${PYTHONPATH:-}" PYTHONUNBUFFERED=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
[[ -f .env ]] || { echo 'Missing .env for API key' >&2; exit 2; }
set -a
source .env
set +a
: "${V515_RUN:?Missing V515_RUN}"
[[ "${V515_APPROVE_MAS_EXECUTIONS:-}" == 60 ]] || { echo 'No 60-call approval' >&2; exit 2; }
PY="$HOME/.conda/envs/masdetective/bin/python"
"$PY" -u -m analysis.role_direct_within_v4.v515_multi_background execute --root "$V515_RUN"
"$PY" -u -m analysis.role_direct_within_v4.v515_multi_background report --root "$V515_RUN"
