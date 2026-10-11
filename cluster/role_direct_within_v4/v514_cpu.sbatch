#!/usr/bin/env bash
#SBATCH --job-name=mas5-v514-match
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=20:00:00
#SBATCH --output=logs/v514-%x-%j.out
#SBATCH --error=logs/v514-%x-%j.err
set -euo pipefail
cd "$HOME/mas_rl_detective"
export PYTHONPATH="$PWD:${PYTHONPATH:-}" PYTHONUNBUFFERED=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
[[ -f .env ]] || { echo 'Missing .env with OpenAI API key' >&2; exit 2; }
set -a
source .env
set +a
: "${V514_RUN:?Missing V514_RUN}"
[[ "${V514_APPROVE_MAS_EXECUTIONS:-}" == 30 ]] || { echo 'Missing explicit execution approval' >&2; exit 2; }
PY="$HOME/.conda/envs/masdetective/bin/python"
"$PY" -u -m analysis.role_direct_within_v4.v514_matched_discrimination execute --root "$V514_RUN"
"$PY" -u -m analysis.role_direct_within_v4.v514_matched_discrimination report --root "$V514_RUN"
