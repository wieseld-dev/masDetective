#!/usr/bin/env bash
#SBATCH --job-name=mas5-v50-collect
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=20:00:00
#SBATCH --output=logs/v50-%x-%j.out
#SBATCH --error=logs/v50-%x-%j.err
set -euo pipefail
cd /home/wieseld/mas_rl_detective
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
set -a; source .env; set +a
export PYTHONUNBUFFERED=1
PY="$HOME/.conda/envs/masdetective/bin/python"
"$PY" -u -m analysis.role_direct_within_v4.v50_matched_expansion execute --root "$V50_ROOT"
"$PY" -u -m analysis.role_direct_within_v4.v50_matched_expansion evidence --root "$V50_ROOT"
