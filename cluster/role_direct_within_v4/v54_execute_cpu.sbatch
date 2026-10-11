#!/usr/bin/env bash
#SBATCH --job-name=mas5-v54-r2-pilot
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=20:00:00
#SBATCH --output=logs/v54-%x-%j.out
#SBATCH --error=logs/v54-%x-%j.err
set -euo pipefail
cd /home/wieseld/mas_rl_detective
export PYTHONPATH="$PWD:${PYTHONPATH:-}" PYTHONUNBUFFERED=1
if [[ ! -f .env ]]; then echo 'Expected .env with OPENAI_API_KEY' >&2; exit 2; fi
set -a
source .env
set +a
"$HOME/.conda/envs/masdetective/bin/python" -u -m analysis.role_direct_within_v4.v54_r2_paired_pilot execute --root "$V54_ROOT"
"$HOME/.conda/envs/masdetective/bin/python" -u -m analysis.role_direct_within_v4.v54_r2_paired_pilot report --root "$V54_ROOT"
