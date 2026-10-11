#!/usr/bin/env bash
#SBATCH --job-name=mas5-v49-pilot
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=20:00:00
#SBATCH --output=logs/v49-%x-%j.out
#SBATCH --error=logs/v49-%x-%j.err
set -euo pipefail
cd /home/wieseld/mas_rl_detective
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
set -a
source .env
set +a
export PYTHONUNBUFFERED=1
"$HOME/.conda/envs/masdetective/bin/python" -u -m analysis.role_direct_within_v4.v49_pilot execute --root "$V49_ROOT"
"$HOME/.conda/envs/masdetective/bin/python" -u -m analysis.role_direct_within_v4.v49_pilot report --root "$V49_ROOT"
