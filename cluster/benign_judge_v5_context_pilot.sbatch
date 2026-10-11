#!/bin/bash
#SBATCH --job-name=benignv5ctx
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=10:00:00
#SBATCH --output=logs/benignv5ctx-%j.out
#SBATCH --error=logs/benignv5ctx-%j.err

set -euo pipefail

cd "$HOME/mas_rl_detective"

PY="/home/wieseld/.conda/envs/masdetective/bin/python"

export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

set -a
source .env
set +a

"$PY" -u \
  analysis/benign_judge_v5_context_pilot.py \
  --concurrency 20
