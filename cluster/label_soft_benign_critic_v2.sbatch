#!/bin/bash
#SBATCH --job-name=benignlbl2
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=10:00:00
#SBATCH --output=logs/benignlbl2-%j.out
#SBATCH --error=logs/benignlbl2-%j.err

set -euo pipefail

cd "$HOME/mas_rl_detective"

PY="/home/wieseld/.conda/envs/masdetective/bin/python"

export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

set -a
source .env
set +a

"$PY" -u \
  analysis/label_soft_benign_critic_v2.py \
  --concurrency 20
