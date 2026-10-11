#!/bin/bash
#SBATCH --job-name=ben-teacher
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=10:00:00
#SBATCH --output=logs/ben-teacher-%j.out
#SBATCH --error=logs/ben-teacher-%j.err

set -euo pipefail

cd "$HOME/mas_rl_detective"

PY="/home/wieseld/.conda/envs/masdetective/bin/python"

export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

set -a
source .env
set +a

"$PY" -u \
  analysis/build_standalone_benign_teacher_v1.py \
  --concurrency 20
