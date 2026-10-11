#!/bin/bash
#SBATCH --job-name=gensens-diag
#SBATCH --partition=gpu
#SBATCH --constraint=rtx_4090
#SBATCH --gres=gpu:1
#SBATCH --exclude=cs-4090-09
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/gensens-diag-%j.out
#SBATCH --error=logs/gensens-diag-%j.err

set -euo pipefail

cd "$HOME/mas_rl_detective"

PY=/home/wieseld/.conda/envs/masdetective/bin/python

export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false

"$PY" -u analysis/eval_general_role_sensor_seen_v1.py
