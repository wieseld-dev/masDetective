#!/bin/bash
#SBATCH --job-name=q38r7-sum
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=logs/q38r7-sum-%j.out
#SBATCH --error=logs/q38r7-sum-%j.err

set -euo pipefail

cd "$HOME/mas_rl_detective"

PY="/home/wieseld/.conda/envs/masdetective/bin/python"

"$PY" -u \
    analysis/q38_vs_r7_utility_summary_v1.py
