#!/bin/bash
#SBATCH --job-name=q38lora-sum
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=logs/q38lora-sum-%j.out
#SBATCH --error=logs/q38lora-sum-%j.err

set -euo pipefail

cd "$HOME/mas_rl_detective"

PY="/home/wieseld/.conda/envs/masdetective/bin/python"

"$PY" -u \
    analysis/q38_utility_lora_summary_v1.py
