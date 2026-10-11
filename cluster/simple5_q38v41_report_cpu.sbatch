#!/bin/bash
#SBATCH --job-name=s5-report
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=20:00:00
#SBATCH --output=logs/s5-report-%j.out
#SBATCH --error=logs/s5-report-%j.err

set -euo pipefail

cd "$HOME/mas_rl_detective"

PY="$HOME/.conda/envs/masdetective/bin/python"

"$PY" \
  analysis/report_simple5_q38v41.py \
  | tee \
  artifacts/v14_R7_q38selectorV41_G8_simple5_v1/final_comparison.txt
