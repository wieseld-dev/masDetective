#!/bin/bash
#SBATCH --job-name=la-analyze
#SBATCH --partition=main
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/la-analyze-%j.out
#SBATCH --error=logs/la-analyze-%j.err

set -euo pipefail

cd "$HOME/mas_rl_detective"

export PYTHONPATH="$HOME/mas_rl_detective:${PYTHONPATH:-}"
export PATH="$HOME/.conda/envs/masdetective/bin:$PATH"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1

mkdir -p "$HOME/.tmp_masdetective"
export TMPDIR="$HOME/.tmp_masdetective"
export JOBLIB_TEMP_FOLDER="$HOME/.tmp_masdetective"

"$HOME/.conda/envs/masdetective/bin/python" \
  -u scripts/local_ambiguity_analyze_v1.py
