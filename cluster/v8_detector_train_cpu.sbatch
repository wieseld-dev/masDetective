#!/usr/bin/env bash
#SBATCH --job-name=v8-detfit
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=logs/v8-detfit-%j.out
set -euo pipefail
cd "$HOME/mas_rl_detective"
mkdir -p logs artifacts
PY="$HOME/.conda/envs/masdetective/bin/python"
"$PY" -u -m training.train_detector_v8
