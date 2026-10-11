#!/bin/bash
#SBATCH --job-name=eig-exec
#SBATCH --partition=main
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=logs/eig-exec-%j.out
#SBATCH --error=logs/eig-exec-%j.err
set -euo pipefail
cd "$HOME/mas_rl_detective"; export PYTHONPATH="$HOME/mas_rl_detective:${PYTHONPATH:-}"; export PATH="$HOME/.conda/envs/masdetective/bin:$PATH"; export PYTHONNOUSERSITE=1; export PYTHONUNBUFFERED=1
PY="$HOME/.conda/envs/masdetective/bin/python"
"$PY" -u scripts/adaptive_v11_execute_round.py --round "$EIG_ROUND" --split dev --root "$EIG_ROOT" --manifest "$EIG_ROOT/manifest_privileged.json" --workers "${EIG_WORKERS:-4}" --retries 3
