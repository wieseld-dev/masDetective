#!/bin/bash
#SBATCH --job-name=la-exec
#SBATCH --partition=main
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=logs/la-exec-%j.out
#SBATCH --error=logs/la-exec-%j.err

set -euo pipefail

cd "$HOME/mas_rl_detective"

export PYTHONPATH="$HOME/mas_rl_detective:${PYTHONPATH:-}"
export PATH="$HOME/.conda/envs/masdetective/bin:$PATH"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1

if [ -f .env ]; then
  set -a
  source .env
  set +a
fi

"$HOME/.conda/envs/masdetective/bin/python" \
  -u scripts/adaptive_v11_execute_round.py \
  --round 1 \
  --split dev \
  --root artifacts/local_ambiguity_v1 \
  --manifest artifacts/local_ambiguity_v1/manifest_privileged.json \
  --workers "${LA_API_WORKERS:-8}"
