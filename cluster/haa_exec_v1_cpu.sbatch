#!/bin/bash
#SBATCH --job-name=haa-exec
#SBATCH --partition=main
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=logs/haa-exec-%j.out
#SBATCH --error=logs/haa-exec-%j.err

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

PY="$HOME/.conda/envs/masdetective/bin/python"

: "${HAA_ROOT:?}"
: "${HAA_ROUND:?}"

"$PY" -u scripts/adaptive_v11_execute_round.py \
  --round "$HAA_ROUND" \
  --split dev \
  --root "$HAA_ROOT" \
  --manifest "$HAA_ROOT/manifest_privileged.json" \
  --workers "${AV11_API_WORKERS:-6}"
