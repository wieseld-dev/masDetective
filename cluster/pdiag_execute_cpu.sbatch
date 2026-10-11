#!/bin/bash
#SBATCH --job-name=pdiag-exe
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=0-00:20:00
#SBATCH --output=logs/pdiag-exe-%j.out
#SBATCH --error=logs/pdiag-exe-%j.err

set -euo pipefail
cd "$HOME/mas_rl_detective"
export PYTHONPATH="$HOME/mas_rl_detective:${PYTHONPATH:-}"

set -a
source .env
set +a

"$HOME/.conda/envs/masdetective/bin/python" -u \
  presence_diag_v1/run.py \
  --phase execute \
  --root "$PDIAG_ROOT"
