#!/bin/bash
#SBATCH --job-name=pdiag-ana
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=0-00:10:00
#SBATCH --output=logs/pdiag-ana-%j.out
#SBATCH --error=logs/pdiag-ana-%j.err

set -euo pipefail
cd "$HOME/mas_rl_detective"
export PYTHONPATH="$HOME/mas_rl_detective:${PYTHONPATH:-}"

"$HOME/.conda/envs/masdetective/bin/python" -u \
  presence_diag_v1/run.py \
  --phase analyze \
  --root "$PDIAG_ROOT" \
  --belief artifacts/adaptive_v11_active_v1/frozen_belief.joblib
