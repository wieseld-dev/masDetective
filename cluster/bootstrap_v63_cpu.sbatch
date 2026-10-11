#!/bin/bash
#SBATCH --partition main
#SBATCH --time 0-03:00:00
#SBATCH --job-name v63-bootstrap
#SBATCH --output logs/v63-bootstrap-%J.out

set -euo pipefail
cd "$HOME/mas_rl_detective"
mkdir -p logs artifacts outputs
PY="$HOME/.conda/envs/masdetective/bin/python"
set -a
source .env
set +a

"$PY" -u -m training.bootstrap_role_estimator_v63 --execute
