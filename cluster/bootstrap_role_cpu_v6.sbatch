#!/bin/bash
#SBATCH --partition main
#SBATCH --time 0-02:00:00
#SBATCH --job-name role-bootstrap-v6
#SBATCH --output logs/role-bootstrap-v6-%J.out

set -euo pipefail

cd "$HOME/mas_rl_detective"
mkdir -p logs artifacts outputs

PY="$HOME/.conda/envs/masdetective/bin/python"

set -a
source .env
set +a

"$PY" -u -m training.bootstrap_role_estimator \
  --limit-systems 8 \
  --execute
