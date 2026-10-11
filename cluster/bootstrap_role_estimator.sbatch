#!/bin/bash
#SBATCH --partition main
#SBATCH --time 0-02:00:00
#SBATCH --job-name mas-role-bootstrap
#SBATCH --output logs/mas-role-bootstrap-%J.out
#SBATCH --gpus=gtx_1080:1
set -euo pipefail
cd "$HOME/mas_rl_detective"
mkdir -p logs artifacts outputs
PY="$HOME/.conda/envs/masdetective/bin/python"
set -a; source .env; set +a
"$PY" -u -m training.bootstrap_role_estimator --limit-systems 8 --execute
