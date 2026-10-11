#!/bin/bash
#SBATCH --partition=main
#SBATCH --job-name=rda2-cpu
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --time=20:00:00
#SBATCH --output=logs/rda2-%x-%j.out
#SBATCH --error=logs/rda2-%x-%j.err
set -euo pipefail
cd /home/wieseld/mas_rl_detective
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
if [ "$ROLE_STAGE" = execute ]; then set -a; source .env; set +a; fi
exec /home/wieseld/.conda/envs/masdetective/bin/python -u \
 analysis/role_direct_adaptive_v2/attack.py "$ROLE_STAGE" --round "$ROLE_ROUND" --root "$ROLE_ROOT" --manifest "$ROLE_MANIFEST"
