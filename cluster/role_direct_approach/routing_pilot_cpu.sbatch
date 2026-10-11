#!/bin/bash
#SBATCH --job-name=rd-route
#SBATCH --partition=main
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --time=20:00:00
#SBATCH --output=logs/rd-route-%j.out
#SBATCH --error=logs/rd-route-%j.err

set -euo pipefail

PROJECT="$HOME/mas_rl_detective"
ROOT="$PROJECT/artifacts/role_direct_routing_pilot_v1"
PY="$HOME/.conda/envs/masdetective/bin/python"

cd "$PROJECT"

export PYTHONPATH="$PROJECT:${PYTHONPATH:-}"

set -a
source "$PROJECT/.env"
set +a

echo "PWD=$PWD"
echo "ROOT=$ROOT"

test -f "$PROJECT/.env"
test -f "$ROOT/manifest_privileged.json"
test -f "$ROOT/candidates_dev_round_01.jsonl"

echo "PRE_EXEC_PASS"

"$PY" -u \
  scripts/adaptive_v11_execute_round.py \
  --round 1 \
  --split dev \
  --root "$ROOT" \
  --manifest "$ROOT/manifest_privileged.json" \
  --workers 5
