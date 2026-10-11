#!/bin/bash
#SBATCH --job-name=rd-seal-exec
#SBATCH --partition=main
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --time=20:00:00
#SBATCH --output=logs/rd-seal-exec-%j.out
#SBATCH --error=logs/rd-seal-exec-%j.err

set -euo pipefail

PROJECT="$HOME/mas_rl_detective"
ROOT="$PROJECT/artifacts/role_direct_sealed_v1"
PY="$HOME/.conda/envs/masdetective/bin/python"

cd "$PROJECT"

export PYTHONPATH="$PROJECT:${PYTHONPATH:-}"

set -a
source "$PROJECT/.env"
set +a

test -s "$ROOT/manifest_privileged.json"
test -s "$ROOT/candidates_dev_round_01.jsonl"

echo "SEALED_EXEC_PREFLIGHT_PASS"

"$PY" -u \
  scripts/adaptive_v11_execute_round.py \
  --round 1 \
  --split dev \
  --root "$ROOT" \
  --manifest "$ROOT/manifest_privileged.json" \
  --workers 5
