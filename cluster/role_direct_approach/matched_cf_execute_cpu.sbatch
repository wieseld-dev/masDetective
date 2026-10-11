#!/bin/bash
#SBATCH --job-name=rd-cf-exec
#SBATCH --partition=main
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --time=20:00:00
#SBATCH --output=logs/rd-cf-exec-%j.out
#SBATCH --error=logs/rd-cf-exec-%j.err

set -euo pipefail

PROJECT="$HOME/mas_rl_detective"
ROOT="$PROJECT/artifacts/role_direct_matched_cf_v1"
PY="$HOME/.conda/envs/masdetective/bin/python"

cd "$PROJECT"

export PYTHONPATH="$PROJECT:${PYTHONPATH:-}"

set -a
source "$PROJECT/.env"
set +a

test -f "$ROOT/manifest_privileged.json"
test -f "$ROOT/candidates_dev_round_01.jsonl"

echo "MATCHED_CF_EXEC_PREFLIGHT_PASS"

"$PY" -u \
  scripts/adaptive_v11_execute_round.py \
  --round 1 \
  --split dev \
  --root "$ROOT" \
  --manifest "$ROOT/manifest_privileged.json" \
  --workers 5
