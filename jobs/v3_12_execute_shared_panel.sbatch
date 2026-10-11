#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=08:00:00
#SBATCH --job-name=v312-exec
#SBATCH --output=logs/v312-exec-%J.out
#SBATCH --error=logs/v312-exec-%J.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

ROOT="artifacts/eqs_v3_11e_final_shared_panel"
MANIFEST="artifacts/v14_adaptive_role_v1/manifest_privileged.json"

ENV_FILE="$HOME/mas_rl_detective/.env"

if [ ! -f "$ENV_FILE" ]; then
    echo "ERROR: missing $ENV_FILE"
    exit 1
fi

set -a
source "$ENV_FILE"
set +a

: "${OPENAI_API_KEY:?OPENAI_API_KEY missing from .env}"

echo "OPENAI_API_KEY: LOADED FROM .env"

PKG_DIR="$(
    find "$PWD" \
      -maxdepth 5 \
      -type d \
      -name adaptive_v11_active \
      -print \
      -quit
)"

if [ -n "$PKG_DIR" ]; then
    export PYTHONPATH="$(dirname "$PKG_DIR"):$PWD:${PYTHONPATH:-}"
else
    export PYTHONPATH="$PWD:${PYTHONPATH:-}"
fi

PYTHON="$HOME/.conda/envs/masdetective/bin/python"

echo "START=$(date)"
echo "TARGET PAIRS=576"
echo "WORKERS=4"

"$PYTHON" -u \
  scripts/adaptive_v11_execute_round.py \
  --round 1 \
  --split train \
  --root "$ROOT" \
  --manifest "$MANIFEST" \
  --workers 4 \
  --retries 3

echo
echo "=== EXECUTION CHECK ==="

"$PYTHON" - <<'PY'
import json
from pathlib import Path

p = Path(
    "artifacts/eqs_v3_11e_final_shared_panel/"
    "responses_train_round_01.jsonl"
)

rows = [
    json.loads(x)
    for x in p.read_text().splitlines()
    if x.strip()
]

ok = sum(
    x.get("execution_status") == "ok"
    for x in rows
)

hits = sum(
    bool(x.get("cache_hit"))
    for x in rows
)

print("responses =", len(rows))
print("ok        =", ok)
print("cache_hits=", hits)

assert len(rows) == 576
assert ok == 576
PY

echo "END=$(date)"
