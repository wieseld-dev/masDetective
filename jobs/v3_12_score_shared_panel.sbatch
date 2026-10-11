#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --job-name=v312-score
#SBATCH --output=logs/v312-score-%J.out
#SBATCH --error=logs/v312-score-%J.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

ROOT="artifacts/eqs_v3_11e_final_shared_panel"

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
echo "FROZEN V13.4 + V13.5"
echo "NO TARGET API CALLS"

"$PYTHON" -u \
  scripts/adaptive_v11_score_round.py \
  --round 1 \
  --split train \
  --root "$ROOT"

echo
echo "=== SCORE CHECK ==="

"$PYTHON" - <<'PY'
import json
from pathlib import Path

p = Path(
    "artifacts/eqs_v3_11e_final_shared_panel/"
    "scored_train_round_01.jsonl"
)

rows = [
    json.loads(x)
    for x in p.read_text().splitlines()
    if x.strip()
]

print("scored =", len(rows))

assert len(rows) == 576
PY

echo "END=$(date)"
