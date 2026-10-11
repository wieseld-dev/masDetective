#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --job-name=v312-scorefix
#SBATCH --output=logs/v312-scorefix-%J.out
#SBATCH --error=logs/v312-scorefix-%J.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

PYTHON="$HOME/.conda/envs/masdetective/bin/python"
ROOT="artifacts/eqs_v3_11e_final_shared_panel"

echo "START=$(date)"
echo "DIRECT FROZEN V13.4 + V13.5 RECOVERY"
echo "NO TARGET API CALLS"

"$PYTHON" -u adaptive_score_v135.py \
  --input "$ROOT/score_input_train_round_01.jsonl" \
  --output "$ROOT/scored_train_round_01.jsonl"

echo
echo "=== OUTPUT CHECK ==="

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

print("scored rows =", len(rows))
print("systems     =", len({x["system_id"] for x in rows}))
print("split values=", sorted({x["split"] for x in rows}))

assert len(rows) == 576
assert len({x["system_id"] for x in rows}) == 24
assert {x["split"] for x in rows} == {"train"}
PY

echo "END=$(date)"
