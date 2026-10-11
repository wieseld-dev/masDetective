#!/bin/bash
#SBATCH --job-name=av11-smokecheck
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=logs/av11-smokecheck-%j.out
#SBATCH --error=logs/av11-smokecheck-%j.err

set -euo pipefail

cd "$HOME/mas_rl_detective"

ROOT="${AV11_ROLE_ROOT:-artifacts/adaptive_v11_role_only_v1}"

"$HOME/.conda/envs/masdetective/bin/python" - <<'PY'
import json
from pathlib import Path

root = Path("artifacts/adaptive_v11_role_only_v1")
p = root / "candidates_dev_round_03.jsonl"

rows = [
    json.loads(x)
    for x in p.read_text().splitlines()
    if x.strip()
]

valid = [x for x in rows if x.get("valid")]

print("rows =", len(rows))
print("valid =", len(valid))

for x in rows:
    print(
        x.get("session_id"),
        "valid=", x.get("valid"),
        "attempts=", x.get("local_attempts"),
        "reason=", x.get("invalid_reason"),
        "query=", repr(x.get("query")),
    )

if len(rows) != 8:
    raise SystemExit(
        f"FAIL: expected 8 rows, got {len(rows)}"
    )

if len(valid) != 8:
    raise SystemExit(
        f"FAIL: compact Round-3 smoke only {len(valid)}/8 valid; "
        "API DEV chain intentionally blocked."
    )

print("PASS: 8/8 compact Round-3 probes valid.")
PY
