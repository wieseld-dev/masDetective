#!/usr/bin/env bash
set -euo pipefail

cd ~/mas_rl_detective

SEED="${1:?seed required}"
ARM="${2:?arm required}"
ADAPTER="${3:?adapter required}"

SRC="artifacts/v3_coverage_elimination_pilot"
V11="artifacts/v11_policy"

BASE="artifacts/v6_final_heldout"
ROOT="$BASE/seed_${SEED}/${ARM}"

echo "=================================================="
echo "V6 FINAL HELDOUT GENERATION"
echo "seed    : $SEED"
echo "arm     : $ARM"
echo "adapter : $ADAPTER"
echo "root    : $ROOT"
echo "=================================================="

for x in \
  "$SRC/states_dev_round_00.jsonl" \
  "$SRC/coverage_dev_round_00.npz" \
  "$ADAPTER" \
  "$V11"
do
    if [[ ! -e "$x" ]]; then
        echo "FATAL missing: $x" >&2
        exit 2
    fi
done

NDEV=$(grep -cve '^[[:space:]]*$' \
  "$SRC/states_dev_round_00.jsonl")

if [[ "$NDEV" -ne 8 ]]; then
    echo "FATAL expected 8 dev systems, got $NDEV" >&2
    exit 3
fi

OUTFILE="$ROOT/candidates_dev_round_01.jsonl"

if [[ -e "$OUTFILE" ]]; then
    echo "FATAL output already exists: $OUTFILE" >&2
    exit 4
fi

mkdir -p "$ROOT"

cp "$SRC/states_dev_round_00.jsonl" "$ROOT/"
cp "$SRC/coverage_dev_round_00.npz" "$ROOT/"

if [[ -f "$SRC/posterior_dev_round_00.npz" ]]; then
    cp "$SRC/posterior_dev_round_00.npz" "$ROOT/"
fi

python analysis/v3_generate.py \
  --round 1 \
  --split dev \
  --root "$ROOT" \
  --adapter "$ADAPTER" \
  --v11-reference "$V11" \
  --group-size 8 \
  --seed "$SEED" \
  --temperature 0.85 \
  --top-p 0.95 \
  --max-new-tokens 192 \
  --max-seq-length 4096 \
  --local-retries 1

python - "$OUTFILE" "$SEED" "$ARM" <<'PY'
import json, sys
from pathlib import Path
from collections import Counter

p = Path(sys.argv[1])
seed = sys.argv[2]
arm = sys.argv[3]

xs = [
    json.loads(x)
    for x in p.open()
    if x.strip()
]

assert len(xs) == 64, (arm, len(xs))

sessions = {str(x["session_id"]) for x in xs}
indices = {int(x["candidate_index"]) for x in xs}
rounds = {int(x["round"]) for x in xs}
attempts = {int(x["local_attempts"]) for x in xs}

per_session = Counter(
    str(x["session_id"])
    for x in xs
)

assert len(sessions) == 8
assert indices == set(range(8))
assert rounds == {1}
assert attempts == {1}
assert set(per_session.values()) == {8}

print(
    f"PASS seed={seed} arm={arm}: "
    f"rows=64 systems=8 candidates/system=8 "
    f"valid={sum(bool(x.get('valid')) for x in xs)}/64"
)
PY
