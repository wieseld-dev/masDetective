#!/usr/bin/env bash
set -euo pipefail

cd ~/mas_rl_detective

ROOT="artifacts/v3_coverage_elimination_pilot"
OUT="artifacts/v4_1_heldout_ab"

E2="artifacts/robust_visible_generator_v1_epoch2/policy"
V41="artifacts/v4_1_gold_generator/policy"
V11="artifacts/v11_policy"

E2ROOT="$OUT/epoch2"
V41ROOT="$OUT/v41"

echo "========================================"
echo " V4.1 FRESH HELD-OUT A/B"
echo "========================================"
date

# ---------- PREFLIGHT ----------
for x in \
  "$ROOT/states_dev_round_00.jsonl" \
  "$ROOT/coverage_dev_round_00.npz" \
  "$E2" \
  "$V41" \
  "$V11"
do
  if [[ ! -e "$x" ]]; then
    echo "FATAL missing: $x" >&2
    exit 2
  fi
done

NDEV=$(grep -cve '^[[:space:]]*$' "$ROOT/states_dev_round_00.jsonl")
echo "held-out dev systems: $NDEV"

if [[ "$NDEV" -ne 8 ]]; then
  echo "FATAL: expected 8 untouched dev systems, found $NDEV" >&2
  exit 3
fi

# Fresh clean directories. Never touch original pilot.
rm -rf "$OUT"
mkdir -p "$E2ROOT" "$V41ROOT"

for D in "$E2ROOT" "$V41ROOT"; do
  cp "$ROOT/states_dev_round_00.jsonl" "$D/"
  cp "$ROOT/coverage_dev_round_00.npz" "$D/"

  # Not required by generation, but archive it if available.
  if [[ -f "$ROOT/posterior_dev_round_00.npz" ]]; then
    cp "$ROOT/posterior_dev_round_00.npz" "$D/"
  fi
done

echo
echo "=== EPOCH2 GENERATION ==="

python analysis/v3_generate.py \
  --round 1 \
  --split dev \
  --root "$E2ROOT" \
  --adapter "$E2" \
  --v11-reference "$V11" \
  --group-size 4 \
  --seed 20260924 \
  --temperature 0.85 \
  --top-p 0.95 \
  --max-new-tokens 192 \
  --max-seq-length 4096 \
  --local-retries 1

echo
echo "=== V4.1 GENERATION ==="

python analysis/v3_generate.py \
  --round 1 \
  --split dev \
  --root "$V41ROOT" \
  --adapter "$V41" \
  --v11-reference "$V11" \
  --group-size 4 \
  --seed 20260924 \
  --temperature 0.85 \
  --top-p 0.95 \
  --max-new-tokens 192 \
  --max-seq-length 4096 \
  --local-retries 1

echo
echo "=== STRICT PAIRED COMPARISON ==="

python - <<'PY'
import json
from pathlib import Path
from collections import Counter

OUT = Path("artifacts/v4_1_heldout_ab")
A = OUT / "epoch2" / "candidates_dev_round_01.jsonl"
B = OUT / "v41"   / "candidates_dev_round_01.jsonl"

def read_jsonl(p):
    with p.open() as f:
        return [json.loads(x) for x in f if x.strip()]

def key(x):
    return (
        str(x["session_id"]),
        int(x["round"]),
        int(x["candidate_index"]),
    )

a = read_jsonl(A)
b = read_jsonl(B)

assert len(a) == 32, f"Epoch2 expected 32 rows, got {len(a)}"
assert len(b) == 32, f"V4.1 expected 32 rows, got {len(b)}"

Aby = {key(x): x for x in a}
Bby = {key(x): x for x in b}

assert len(Aby) == 32, "duplicate Epoch2 keys"
assert len(Bby) == 32, "duplicate V4.1 keys"
assert set(Aby) == set(Bby), "A/B context keys differ"

rows = []
prompt_same = 0
seed_same = 0
query_same = 0
both_valid = 0
e2_valid = 0
v41_valid = 0

for k in sorted(Aby):
    x = Aby[k]
    y = Bby[k]

    ps = str(x["prompt"]) == str(y["prompt"])
    ss = int(x["seed"]) == int(y["seed"])
    qs = str(x["query"]) == str(y["query"])

    # With local-retries=1 this MUST be paired on exactly one attempt.
    assert int(x["local_attempts"]) == 1
    assert int(y["local_attempts"]) == 1

    prompt_same += ps
    seed_same += ss
    query_same += qs
    e2_valid += bool(x["valid"])
    v41_valid += bool(y["valid"])
    both_valid += bool(x["valid"]) and bool(y["valid"])

    rows.append({
        "session_id": k[0],
        "round": k[1],
        "candidate_index": k[2],
        "prompt_identical": ps,
        "seed_identical": ss,
        "seed": int(x["seed"]),
        "query_identical": qs,
        "epoch2_query": str(x["query"]),
        "v41_query": str(y["query"]),
        "epoch2_valid": bool(x["valid"]),
        "v41_valid": bool(y["valid"]),
        "epoch2_invalid_reason": str(x.get("invalid_reason", "")),
        "v41_invalid_reason": str(y.get("invalid_reason", "")),
        "prompt": str(x["prompt"]),
    })

# Pairing is non-negotiable.
assert prompt_same == 32, f"FATAL prompt mismatch {prompt_same}/32"
assert seed_same == 32, f"FATAL seed mismatch {seed_same}/32"

changed = [x for x in rows if not x["query_identical"]]

with (OUT / "comparison.jsonl").open("w") as f:
    for x in rows:
        f.write(json.dumps(x, ensure_ascii=False) + "\n")

summary = {
    "heldout_systems": len(set(x["session_id"] for x in rows)),
    "contexts": len(rows),
    "prompt_identical": prompt_same,
    "seed_identical": seed_same,
    "query_identical": query_same,
    "query_changed": len(changed),
    "change_rate": len(changed) / len(rows),
    "epoch2_valid": e2_valid,
    "v41_valid": v41_valid,
    "both_valid": both_valid,
    "changes_by_candidate_index": dict(
        Counter(str(x["candidate_index"]) for x in changed)
    ),
}

(OUT / "summary.json").write_text(
    json.dumps(summary, indent=2),
    encoding="utf-8",
)

print("========================================")
print(" HELD-OUT A/B SUMMARY")
print("========================================")
print(f"held-out systems:       {summary['heldout_systems']}")
print(f"contexts:               {len(rows)}")
print(f"identical prompts:      {prompt_same}/32")
print(f"identical seeds:        {seed_same}/32")
print(f"Epoch2 == V4.1 query:   {query_same}/32 = {query_same/32:.3f}")
print(f"CHANGED:                {len(changed)}/32 = {len(changed)/32:.3f}")
print(f"Epoch2 valid:           {e2_valid}/32")
print(f"V4.1 valid:             {v41_valid}/32")
print(f"both valid:             {both_valid}/32")

print()
print("=== ALL CHANGED PAIRS ===")
if not changed:
    print("NONE")
else:
    for i, x in enumerate(changed, 1):
        print(
            f"\n--- {i}/{len(changed)} | "
            f"{x['session_id']} | j{x['candidate_index']} ---"
        )
        print("E2 :", x["epoch2_query"])
        print("V41:", x["v41_query"])
        print(
            "valid:",
            x["epoch2_valid"],
            "->",
            x["v41_valid"],
        )

print()
print("wrote:", OUT / "comparison.jsonl")
print("wrote:", OUT / "summary.json")
PY

echo
echo "=== COMPLETE ==="
date
