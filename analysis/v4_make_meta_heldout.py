#!/usr/bin/env python3

import json
import random
from pathlib import Path
import numpy as np

ROOT = Path("artifacts/v3_coverage_elimination_pilot")
DEV = Path("artifacts/v4_manual_meta_calibration/calibration_set.jsonl")
OUT = Path("artifacts/v4_meta_heldout")
OUT.mkdir(parents=True, exist_ok=True)

OUTFILE = OUT / "heldout_set.jsonl"


def read_jsonl(p):
    with p.open(encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def key(x):
    return (
        int(x["round"]),
        str(x["session_id"]),
        int(x["candidate_index"]),
    )


if OUTFILE.exists():
    rows = read_jsonl(OUTFILE)
    print("already exists:", OUTFILE)
    print("N:", len(rows))
    raise SystemExit(0)

all_rows = read_jsonl(
    ROOT / "membership_relabel_candidates.jsonl"
)
dev = read_jsonl(DEV)

assert len(all_rows) == 384
assert len(dev) == 80

dev_keys = {key(x) for x in dev}
dev_queries = {
    " ".join(str(x["query"]).split()).lower()
    for x in dev
}

remaining = []

for x in all_rows:
    if key(x) in dev_keys:
        continue

    q = str(
        x.get("query")
        or x.get("prompt")
        or x.get("completion")
        or ""
    ).strip()

    # membership_relabel may not carry query in some versions.
    # Recover it from V3 response files if needed below.
    remaining.append({
        **x,
        "_query_tmp": q,
    })

# Load executed responses to recover canonical query text.
response_map = {}

for r in range(1, 7):
    p = ROOT / f"responses_train_round_{r:02d}.jsonl"
    if not p.exists():
        continue

    for z in read_jsonl(p):
        response_map[key(z)] = str(z["query"]).strip()

clean = []

for x in remaining:
    q = response_map.get(key(x), x["_query_tmp"]).strip()
    assert q, key(x)

    qnorm = " ".join(q.split()).lower()

    # Extra leakage protection: exact query appearing in dev is excluded.
    if qnorm in dev_queries:
        continue

    clean.append({
        "round": int(x["round"]),
        "session_id": str(x["session_id"]),
        "candidate_index": int(x["candidate_index"]),
        "query": q,
        "reward": float(x["delta_sz"] + x["delta_margin"]),
        "delta_sz": float(x["delta_sz"]),
        "delta_margin": float(x["delta_margin"]),
        "maxfalse_drop": float(x["maxfalse_drop"]),
    })

assert len(clean) >= 304, len(clean)

selected = []
used = set()

# 20 highest-reward remaining examples.
for x in sorted(clean, key=lambda z: z["reward"], reverse=True):
    if len(selected) == 20:
        break
    selected.append({**x, "bucket": "TOP"})
    used.add(key(x))

# 20 lowest-reward.
for x in sorted(clean, key=lambda z: z["reward"]):
    if sum(z["bucket"] == "BOTTOM" for z in selected) == 20:
        break
    if key(x) in used:
        continue
    selected.append({**x, "bucket": "BOTTOM"})
    used.add(key(x))

# 20 near median reward.
available = [x for x in clean if key(x) not in used]
median = float(np.median([x["reward"] for x in available]))

for x in sorted(
    available,
    key=lambda z: abs(z["reward"] - median)
):
    if sum(z["bucket"] == "MIDDLE" for z in selected) == 20:
        break
    if key(x) in used:
        continue
    selected.append({**x, "bucket": "MIDDLE"})
    used.add(key(x))

# 20 random from everything still unused.
available = [x for x in clean if key(x) not in used]
rng = random.Random(20260923)

for x in rng.sample(available, 20):
    selected.append({**x, "bucket": "RANDOM"})
    used.add(key(x))

assert len(selected) == 80

# Hide bucket/order while manually labeling.
random.Random(8675309).shuffle(selected)

with OUTFILE.open("w", encoding="utf-8") as f:
    for x in selected:
        f.write(json.dumps(x, ensure_ascii=False) + "\n")

print("wrote:", OUTFILE)
print("N:", len(selected))
print("unique keys:", len({key(x) for x in selected}))
print(
    "buckets:",
    {
        b: sum(x["bucket"] == b for x in selected)
        for b in ["TOP", "MIDDLE", "RANDOM", "BOTTOM"]
    }
)
