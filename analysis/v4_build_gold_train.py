#!/usr/bin/env python3

import json
from pathlib import Path
from collections import Counter
import numpy as np

JUDG = Path("artifacts/v4_generator/candidate_judgments.jsonl")
V3 = Path("artifacts/v3_coverage_elimination_pilot")
OUT = Path("artifacts/v4_gold_train_v1")

OUT.mkdir(parents=True, exist_ok=True)


def read_jsonl(p):
    with Path(p).open(encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def key(x):
    return (
        int(x["round"]),
        str(x["session_id"]),
        int(x["candidate_index"]),
    )


rows = [
    x for x in read_jsonl(JUDG)
    if "error" not in x
]

# Same 48 candidates manually audited.
gold48 = [
    x for x in rows
    if x["label"] == "NON_META"
    and float(x["delta_sz"]) > 0
    and float(x["delta_margin"]) > 0
    and float(x["maxfalse_drop"]) > 0
]

gold48.sort(
    key=lambda x: float(x["info_reward"]),
    reverse=True,
)

assert len(gold48) == 48, len(gold48)

# Human audit:
# 06 and 39 = META
# 20 = AMBIGUOUS
EXCLUDE = {6, 20, 39}

gold = [
    x for i, x in enumerate(gold48, 1)
    if i not in EXCLUDE
]

assert len(gold) == 45, len(gold)

gold_by_key = {key(x): x for x in gold}
assert len(gold_by_key) == 45

# Original generator training rows.
train_by_key = {}

for rnd in range(1, 7):
    p = V3 / f"train_rows_round_{rnd:02d}.jsonl"

    for x in read_jsonl(p):
        train_by_key[key(x)] = x

missing = [
    k for k in gold_by_key
    if k not in train_by_key
]

assert not missing, missing

# Reward used only to determine positive update strength.
scores = np.asarray(
    [
        float(x["info_reward"])
        for x in gold
    ],
    dtype=float,
)

q10, q90 = np.quantile(scores, [0.10, 0.90])

assert q90 > q10

def weight(score):
    z = (
        np.clip(score, q10, q90) - q10
    ) / (q90 - q10)

    # Every human-Gold example stays positive;
    # strongest examples receive at most 2x weakest.
    return float(0.5 + 0.5 * z)


out_by_round = {
    r: [] for r in range(1, 7)
}

manifest = []

for g in gold:
    k = key(g)
    old = dict(train_by_key[k])

    w = weight(float(g["info_reward"]))

    rec = dict(old)

    # IMPORTANT:
    # Positive-only local update.
    rec["local_advantage"] = w

    # MC return is disabled during training anyway.
    rec["reward"] = float(g["info_reward"])
    rec["chosen_for_continuation"] = False

    # Anchor KL to the epoch2 policy that generated
    # this candidate, instead of the older V11 ref.
    rec["ref_logp"] = float(rec["old_logp"])

    rec["v4_gold"] = True
    rec["v4_info_reward"] = float(g["info_reward"])
    rec["v4_delta_sz"] = float(g["delta_sz"])
    rec["v4_delta_margin"] = float(g["delta_margin"])
    rec["v4_maxfalse_drop"] = float(g["maxfalse_drop"])
    rec["v4_gold_weight"] = w

    rnd = int(g["round"])
    out_by_round[rnd].append(rec)

    manifest.append({
        "round": rnd,
        "session_id": g["session_id"],
        "candidate_index": int(g["candidate_index"]),
        "query": g["query"],
        "info_reward": float(g["info_reward"]),
        "delta_sz": float(g["delta_sz"]),
        "delta_margin": float(g["delta_margin"]),
        "maxfalse_drop": float(g["maxfalse_drop"]),
        "weight": w,
    })


for rnd in range(1, 7):
    p = OUT / f"train_rows_round_{rnd:02d}.jsonl"

    with p.open("w", encoding="utf-8") as f:
        for x in out_by_round[rnd]:
            f.write(
                json.dumps(x, ensure_ascii=False)
                + "\n"
            )


with (OUT / "gold_manifest.jsonl").open(
    "w",
    encoding="utf-8",
) as f:
    for x in manifest:
        f.write(
            json.dumps(x, ensure_ascii=False)
            + "\n"
        )


print("=== V4 GOLD TRAIN BUILD ===")
print("rows:", len(manifest))
print(
    "distinct states:",
    len({
        (x["round"], x["session_id"])
        for x in manifest
    })
)
print(
    "round counts:",
    dict(
        sorted(
            Counter(
                x["round"] for x in manifest
            ).items()
        )
    )
)
print(f"reward q10/q90: {q10:+.5f} / {q90:+.5f}")
print(
    "weight range:",
    f"{min(x['weight'] for x in manifest):.3f}",
    "to",
    f"{max(x['weight'] for x in manifest):.3f}",
)

# Safety checks.
all_train = [
    x
    for r in range(1, 7)
    for x in out_by_round[r]
]

assert len(all_train) == 45
assert all(
    abs(float(x["ref_logp"]) - float(x["old_logp"])) < 1e-12
    for x in all_train
)
assert all(
    float(x["local_advantage"]) > 0
    for x in all_train
)

print("ref_logp == old_logp: YES")
print("all advantages positive: YES")
print("wrote:", OUT)
