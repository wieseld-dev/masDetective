#!/usr/bin/env python3

import json
import hashlib
from collections import defaultdict
from pathlib import Path

ROOT = Path("artifacts/v137_presence_v1")
TRAIN = ROOT / "dataset_v1/train_v137.jsonl"
DEV = ROOT / "dataset_v1/dev_v137.jsonl"

OUT = Path("artifacts/v138_pairwise_v1")
OUT.mkdir(parents=True, exist_ok=True)


def readj(path):
    with Path(path).open() as f:
        return [
            json.loads(x)
            for x in f
            if x.strip()
        ]


def writej(path, rows):
    with Path(path).open("w") as f:
        for x in rows:
            f.write(
                json.dumps(
                    x,
                    ensure_ascii=False,
                )
                + "\n"
            )


def sha(path):
    return hashlib.sha256(
        Path(path).read_bytes()
    ).hexdigest()


train = readj(TRAIN)
dev = readj(DEV)

# Only the explicitly matched fresh T1 pairs.
t1 = [
    x for x in train
    if x["probe_family"] == "T1"
]

if len(t1) != 190:
    raise RuntimeError(
        f"Expected 190 T1 rows, got {len(t1)}"
    )

pairs = defaultdict(list)

for x in t1:
    pid = x.get("pair_id")

    if not pid:
        raise RuntimeError(
            f"Missing pair_id: {x['example_id']}"
        )

    pairs[pid].append(x)


out = []

for pid in sorted(pairs):
    rr = pairs[pid]

    if len(rr) != 2:
        raise RuntimeError(
            f"{pid}: expected 2 rows, got {len(rr)}"
        )

    pos = [
        x for x in rr
        if int(x["label"]) == 1
    ]

    neg = [
        x for x in rr
        if int(x["label"]) == 0
    ]

    if len(pos) != 1 or len(neg) != 1:
        raise RuntimeError(
            f"{pid}: expected one positive and one negative"
        )

    pos = pos[0]
    neg = neg[0]

    if pos["target_role"] != neg["target_role"]:
        raise RuntimeError(
            f"{pid}: target mismatch"
        )

    if pos["query"] != neg["query"]:
        raise RuntimeError(
            f"{pid}: query mismatch"
        )

    if (
        pos["public_role_description"]
        != neg["public_role_description"]
    ):
        raise RuntimeError(
            f"{pid}: description mismatch"
        )

    out.append({
        "pair_id": pid,

        "target_role":
            pos["target_role"],

        "query":
            pos["query"],

        "positive_example_id":
            pos["example_id"],

        "negative_example_id":
            neg["example_id"],

        "positive_system_id":
            pos["system_id"],

        "negative_system_id":
            neg["system_id"],

        "positive_input":
            pos["model_input"],

        "negative_input":
            neg["model_input"],
    })


if len(out) != 95:
    raise RuntimeError(
        f"Expected 95 pairs, got {len(out)}"
    )

if len({
    x["positive_system_id"]
    for x in out
} | {
    x["negative_system_id"]
    for x in out
}) != 190:
    raise RuntimeError(
        "TRAIN pair systems are not unique"
    )


# Preserve DEV exactly as V13.7.
train_systems = {
    x["positive_system_id"]
    for x in out
} | {
    x["negative_system_id"]
    for x in out
}

dev_systems = {
    x["system_id"]
    for x in dev
}

if train_systems & dev_systems:
    raise RuntimeError(
        "TRAIN/DEV system leakage"
    )


PAIR_OUT = OUT / "train_pairs.jsonl"
DEV_OUT = OUT / "dev_v138.jsonl"

writej(PAIR_OUT, out)
writej(DEV_OUT, dev)

meta = {
    "version":
        "v138_pairwise_v1",

    "train_pairs":
        len(out),

    "train_individual_examples":
        2 * len(out),

    "dev_rows":
        len(dev),

    "dev_systems":
        len(dev_systems),

    "train_pairs_sha256":
        sha(PAIR_OUT),

    "dev_sha256":
        sha(DEV_OUT),

    "objective":
        "same-target BCE + pairwise logistic ranking",

    "internal_test_used":
        False,
}

(
    OUT / "meta.json"
).write_text(
    json.dumps(
        meta,
        indent=2,
        sort_keys=True,
    )
)

print("=" * 90)
print("V13.8 PAIRWISE DATASET")
print("=" * 90)

print("pairs             :", len(out))
print("individual train  :", 2 * len(out))
print("DEV rows          :", len(dev))
print("DEV systems       :", len(dev_systems))

print()
print(
    "roles represented:",
    len({
        x["target_role"]
        for x in out
    })
)

print(
    "TRAIN/DEV overlap:",
    len(
        train_systems
        & dev_systems
    )
)

print()
print("PAIR INTEGRITY: PASS")
print("SYSTEM LEAKAGE: 0")
print("INTERNAL TEST USED: NO")
print("V13.8 DATASET BUILD: PASS")
