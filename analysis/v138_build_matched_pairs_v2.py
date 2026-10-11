#!/usr/bin/env python3

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

V137 = Path("artifacts/v137_presence_v1")
TRAIN = V137 / "dataset_v1/train_v137.jsonl"
DEV = V137 / "dataset_v1/dev_v137.jsonl"
CANON = Path("artifacts/canonical_parallel_v1")

OUT = Path("artifacts/v138_pairwise_v2")
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


def jaccard(a, b):
    a = set(a)
    b = set(b)

    if not (a or b):
        return 1.0

    return len(a & b) / len(a | b)


train = readj(TRAIN)
dev = readj(DEV)

t0 = [
    x for x in train
    if x["probe_family"] == "T0"
]

t1 = [
    x for x in train
    if x["probe_family"] == "T1"
]

if len(t0) != 720:
    raise RuntimeError(
        f"Expected 720 T0 rows, got {len(t0)}"
    )

if len(t1) != 190:
    raise RuntimeError(
        f"Expected 190 T1 rows, got {len(t1)}"
    )


# ------------------------------------------------------------
# Recover T0 K/mode/truth metadata from canonical responses.
# ------------------------------------------------------------

q_to_role = {}

for x in t0:
    q = x["query"]
    role = x["target_role"]

    if q in q_to_role and q_to_role[q] != role:
        raise RuntimeError(
            f"T0 query maps to multiple roles: {q}"
        )

    q_to_role[q] = role


canon_meta = {}

for cohort in [
    "k3",
    "k4",
    "high",
]:
    rp = (
        CANON
        / cohort
        / "responses_train_round_01.jsonl"
    )

    for r in readj(rp):
        sid = str(
            r.get("system_id")
            or r.get("session_id")
        )

        q = str(r["query"])

        if q not in q_to_role:
            raise RuntimeError(
                f"Unknown canonical T0 query: {q}"
            )

        role = q_to_role[q]

        truth = set(
            map(
                str,
                r.get(
                    "truth_roles_training_only",
                    [],
                ),
            )
        )

        key = (
            sid,
            role,
        )

        canon_meta[key] = {
            "K": len(truth),
            "mode": str(r["mode"]),
            "truth": truth,
        }


# ------------------------------------------------------------
# T1 pairs already carefully matched.
# ------------------------------------------------------------

pairs = []

t1_groups = defaultdict(list)

for x in t1:
    t1_groups[
        x["pair_id"]
    ].append(x)

if len(t1_groups) != 95:
    raise RuntimeError(
        f"Expected 95 T1 pair IDs, got {len(t1_groups)}"
    )


def make_pair(
    pid,
    family,
    pos,
    neg,
    extra=None,
):
    if pos["target_role"] != neg["target_role"]:
        raise RuntimeError(
            f"{pid}: role mismatch"
        )

    return {
        "pair_id": pid,
        "family": family,

        "target_role":
            pos["target_role"],

        "public_role_description":
            pos["public_role_description"],

        # query retained ONLY for audit.
        # V13.8 model will NOT receive it.
        "query_audit_only":
            pos["query"],

        "positive_example_id":
            pos["example_id"],

        "negative_example_id":
            neg["example_id"],

        "positive_system_id":
            pos["system_id"],

        "negative_system_id":
            neg["system_id"],

        "positive_response":
            pos["final_output"],

        "negative_response":
            neg["final_output"],

        **(extra or {}),
    }


for pid in sorted(t1_groups):
    rr = t1_groups[pid]

    if len(rr) != 2:
        raise RuntimeError(
            f"{pid}: pair size {len(rr)}"
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
            f"{pid}: invalid labels"
        )

    pos = pos[0]
    neg = neg[0]

    if pos["query"] != neg["query"]:
        raise RuntimeError(
            f"{pid}: query mismatch"
        )

    pairs.append(
        make_pair(
            pid,
            "T1",
            pos,
            neg,
        )
    )


# ------------------------------------------------------------
# Build hard matched T0 pairs.
#
# For each installed example, select a UNIQUE absent example
# for the same role/query.
#
# Priority:
#   same mode
#   same K
#   maximum overlap in OTHER installed roles
# ------------------------------------------------------------

t0_by_role = defaultdict(
    lambda: {
        0: [],
        1: [],
    }
)

for x in t0:
    t0_by_role[
        x["target_role"]
    ][
        int(x["label"])
    ].append(x)


t0_pair_count = 0

for role in sorted(t0_by_role):
    positives = sorted(
        t0_by_role[role][1],
        key=lambda x:
            x["system_id"],
    )

    negatives = sorted(
        t0_by_role[role][0],
        key=lambda x:
            x["system_id"],
    )

    unused = set(
        range(len(negatives))
    )

    if len(negatives) < len(positives):
        raise RuntimeError(
            f"{role}: insufficient T0 negatives"
        )

    for pi, pos in enumerate(positives):
        pm = canon_meta[
            (
                pos["system_id"],
                role,
            )
        ]

        p_other = (
            set(pm["truth"])
            - {role}
        )

        candidates = []

        for ni in unused:
            neg = negatives[ni]

            nm = canon_meta[
                (
                    neg["system_id"],
                    role,
                )
            ]

            n_other = set(
                nm["truth"]
            )

            overlap = jaccard(
                p_other,
                n_other,
            )

            cost = (
                int(
                    pm["mode"]
                    != nm["mode"]
                ),

                abs(
                    pm["K"]
                    - nm["K"]
                ),

                -overlap,

                neg["system_id"],
            )

            candidates.append(
                (
                    cost,
                    ni,
                    overlap,
                    neg,
                    nm,
                )
            )

        (
            _,
            ni,
            overlap,
            neg,
            nm,
        ) = min(candidates)

        unused.remove(ni)

        pid = (
            f"T0PAIR::{role}::{pi:02d}"
        )

        pairs.append(
            make_pair(
                pid,
                "T0",
                pos,
                neg,
                {
                    "positive_K_training_only":
                        pm["K"],

                    "negative_K_training_only":
                        nm["K"],

                    "positive_mode_training_only":
                        pm["mode"],

                    "negative_mode_training_only":
                        nm["mode"],

                    "other_role_jaccard_training_only":
                        overlap,
                },
            )
        )

        t0_pair_count += 1


if t0_pair_count != 171:
    raise RuntimeError(
        f"Expected 171 T0 pairs, got {t0_pair_count}"
    )

if len(pairs) != 266:
    raise RuntimeError(
        f"Expected 266 total pairs, got {len(pairs)}"
    )


# ------------------------------------------------------------
# Pair integrity.
# ------------------------------------------------------------

ids = [
    x["pair_id"]
    for x in pairs
]

if len(ids) != len(set(ids)):
    raise RuntimeError(
        "Duplicate pair IDs"
    )

role_counts = Counter(
    x["target_role"]
    for x in pairs
)

if len(role_counts) != 20:
    raise RuntimeError(
        f"Expected 20 roles, got {len(role_counts)}"
    )

if min(role_counts.values()) < 12:
    raise RuntimeError(
        f"Role pair floor below 12: {role_counts}"
    )


# No train/dev system overlap.
train_systems = set()

for x in pairs:
    train_systems.add(
        x["positive_system_id"]
    )
    train_systems.add(
        x["negative_system_id"]
    )

dev_systems = {
    x["system_id"]
    for x in dev
}

overlap = (
    train_systems
    & dev_systems
)

if overlap:
    raise RuntimeError(
        f"TRAIN/DEV system leakage: {len(overlap)}"
    )


PAIR_OUT = (
    OUT / "train_pairs_v2.jsonl"
)

DEV_OUT = (
    OUT / "dev_v138.jsonl"
)

writej(
    PAIR_OUT,
    sorted(
        pairs,
        key=lambda x:
            (
                x["target_role"],
                x["family"],
                x["pair_id"],
            ),
    ),
)

writej(
    DEV_OUT,
    dev,
)


meta = {
    "version":
        "v138_pairwise_v2",

    "T0_pairs":
        t0_pair_count,

    "T1_pairs":
        95,

    "total_pairs":
        len(pairs),

    "roles":
        dict(
            sorted(
                role_counts.items()
            )
        ),

    "train_pairs_sha256":
        sha(PAIR_OUT),

    "dev_sha256":
        sha(DEV_OUT),

    "model_input":
        [
            "target_role",
            "public_role_description",
            "visible_final_response",
        ],

    "raw_query_visible_to_model":
        False,

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
print("V13.8 MATCHED PAIRS V2")
print("=" * 90)

print(
    "T0 matched pairs :",
    t0_pair_count,
)

print(
    "T1 matched pairs :",
    95,
)

print(
    "TOTAL pairs      :",
    len(pairs),
)

print(
    "roles            :",
    len(role_counts),
)

print(
    "min pairs / role :",
    min(role_counts.values()),
)

print(
    "max pairs / role :",
    max(role_counts.values()),
)

print(
    "DEV rows         :",
    len(dev),
)

print(
    "DEV systems      :",
    len(dev_systems),
)

print()
print("RAW QUERY GIVEN TO MODEL: NO")
print("TRAIN/DEV SYSTEM LEAKAGE: 0")
print("INTERNAL TEST USED: NO")
print("V13.8 MATCHED DATASET: PASS")
