#!/usr/bin/env python3

import itertools
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


CF = Path(
    "artifacts/eqs_v3_6_counterfactual/"
    "state_query_rows.jsonl"
)

BANK_PATH = Path(
    "artifacts/eqs_v3_2_closed_loop/"
    "pilot/query_bank.json"
)

OUT = Path(
    "artifacts/eqs_v3_10_action_overlap"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)


def read_jsonl(path):
    out = []

    with path.open() as f:
        for line in f:
            line = line.strip()

            if line:
                out.append(
                    json.loads(line)
                )

    return out


def norm(q):
    q = str(q).strip().lower()

    q = re.sub(
        r"\s+",
        " ",
        q,
    )

    q = re.sub(
        r"[\"'`]",
        "",
        q,
    )

    return q


rows = read_jsonl(CF)

systems = defaultdict(set)

original_text = {}

for x in rows:
    sid = str(
        x["session_id"]
    )

    q = norm(
        x["query"]
    )

    systems[sid].add(q)

    original_text.setdefault(
        q,
        str(x["query"]),
    )


sids = sorted(systems)

N = len(sids)

support = Counter()

for sid in sids:
    for q in systems[sid]:
        support[q] += 1


# ------------------------------------------------------------
# Fixed deployment bank
# ------------------------------------------------------------

bank_obj = json.loads(
    BANK_PATH.read_text()
)

bank_raw = bank_obj["queries"]

bank = {
    norm(q)
    for q in bank_raw
}


# ------------------------------------------------------------
# Pairwise Jaccard
# ------------------------------------------------------------

jaccards = []
shared_counts = []

for a, b in itertools.combinations(
    sids,
    2,
):
    A = systems[a]
    B = systems[b]

    inter = len(
        A & B
    )

    union = len(
        A | B
    )

    shared_counts.append(
        inter
    )

    jaccards.append(
        inter / union
        if union
        else 0.0
    )


# ------------------------------------------------------------
# Per-system overlap with fixed bank
# ------------------------------------------------------------

bank_overlap = []

for sid in sids:
    qs = systems[sid]

    hit = len(
        qs & bank
    )

    bank_overlap.append(
        {
            "session_id":
            sid,

            "queries":
            len(qs),

            "bank_overlap":
            hit,

            "fraction":
            hit / len(qs)
            if qs
            else 0.0,
        }
    )


# ------------------------------------------------------------
# Support thresholds
# ------------------------------------------------------------

thresholds = {}

for frac in [
    1.00,
    0.75,
    0.50,
    0.25,
    0.10,
]:
    needed = max(
        1,
        int(
            np.ceil(
                N * frac
            )
        ),
    )

    count = sum(
        1
        for v in support.values()
        if v >= needed
    )

    thresholds[
        f"queries_in_at_least_{int(frac*100)}pct_systems"
    ] = count


common_all = [
    q
    for q, n in support.items()
    if n == N
]


top = sorted(
    support.items(),
    key=lambda z: (
        -z[1],
        z[0],
    ),
)[:30]


summary = {
    "systems":
    N,

    "unique_queries_total":
    len(support),

    "mean_unique_queries_per_system":
    float(
        np.mean([
            len(systems[sid])
            for sid in sids
        ])
    ),

    "exact_queries_shared_by_all_systems":
    len(common_all),

    "max_system_support_for_any_query":
    int(
        max(
            support.values()
        )
    ),

    "max_support_fraction":
    float(
        max(
            support.values()
        ) / N
    ),

    "mean_pairwise_shared_queries":
    float(
        np.mean(
            shared_counts
        )
    ),

    "median_pairwise_shared_queries":
    float(
        np.median(
            shared_counts
        )
    ),

    "mean_pairwise_jaccard":
    float(
        np.mean(
            jaccards
        )
    ),

    "fixed_bank_size":
    len(bank),

    "mean_queries_per_system_in_fixed_bank":
    float(
        np.mean([
            x["bank_overlap"]
            for x in bank_overlap
        ])
    ),

    "mean_fraction_system_pool_in_fixed_bank":
    float(
        np.mean([
            x["fraction"]
            for x in bank_overlap
        ])
    ),

    **thresholds,
}


print("=" * 100)
print("V3.10 — ACTION-SPACE OVERLAP AUDIT")
print("=" * 100)

for k, v in summary.items():
    if isinstance(v, float):
        print(
            f"{k}: {v:.6f}"
        )
    else:
        print(
            f"{k}: {v}"
        )


print()
print("=" * 100)
print("TOP QUERIES BY CROSS-SYSTEM SUPPORT")
print("=" * 100)

for q, n in top:
    print(
        f"{n:2d}/{N} | "
        f"{n/N:.3f} | "
        f"{original_text[q]}"
    )


print()
print("=" * 100)
print("FIXED 512-BANK OVERLAP PER SYSTEM")
print("=" * 100)

for x in bank_overlap:
    print(
        f"{x['session_id']} | "
        f"{x['bank_overlap']:3d}/"
        f"{x['queries']:3d} | "
        f"{x['fraction']:.3f}"
    )


# ------------------------------------------------------------
# Diagnosis
# ------------------------------------------------------------

shared50 = summary[
    "queries_in_at_least_50pct_systems"
]

jacc = summary[
    "mean_pairwise_jaccard"
]

bank_frac = summary[
    "mean_fraction_system_pool_in_fixed_bank"
]


print()
print("=" * 100)
print("DIAGNOSIS")
print("=" * 100)

if (
    shared50 >= 16
    and jacc >= 0.20
):
    diagnosis = "ENOUGH_SHARED_SUPPORT"

    text = (
        "There is meaningful exact cross-system query overlap. "
        "We can construct a shared offline State×Query dataset "
        "without new MAS executions."
    )

elif bank_frac >= 0.50:
    diagnosis = "BANK_ALIGNED_BUT_SPARSE_CROSS_SYSTEM"

    text = (
        "Pools substantially overlap the deployment bank, "
        "but individual systems do not share enough executed "
        "queries for clean cross-system contextual supervision."
    )

else:
    diagnosis = "ACTION_SPACE_MISMATCH"

    text = (
        "Existing counterfactual pools are largely system-specific. "
        "A deliberately shared query bank should be executed across "
        "TRAIN MAS before further selector training."
    )


print(
    f"DIAGNOSIS: {diagnosis}"
)

print(text)


result = {
    "summary":
    summary,

    "top_cross_system_queries":
    [
        {
            "query":
            original_text[q],

            "systems":
            n,

            "fraction":
            n / N,
        }
        for q, n in top
    ],

    "fixed_bank_overlap":
    bank_overlap,

    "diagnosis":
    diagnosis,

    "interpretation":
    text,
}


(
    OUT / "summary.json"
).write_text(
    json.dumps(
        result,
        indent=2,
    )
)

print()
print(
    "SAVED:",
    OUT / "summary.json",
)
