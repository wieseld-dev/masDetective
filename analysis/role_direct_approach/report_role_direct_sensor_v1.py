#!/usr/bin/env python3

import json
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean

from sklearn.metrics import roc_auc_score


ROOT = Path(
    "artifacts/role_direct_sensor_v1"
)

BASELINE_ROOT = Path(
    "artifacts/role_direct_matched_cf_v1"
)


def read_jsonl(path):
    return [
        json.loads(x)
        for x in Path(path)
        .read_text(
            encoding="utf-8"
        )
        .splitlines()
        if x.strip()
    ]


def win(a, b):
    if a > b:
        return 1.0
    if a < b:
        return 0.0
    return 0.5


def bootstrap(vals, n=10000):

    rng = random.Random(
        20261008
    )

    vals = list(
        map(
            float,
            vals,
        )
    )

    out = []

    for _ in range(n):

        sample = [
            vals[
                rng.randrange(
                    len(vals)
                )
            ]
            for _ in vals
        ]

        out.append(
            mean(sample)
        )

    out.sort()

    return [
        out[
            int(
                0.025
                * len(out)
            )
        ],
        out[
            int(
                0.975
                * len(out)
            )
        ],
    ]


rows = []

for fold in range(
    1,
    6,
):

    p = (
        ROOT
        / f"fold_{fold}"
        / "test_predictions.jsonl"
    )

    rr = read_jsonl(
        p
    )

    assert len(rr) == 120, (
        fold,
        len(rr),
    )

    for x in rr:
        x["fold"] = fold

    rows.extend(
        rr
    )


assert len(rows) == 600


# Every matched system pair appears OOF exactly once.
pair_indices = defaultdict(set)

for x in rows:
    pair_indices[
        (
            x["role"],
            x["pair_id"],
        )
    ].add(
        x["fold"]
    )

assert len(pair_indices) == 100

assert all(
    len(v) == 1
    for v in pair_indices.values()
)


global_auc = float(
    roc_auc_score(
        [
            x["label"]
            for x in rows
        ],
        [
            x["logit"]
            for x in rows
        ],
    )
)


by_probe = defaultdict(dict)

for x in rows:

    key = (
        x["role"],
        x["pair_id"],
        x["probe_rank"],
    )

    by_probe[key][
        int(
            x["label"]
        )
    ] = float(
        x["logit"]
    )


assert len(by_probe) == 300


probe_wins = []

for key, d in by_probe.items():

    assert set(d) == {
        0,
        1,
    }

    probe_wins.append(
        win(
            d[1],
            d[0],
        )
    )


by_pair = defaultdict(list)

for (
    role,
    pair_id,
    probe_rank,
), d in by_probe.items():

    by_pair[
        (
            role,
            pair_id,
        )
    ].append({
        "probe_rank":
            probe_rank,

        "pos":
            d[1],

        "neg":
            d[0],
    })


assert len(by_pair) == 100


accumulation = {}

pair_k3 = []

for k in [
    1,
    2,
    3,
]:

    vals = []

    deltas = []

    for key, rr in by_pair.items():

        rr = sorted(
            rr,
            key=lambda x:
                x[
                    "probe_rank"
                ],
        )

        assert len(rr) == 3

        cur = rr[:k]

        p = mean(
            x["pos"]
            for x in cur
        )

        n = mean(
            x["neg"]
            for x in cur
        )

        vals.append(
            win(
                p,
                n,
            )
        )

        deltas.append(
            p - n
        )

    accumulation[
        str(k)
    ] = {
        "paired_win":
            float(
                mean(vals)
            ),

        "ci95":
            bootstrap(
                vals
            ),

        "mean_logit_delta":
            float(
                mean(
                    deltas
                )
            ),
    }

    if k == 3:
        pair_k3 = vals


# ------------------------------------------------------------
# Per-role OOF metrics:
# 5 held-out system pairs × 3 probes = 15 pos / 15 neg.
# ------------------------------------------------------------

per_role = {}

roles = sorted({
    x["role"]
    for x in rows
})


for role in roles:

    rr = [
        x
        for x in rows
        if x["role"]
        == role
    ]

    assert len(rr) == 30

    auc = float(
        roc_auc_score(
            [
                x["label"]
                for x in rr
            ],
            [
                x["logit"]
                for x in rr
            ],
        )
    )

    rpairs = [
        (
            key,
            val,
        )
        for key, val in by_pair.items()
        if key[0] == role
    ]

    assert len(rpairs) == 5

    wins = []

    for key, vv in rpairs:

        vv = sorted(
            vv,
            key=lambda x:
                x[
                    "probe_rank"
                ],
        )

        p = mean(
            x["pos"]
            for x in vv
        )

        n = mean(
            x["neg"]
            for x in vv
        )

        wins.append(
            win(
                p,
                n,
            )
        )

    per_role[role] = {
        "oof_auc":
            auc,

        "pair3_win":
            float(
                mean(
                    wins
                )
            ),
    }


macro_role_auc = float(
    mean(
        x["oof_auc"]
        for x in per_role.values()
    )
)

macro_role_pair = float(
    mean(
        x["pair3_win"]
        for x in per_role.values()
    )
)


report = {
    "experiment":
        "ROLE DIRECT SENSOR V1 — 5-FOLD OOF",

    "oof_rows":
        len(rows),

    "oof_matched_queries":
        len(by_probe),

    "oof_matched_system_pairs":
        len(by_pair),

    "global_auc":
        global_auc,

    "matched_probe_win":
        float(
            mean(
                probe_wins
            )
        ),

    "macro_role_auc":
        macro_role_auc,

    "macro_role_pair3_win":
        macro_role_pair,

    "accumulation":
        accumulation,

    "per_role":
        per_role,
}


# Baselines, if present.
baseline = {}

p = (
    BASELINE_ROOT
    / "matched_cf_report.json"
)

if p.exists():

    old = json.loads(
        p.read_text()
    )

    baseline[
        "raw_v135_probe_win"
    ] = old.get(
        "v135_probe_matched_win"
    )

    baseline[
        "raw_v135_pair3_win"
    ] = (
        old.get(
            "accumulation",
            {},
        )
        .get(
            "3",
            {},
        )
        .get(
            "v135_paired_win"
        )
    )


p = (
    BASELINE_ROOT
    / "cv_role_calibration_v1.json"
)

if p.exists():

    old = json.loads(
        p.read_text()
    )

    dual = old.get(
        "v135_plus_v134",
        {},
    )

    baseline[
        "v135_v134_calibrated_pair_win"
    ] = dual.get(
        "macro_pair_win"
    )

    baseline[
        "v135_v134_calibrated_global_auc"
    ] = dual.get(
        "global_auc"
    )


report[
    "baselines"
] = baseline


(
    ROOT
    / "oof_report.json"
).write_text(
    json.dumps(
        report,
        indent=2,
    )
)


print("=" * 100)
print(
    "ROLE DIRECT SENSOR V1 — 5-FOLD OOF"
)
print("=" * 100)

print(
    "OOF rows                 =",
    len(rows),
)

print(
    "matched query pairs      =",
    len(by_probe),
)

print(
    "matched system pairs     =",
    len(by_pair),
)

print()
print(
    "GLOBAL AUC               =",
    round(
        global_auc,
        4,
    ),
)

print(
    "MATCHED PROBE WIN        =",
    round(
        mean(
            probe_wins
        ),
        4,
    ),
)

print(
    "MACRO ROLE AUC           =",
    round(
        macro_role_auc,
        4,
    ),
)

print(
    "MACRO ROLE PAIR3 WIN     =",
    round(
        macro_role_pair,
        4,
    ),
)

print()
print("=" * 100)
print("ACCUMULATION")
print("=" * 100)

for k, x in accumulation.items():

    print(
        f"k={k} "
        f"win={x['paired_win']:.4f} "
        f"CI=[{x['ci95'][0]:.4f}, "
        f"{x['ci95'][1]:.4f}] "
        f"delta={x['mean_logit_delta']:+.4f}"
    )


print()
print("=" * 100)
print("BASELINES")
print("=" * 100)

for k, v in baseline.items():
    print(
        f"{k:38s} {v}"
    )


print()
print("=" * 100)
print("PER ROLE")
print("=" * 100)

for role, x in per_role.items():

    print(
        f"{role:26s} "
        f"AUC={x['oof_auc']:.3f} "
        f"pair3={x['pair3_win']:.3f}"
    )


print()
print(
    "REPORT =",
    ROOT
    / "oof_report.json"
)
