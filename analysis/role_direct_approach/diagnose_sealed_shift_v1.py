#!/usr/bin/env python3

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev

from sklearn.metrics import roc_auc_score


ROOT = Path(
    "artifacts/role_direct_sealed_v1"
)

rows = [
    json.loads(x)
    for x in (
        ROOT / "final_sensor_scores.jsonl"
    ).read_text().splitlines()
    if x.strip()
]


def win(a, b):
    if a > b:
        return 1.0
    if a < b:
        return 0.0
    return 0.5


def stats(v):
    v = list(map(float, v))
    return (
        mean(v),
        pstdev(v),
        min(v),
        max(v),
    )


print("=" * 110)
print("CALIBRATION vs SEALED SCORE DISTRIBUTION")
print("=" * 110)

for panel in [
    "calibration",
    "sealed_test",
]:

    rr = [
        x for x in rows
        if x["panel"] == panel
    ]

    pos = [
        x["logit"]
        for x in rr
        if int(x["label"]) == 1
    ]

    neg = [
        x["logit"]
        for x in rr
        if int(x["label"]) == 0
    ]

    pm, ps, pmin, pmax = stats(pos)
    nm, ns, nmin, nmax = stats(neg)

    auc = roc_auc_score(
        [x["label"] for x in rr],
        [x["logit"] for x in rr],
    )

    print()
    print(panel.upper())
    print(
        f"POS mean={pm:+.3f} std={ps:.3f} "
        f"range=[{pmin:+.3f},{pmax:+.3f}]"
    )
    print(
        f"NEG mean={nm:+.3f} std={ns:.3f} "
        f"range=[{nmin:+.3f},{nmax:+.3f}]"
    )
    print(
        f"mean separation={pm-nm:+.3f}"
    )
    print(
        f"global row AUC={auc:.4f}"
    )


print()
print("=" * 110)
print("EACH PROBE / STRATEGY")
print("=" * 110)

for panel in [
    "calibration",
    "sealed_test",
]:

    print()
    print("PANEL:", panel)

    for rank in [1, 2, 3]:

        rr = [
            x for x in rows
            if x["panel"] == panel
            and int(x["probe_rank"]) == rank
        ]

        strategies = sorted({
            x["probe_strategy"]
            for x in rr
        })

        assert len(strategies) == 1

        strategy = strategies[0]

        g = defaultdict(dict)

        for x in rr:
            g[
                (
                    x["target_role"],
                    x["pair_id"],
                )
            ][int(x["label"])] = float(
                x["logit"]
            )

        assert len(g) == 100

        wins = [
            win(d[1], d[0])
            for d in g.values()
        ]

        margins = [
            d[1] - d[0]
            for d in g.values()
        ]

        auc = roc_auc_score(
            [x["label"] for x in rr],
            [x["logit"] for x in rr],
        )

        print(
            f"rank={rank} "
            f"{strategy:22s} "
            f"matched_win={mean(wins):.3f} "
            f"margin={mean(margins):+.3f} "
            f"row_auc={auc:.3f}"
        )


print()
print("=" * 110)
print("SEALED PER ROLE — MATCHED RANKING K=3")
print("=" * 110)

sealed = [
    x for x in rows
    if x["panel"] == "sealed_test"
]

g = defaultdict(
    lambda: defaultdict(list)
)

for x in sealed:
    g[x["target_role"]][
        (
            x["pair_id"],
            int(x["label"]),
        )
    ].append(
        (
            int(x["probe_rank"]),
            float(x["logit"]),
        )
    )


for role in sorted(g):

    by_pair = defaultdict(dict)

    for (
        pair_id,
        label,
    ), vv in g[role].items():

        vv = sorted(vv)

        assert len(vv) == 3

        by_pair[pair_id][label] = mean(
            z for _, z in vv
        )

    assert len(by_pair) == 5

    wins = [
        win(d[1], d[0])
        for d in by_pair.values()
    ]

    margins = [
        d[1] - d[0]
        for d in by_pair.values()
    ]

    print(
        f"{role:26s} "
        f"rank={mean(wins):.3f} "
        f"margin={mean(margins):+.3f}"
    )
