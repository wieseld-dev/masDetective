#!/usr/bin/env python3

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, median


ROOT = Path(
    "artifacts/role_direct_sealed_v1"
)

SCORES = (
    ROOT / "final_sensor_scores.jsonl"
)

OUT = (
    ROOT / "frozen_calibration.json"
)


def read_jsonl(path):
    return [
        json.loads(x)
        for x in Path(path).read_text().splitlines()
        if x.strip()
    ]


def ba(labels, preds):

    pos = [
        p
        for y, p in zip(labels, preds)
        if y == 1
    ]

    neg = [
        p
        for y, p in zip(labels, preds)
        if y == 0
    ]

    tpr = sum(pos) / len(pos)

    tnr = sum(
        1 - p
        for p in neg
    ) / len(neg)

    return 0.5 * (
        tpr + tnr
    )


def choose_threshold(
    scores,
    labels,
):

    vals = sorted(
        set(
            map(float, scores)
        )
    )

    span = max(
        1.0,
        max(vals) - min(vals),
    )

    eps = span * 1e-6

    cand = [
        vals[0] - eps
    ]

    cand += [
        0.5 * (a + b)
        for a, b in zip(
            vals[:-1],
            vals[1:],
        )
    ]

    cand.append(
        vals[-1] + eps
    )

    pos = [
        s
        for s, y in zip(
            scores,
            labels,
        )
        if y == 1
    ]

    neg = [
        s
        for s, y in zip(
            scores,
            labels,
        )
        if y == 0
    ]

    mid = 0.5 * (
        median(pos)
        + median(neg)
    )

    ranked = []

    for t in cand:

        preds = [
            int(s >= t)
            for s in scores
        ]

        ranked.append(
            (
                ba(labels, preds),
                -abs(t - mid),
                t,
            )
        )

    ranked.sort(
        reverse=True
    )

    return float(
        ranked[0][2]
    )


def aggregate(rows, k):

    g = defaultdict(list)

    for x in rows:

        g[
            (
                x["target_role"],
                x["pair_id"],
                int(x["label"]),
            )
        ].append(x)

    out = []

    for (
        role,
        pair_id,
        label,
    ), rr in g.items():

        rr = sorted(
            rr,
            key=lambda x:
                int(
                    x["probe_rank"]
                ),
        )

        assert len(rr) == 3

        out.append({
            "role":
                role,

            "pair_id":
                pair_id,

            "label":
                label,

            "score":
                float(
                    mean(
                        float(x["logit"])
                        for x in rr[:k]
                    )
                ),
        })

    return out


rows = read_jsonl(SCORES)

cal = [
    x
    for x in rows
    if x["panel"]
    == "calibration"
]

assert len(cal) == 600

assert len({
    (
        x["target_role"],
        x["pair_id"],
    )
    for x in cal
}) == 100


result = {
    "version":
        "role_direct_sealed_calibration_v1",

    "source":
        str(SCORES),

    "calibration_rows":
        600,

    "calibration_pairs":
        100,

    "sealed_test_metrics_seen":
        False,

    "primary_endpoint": {
        "k":
            3,

        "threshold_scheme":
            "role_specific",

        "metric":
            "balanced_accuracy",
    },

    "k": {},
}


for k in [1, 2, 3]:

    aa = aggregate(
        cal,
        k,
    )

    assert len(aa) == 200

    global_threshold = (
        choose_threshold(
            [
                x["score"]
                for x in aa
            ],
            [
                x["label"]
                for x in aa
            ],
        )
    )

    roles = sorted({
        x["role"]
        for x in aa
    })

    assert len(roles) == 20

    role_thresholds = {}

    for role in roles:

        rr = [
            x
            for x in aa
            if x["role"] == role
        ]

        assert len(rr) == 10

        role_thresholds[
            role
        ] = choose_threshold(
            [
                x["score"]
                for x in rr
            ],
            [
                x["label"]
                for x in rr
            ],
        )

    result["k"][str(k)] = {
        "global_threshold":
            global_threshold,

        "role_thresholds":
            role_thresholds,
    }


if OUT.exists():
    raise RuntimeError(
        f"Refusing to overwrite frozen calibration: {OUT}"
    )

OUT.write_text(
    json.dumps(
        result,
        indent=2,
    )
)

print("=" * 95)
print("CALIBRATION FROZEN — SEALED TEST NOT OPENED")
print("=" * 95)
print("calibration rows  = 600")
print("calibration pairs = 100")
print("primary endpoint  = K=3 ROLE-SPECIFIC BA")
print("output            =", OUT)
print()
print("CALIBRATION_FREEZE_PASS")
