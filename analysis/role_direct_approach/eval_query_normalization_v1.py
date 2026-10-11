#!/usr/bin/env python3

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

from sklearn.metrics import roc_auc_score


ROOT = Path(
    "artifacts/role_direct_sealed_v1"
)

RAW = ROOT / "final_sensor_scores.jsonl"
BASE = ROOT / "query_baseline_scores.jsonl"
OUT = ROOT / "query_normalization_report.json"


def read_jsonl(path):
    return [
        json.loads(x)
        for x in Path(path).read_text().splitlines()
        if x.strip()
    ]


def balanced_accuracy(labels, preds):

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


def choose_threshold(scores, labels):

    vals = sorted(
        set(map(float, scores))
    )

    span = max(
        1.0,
        max(vals) - min(vals),
    )

    eps = span * 1e-6

    candidates = [
        vals[0] - eps
    ]

    candidates += [
        0.5 * (a + b)
        for a, b in zip(
            vals[:-1],
            vals[1:],
        )
    ]

    candidates.append(
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

    middle = 0.5 * (
        median(pos)
        + median(neg)
    )

    ranked = []

    for t in candidates:

        preds = [
            int(s >= t)
            for s in scores
        ]

        ranked.append(
            (
                balanced_accuracy(
                    labels,
                    preds,
                ),
                -abs(t - middle),
                t,
            )
        )

    ranked.sort(
        reverse=True
    )

    return float(
        ranked[0][2]
    )


def aggregate(rows, k, field):

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
                int(x["probe_rank"]),
        )

        assert len(rr) == 3

        cur = rr[:k]

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
                        float(x[field])
                        for x in cur
                    )
                ),
        })

    return out


def metrics(rows):

    labels = [
        int(x["label"])
        for x in rows
    ]

    preds = [
        int(x["pred"])
        for x in rows
    ]

    tp = sum(
        y == 1 and p == 1
        for y, p in zip(
            labels,
            preds,
        )
    )

    tn = sum(
        y == 0 and p == 0
        for y, p in zip(
            labels,
            preds,
        )
    )

    fp = sum(
        y == 0 and p == 1
        for y, p in zip(
            labels,
            preds,
        )
    )

    fn = sum(
        y == 1 and p == 0
        for y, p in zip(
            labels,
            preds,
        )
    )

    tpr = tp / (tp + fn)
    tnr = tn / (tn + fp)
    fpr = fp / (fp + tn)

    acc = (
        tp + tn
    ) / len(rows)

    return {
        "accuracy":
            float(acc),

        "balanced_accuracy":
            float(
                0.5
                * (
                    tpr + tnr
                )
            ),

        "tpr":
            float(tpr),

        "fpr":
            float(fpr),

        "tp":
            int(tp),

        "tn":
            int(tn),

        "fp":
            int(fp),

        "fn":
            int(fn),
    }


def exact_pair(rows):

    g = defaultdict(list)

    for x in rows:
        g[
            (
                x["role"],
                x["pair_id"],
            )
        ].append(x)

    assert len(g) == 100

    return float(
        mean(
            int(
                all(
                    z["pred"]
                    == z["label"]
                    for z in rr
                )
            )
            for rr in g.values()
        )
    )


def pair_ranking(rows):

    g = defaultdict(dict)

    for x in rows:
        g[
            (
                x["role"],
                x["pair_id"],
            )
        ][x["label"]] = x["score"]

    assert len(g) == 100

    wins = []

    for d in g.values():

        if d[1] > d[0]:
            wins.append(1.0)

        elif d[1] < d[0]:
            wins.append(0.0)

        else:
            wins.append(0.5)

    return float(mean(wins))


raw = read_jsonl(RAW)
base = read_jsonl(BASE)

assert len(raw) == 1200
assert len(base) == 120


bmap = {}

for x in base:

    key = (
        str(x["target_role"]),
        str(x["panel"]),
        int(x["probe_rank"]),
        str(x["query"]),
    )

    assert key not in bmap

    bmap[key] = float(
        x["query_baseline_logit"]
    )


for x in raw:

    key = (
        str(x["target_role"]),
        str(x["panel"]),
        int(x["probe_rank"]),
        str(x["query"]),
    )

    assert key in bmap

    x["query_baseline_logit"] = (
        bmap[key]
    )

    x["normalized_logit"] = (
        float(x["logit"])
        - bmap[key]
    )


cal = [
    x
    for x in raw
    if x["panel"] == "calibration"
]

test = [
    x
    for x in raw
    if x["panel"] == "sealed_test"
]

assert len(cal) == 600
assert len(test) == 600


report = {
    "status":
        "POSTHOC DEVELOPMENT DIAGNOSTIC",

    "normalization":
        (
            "sensor(role,query,response) "
            "- sensor(role,query,empty_response)"
        ),

    "results": {},
}


for field in [
    "logit",
    "normalized_logit",
]:

    name = (
        "raw"
        if field == "logit"
        else "query_normalized"
    )

    report["results"][name] = {}

    for k in [1, 2, 3]:

        tr = aggregate(
            cal,
            k,
            field,
        )

        te = aggregate(
            test,
            k,
            field,
        )

        assert len(tr) == 200
        assert len(te) == 200

        # ----------------------------------------------
        # Global CAL-only threshold
        # ----------------------------------------------

        gt = choose_threshold(
            [
                x["score"]
                for x in tr
            ],
            [
                x["label"]
                for x in tr
            ],
        )

        global_rows = [
            {
                **x,
                "pred":
                    int(
                        x["score"] >= gt
                    ),
            }
            for x in te
        ]

        gm = metrics(
            global_rows
        )

        gm[
            "exact_pair"
        ] = exact_pair(
            global_rows
        )

        gm[
            "matched_ranking"
        ] = pair_ranking(
            global_rows
        )

        gm[
            "threshold"
        ] = gt

        # ----------------------------------------------
        # Role-specific CAL-only thresholds
        # ----------------------------------------------

        role_thresholds = {}

        roles = sorted({
            x["role"]
            for x in tr
        })

        assert len(roles) == 20

        for role in roles:

            rr = [
                x
                for x in tr
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

        role_rows = [
            {
                **x,
                "pred":
                    int(
                        x["score"]
                        >= role_thresholds[
                            x["role"]
                        ]
                    ),
            }
            for x in te
        ]

        rm = metrics(
            role_rows
        )

        rm[
            "exact_pair"
        ] = exact_pair(
            role_rows
        )

        rm[
            "matched_ranking"
        ] = pair_ranking(
            role_rows
        )

        # ----------------------------------------------
        # Per-role absolute results
        # ----------------------------------------------

        per_role = {}

        for role in roles:

            rr = [
                x
                for x in role_rows
                if x["role"] == role
            ]

            assert len(rr) == 10

            per_role[role] = metrics(
                rr
            )

        report[
            "results"
        ][name][str(k)] = {
            "global":
                gm,

            "role_specific":
                rm,

            "per_role":
                per_role,
        }


# Ranking MUST be invariant under subtracting same-query baseline.
for k in ["1", "2", "3"]:

    a = report[
        "results"
    ]["raw"][k][
        "global"
    ][
        "matched_ranking"
    ]

    b = report[
        "results"
    ][
        "query_normalized"
    ][k][
        "global"
    ][
        "matched_ranking"
    ]

    assert abs(a - b) < 1e-12, (
        k,
        a,
        b,
    )


# Row-level AUC diagnostic.
report[
    "row_auc"
] = {
    "cal_raw":
        float(
            roc_auc_score(
                [
                    x["label"]
                    for x in cal
                ],
                [
                    x["logit"]
                    for x in cal
                ],
            )
        ),

    "cal_normalized":
        float(
            roc_auc_score(
                [
                    x["label"]
                    for x in cal
                ],
                [
                    x["normalized_logit"]
                    for x in cal
                ],
            )
        ),

    "sealed_raw":
        float(
            roc_auc_score(
                [
                    x["label"]
                    for x in test
                ],
                [
                    x["logit"]
                    for x in test
                ],
            )
        ),

    "sealed_normalized":
        float(
            roc_auc_score(
                [
                    x["label"]
                    for x in test
                ],
                [
                    x["normalized_logit"]
                    for x in test
                ],
            )
        ),
}


OUT.write_text(
    json.dumps(
        report,
        indent=2,
    )
)


print("=" * 105)
print("QUERY-BASELINE NORMALIZATION — POSTHOC DIAGNOSTIC")
print("=" * 105)

print()
print("ROW AUC")
for k, v in report["row_auc"].items():
    print(
        f"{k:20s} = {v:.4f}"
    )


print()
print("=" * 105)
print("SEALED ABSOLUTE DETECTION")
print("=" * 105)

for k in ["1", "2", "3"]:

    raw_r = report[
        "results"
    ]["raw"][k][
        "role_specific"
    ]

    nor_r = report[
        "results"
    ]["query_normalized"][k][
        "role_specific"
    ]

    raw_g = report[
        "results"
    ]["raw"][k][
        "global"
    ]

    nor_g = report[
        "results"
    ]["query_normalized"][k][
        "global"
    ]

    print()
    print(f"K={k}")

    print(
        "  RAW GLOBAL       "
        f"BA={raw_g['balanced_accuracy']:.3f} "
        f"FPR={raw_g['fpr']:.3f} "
        f"EXACT={raw_g['exact_pair']:.3f}"
    )

    print(
        "  NORM GLOBAL      "
        f"BA={nor_g['balanced_accuracy']:.3f} "
        f"FPR={nor_g['fpr']:.3f} "
        f"EXACT={nor_g['exact_pair']:.3f}"
    )

    print(
        "  RAW ROLE         "
        f"BA={raw_r['balanced_accuracy']:.3f} "
        f"FPR={raw_r['fpr']:.3f} "
        f"EXACT={raw_r['exact_pair']:.3f}"
    )

    print(
        "  NORM ROLE        "
        f"BA={nor_r['balanced_accuracy']:.3f} "
        f"FPR={nor_r['fpr']:.3f} "
        f"EXACT={nor_r['exact_pair']:.3f}"
    )

    print(
        "  MATCHED RANKING  "
        f"{nor_r['matched_ranking']:.3f}"
    )


print()
print("=" * 105)
print("PER ROLE — K=3 NORMALIZED ROLE THRESHOLD")
print("=" * 105)

for role, x in (
    report[
        "results"
    ][
        "query_normalized"
    ][
        "3"
    ][
        "per_role"
    ].items()
):

    print(
        f"{role:26s} "
        f"ACC={x['accuracy']:.3f} "
        f"TPR={x['tpr']:.3f} "
        f"FPR={x['fpr']:.3f}"
    )


print()
print(
    "REPORT =",
    OUT,
)
