#!/usr/bin/env python3

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

import numpy as np


ROOT = Path(
    "artifacts/role_direct_sensor_v1"
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


def balanced_accuracy(
    labels,
    preds,
):
    pos = [
        p
        for y, p in zip(
            labels,
            preds,
        )
        if y == 1
    ]

    neg = [
        p
        for y, p in zip(
            labels,
            preds,
        )
        if y == 0
    ]

    tpr = (
        sum(pos)
        / len(pos)
    )

    tnr = (
        sum(
            1 - x
            for x in neg
        )
        / len(neg)
    )

    return (
        0.5
        * (
            tpr
            + tnr
        )
    )


def choose_threshold(
    scores,
    labels,
):
    """
    TRAIN-ONLY threshold selection.

    Direction is FIXED:
        higher sensor score => TARGET PRESENT

    No role-specific sign flipping is allowed.
    """

    scores = list(
        map(
            float,
            scores,
        )
    )

    labels = list(
        map(
            int,
            labels,
        )
    )

    vals = sorted(
        set(scores)
    )

    assert vals

    span = max(
        1.0,
        max(vals)
        - min(vals),
    )

    eps = (
        span
        * 1e-6
    )

    candidates = [
        vals[0] - eps
    ]

    for a, b in zip(
        vals[:-1],
        vals[1:],
    ):
        candidates.append(
            0.5
            * (
                a + b
            )
        )

    candidates.append(
        vals[-1] + eps
    )

    pos_scores = [
        s
        for s, y in zip(
            scores,
            labels,
        )
        if y == 1
    ]

    neg_scores = [
        s
        for s, y in zip(
            scores,
            labels,
        )
        if y == 0
    ]

    target_mid = (
        0.5
        * (
            median(pos_scores)
            + median(neg_scores)
        )
    )

    ranked = []

    for t in candidates:

        preds = [
            int(
                s >= t
            )
            for s in scores
        ]

        ba = balanced_accuracy(
            labels,
            preds,
        )

        ranked.append(
            (
                ba,
                -abs(
                    t
                    - target_mid
                ),
                t,
            )
        )

    ranked.sort(
        reverse=True
    )

    best_ba, _, best_t = (
        ranked[0]
    )

    return (
        float(best_t),
        float(best_ba),
    )


def aggregate(
    rows,
    k,
):
    """
    Convert probe rows into one absolute
    score per MAS condition.

    k=1/2/3 means average the first k
    fixed direct probes.
    """

    g = defaultdict(list)

    for x in rows:

        key = (
            str(
                x["role"]
            ),
            str(
                x["pair_id"]
            ),
            int(
                x["label"]
            ),
        )

        g[key].append(
            x
        )

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

        cur = rr[:k]

        out.append({
            "role":
                role,

            "pair_id":
                pair_id,

            "label":
                int(label),

            "score":
                float(
                    mean(
                        float(
                            x["logit"]
                        )
                        for x in cur
                    )
                ),
        })

    return out


def classification_metrics(
    rows,
):
    labels = [
        int(
            x["label"]
        )
        for x in rows
    ]

    preds = [
        int(
            x["pred"]
        )
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

    tpr = (
        tp
        / (
            tp + fn
        )
    )

    tnr = (
        tn
        / (
            tn + fp
        )
    )

    fpr = (
        fp
        / (
            fp + tn
        )
    )

    acc = (
        tp + tn
    ) / len(rows)

    ba = (
        0.5
        * (
            tpr
            + tnr
        )
    )

    return {
        "n":
            len(rows),

        "accuracy":
            float(acc),

        "balanced_accuracy":
            float(ba),

        "tpr":
            float(tpr),

        "tnr":
            float(tnr),

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


all_results = {}


for k in [
    1,
    2,
    3,
]:

    global_test = []
    role_test = []

    threshold_dump = []

    for fold in range(
        1,
        6,
    ):

        fold_root = (
            ROOT
            / f"fold_{fold}"
        )

        train_rows = read_jsonl(
            fold_root
            / "calibration_train_predictions.jsonl"
        )

        test_rows = read_jsonl(
            fold_root
            / "test_predictions.jsonl"
        )

        assert len(train_rows) == 480
        assert len(test_rows) == 120

        train = aggregate(
            train_rows,
            k,
        )

        test = aggregate(
            test_rows,
            k,
        )

        # 20 roles * 4 train pairs * 2 labels.
        assert len(train) == 160

        # 20 roles * 1 heldout pair * 2 labels.
        assert len(test) == 40

        # ====================================================
        # GLOBAL threshold:
        # one threshold for all roles in this fold.
        # ====================================================

        gt, gba = choose_threshold(
            [
                x["score"]
                for x in train
            ],
            [
                x["label"]
                for x in train
            ],
        )

        for x in test:

            global_test.append({
                **x,
                "fold":
                    fold,

                "threshold":
                    gt,

                "pred":
                    int(
                        x["score"]
                        >= gt
                    ),
            })

        # ====================================================
        # ROLE-SPECIFIC threshold:
        # 4 POS + 4 NEG training systems per role.
        # ====================================================

        roles = sorted({
            x["role"]
            for x in train
        })

        assert len(roles) == 20

        for role in roles:

            trr = [
                x
                for x in train
                if x["role"]
                == role
            ]

            ter = [
                x
                for x in test
                if x["role"]
                == role
            ]

            assert len(trr) == 8
            assert len(ter) == 2

            rt, rba = choose_threshold(
                [
                    x["score"]
                    for x in trr
                ],
                [
                    x["label"]
                    for x in trr
                ],
            )

            threshold_dump.append({
                "fold":
                    fold,

                "k":
                    k,

                "role":
                    role,

                "global_threshold":
                    gt,

                "global_train_ba":
                    gba,

                "role_threshold":
                    rt,

                "role_train_ba":
                    rba,
            })

            for x in ter:

                role_test.append({
                    **x,
                    "fold":
                        fold,

                    "threshold":
                        rt,

                    "pred":
                        int(
                            x["score"]
                            >= rt
                        ),
                })

    assert len(global_test) == 200
    assert len(role_test) == 200

    gm = classification_metrics(
        global_test
    )

    rm = classification_metrics(
        role_test
    )

    # --------------------------------------------------------
    # Exact pair correctness:
    # both PRESENT and ABSENT must be classified correctly.
    # --------------------------------------------------------

    def exact_pair_rate(rows):

        g = defaultdict(list)

        for x in rows:

            g[
                (
                    x["role"],
                    x["pair_id"],
                )
            ].append(x)

        assert len(g) == 100

        vals = []

        for key, rr in g.items():

            assert len(rr) == 2

            vals.append(
                int(
                    all(
                        x["pred"]
                        == x["label"]
                        for x in rr
                    )
                )
            )

        return float(
            mean(vals)
        )

    gm[
        "exact_matched_pair"
    ] = exact_pair_rate(
        global_test
    )

    rm[
        "exact_matched_pair"
    ] = exact_pair_rate(
        role_test
    )

    # --------------------------------------------------------
    # Per-role held-out classification accuracy.
    # 5 folds * 2 conditions = 10 decisions / role.
    # --------------------------------------------------------

    per_role = {}

    for role in sorted({
        x["role"]
        for x in role_test
    }):

        rr = [
            x
            for x in role_test
            if x["role"]
            == role
        ]

        assert len(rr) == 10

        per_role[
            role
        ] = classification_metrics(
            rr
        )

    all_results[
        str(k)
    ] = {
        "global_threshold":
            gm,

        "role_specific_threshold":
            rm,

        "per_role":
            per_role,

        "thresholds":
            threshold_dump,
    }


report = {
    "experiment":
        "ROLE DIRECT SENSOR V1 — LEAKAGE-SAFE ABSOLUTE DETECTION",

    "protocol":
        (
            "For each fold, threshold selection uses only "
            "the four training pair-indices. "
            "Held-out pair-index is used only once for evaluation."
        ),

    "direction":
        (
            "fixed: larger sensor logit means target present; "
            "no test-time or per-role sign flipping"
        ),

    "results":
        all_results,
}


(
    ROOT
    / "absolute_detection_report.json"
).write_text(
    json.dumps(
        report,
        indent=2,
    ),
    encoding="utf-8",
)


print("=" * 105)
print(
    "ROLE DIRECT SENSOR V1 — ABSOLUTE DETECTION"
)
print("=" * 105)

for k in [
    1,
    2,
    3,
]:

    x = all_results[
        str(k)
    ]

    g = x[
        "global_threshold"
    ]

    r = x[
        "role_specific_threshold"
    ]

    print()
    print(
        f"K={k} DIRECT PROBE(S)"
    )

    print(
        "  GLOBAL threshold:"
        f" BA={g['balanced_accuracy']:.3f}"
        f" ACC={g['accuracy']:.3f}"
        f" TPR={g['tpr']:.3f}"
        f" FPR={g['fpr']:.3f}"
        f" PAIR_EXACT={g['exact_matched_pair']:.3f}"
    )

    print(
        "  ROLE threshold:  "
        f" BA={r['balanced_accuracy']:.3f}"
        f" ACC={r['accuracy']:.3f}"
        f" TPR={r['tpr']:.3f}"
        f" FPR={r['fpr']:.3f}"
        f" PAIR_EXACT={r['exact_matched_pair']:.3f}"
    )


print()
print("=" * 105)
print(
    "PER ROLE — K=2 ROLE-SPECIFIC THRESHOLD"
)
print("=" * 105)

for role, x in (
    all_results[
        "2"
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
    ROOT
    / "absolute_detection_report.json"
)
