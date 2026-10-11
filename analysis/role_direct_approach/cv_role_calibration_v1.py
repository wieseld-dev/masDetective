#!/usr/bin/env python3

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score


ROOT = Path(
    "artifacts/role_direct_matched_cf_v1"
)

rows = [
    json.loads(x)
    for x in (
        ROOT
        / "matched_probe_comparisons.jsonl"
    ).read_text().splitlines()
    if x.strip()
]


# ------------------------------------------------------------
# Convert 3 probes -> one system-level feature vector.
#
# For each role/pair:
#
# POS = target installed
# NEG = target absent
#
# features:
#   mean V13.5 across 3 direct probes
#   mean V13.4 across 3 direct probes
# ------------------------------------------------------------

groups = defaultdict(list)

for x in rows:
    groups[
        (
            x["target_role"],
            x["pair_id"],
        )
    ].append(x)


samples = defaultdict(list)

for (role, pair_id), rr in groups.items():

    rr = sorted(
        rr,
        key=lambda x:
            x["probe_rank"]
    )

    assert len(rr) == 3

    pos135 = np.mean([
        x["v135_pos"]
        for x in rr
    ])

    neg135 = np.mean([
        x["v135_neg"]
        for x in rr
    ])

    pos134 = np.mean([
        x["v134_pos"]
        for x in rr
    ])

    neg134 = np.mean([
        x["v134_neg"]
        for x in rr
    ])

    samples[role].append({
        "pair_id": pair_id,
        "y": 1,
        "v135": float(pos135),
        "v134": float(pos134),
    })

    samples[role].append({
        "pair_id": pair_id,
        "y": 0,
        "v135": float(neg135),
        "v134": float(neg134),
    })


def evaluate(features):

    all_y = []
    all_p = []

    per_role = {}

    pair_wins = []

    for role in sorted(samples):

        rr = samples[role]

        pair_ids = sorted({
            x["pair_id"]
            for x in rr
        })

        assert len(pair_ids) == 5

        role_y = []
        role_p = []

        role_pair_wins = []

        for heldout in pair_ids:

            train = [
                x for x in rr
                if x["pair_id"] != heldout
            ]

            test = [
                x for x in rr
                if x["pair_id"] == heldout
            ]

            assert len(train) == 8
            assert len(test) == 2

            Xtr = np.array([
                [x[f] for f in features]
                for x in train
            ])

            ytr = np.array([
                x["y"]
                for x in train
            ])

            Xte = np.array([
                [x[f] for f in features]
                for x in test
            ])

            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    C=1.0,
                    class_weight="balanced",
                    solver="liblinear",
                    random_state=0,
                ),
            )

            model.fit(
                Xtr,
                ytr,
            )

            probs = model.predict_proba(
                Xte
            )[:, 1]

            for x, p in zip(
                test,
                probs,
            ):
                role_y.append(
                    x["y"]
                )

                role_p.append(
                    float(p)
                )

                all_y.append(
                    x["y"]
                )

                all_p.append(
                    float(p)
                )

            pos = [
                float(p)
                for x, p in zip(
                    test,
                    probs,
                )
                if x["y"] == 1
            ][0]

            neg = [
                float(p)
                for x, p in zip(
                    test,
                    probs,
                )
                if x["y"] == 0
            ][0]

            w = (
                1.0
                if pos > neg
                else 0.5
                if pos == neg
                else 0.0
            )

            role_pair_wins.append(
                w
            )

            pair_wins.append(
                w
            )

        per_role[role] = {
            "auc":
                float(
                    roc_auc_score(
                        role_y,
                        role_p,
                    )
                ),

            "pair_win":
                float(
                    np.mean(
                        role_pair_wins
                    )
                ),
        }

    macro_auc = float(
        np.mean([
            x["auc"]
            for x in per_role.values()
        ])
    )

    macro_pair = float(
        np.mean([
            x["pair_win"]
            for x in per_role.values()
        ])
    )

    global_auc = float(
        roc_auc_score(
            all_y,
            all_p,
        )
    )

    return {
        "macro_role_auc":
            macro_auc,

        "macro_pair_win":
            macro_pair,

        "global_auc":
            global_auc,

        "per_role":
            per_role,
    }


r135 = evaluate([
    "v135",
])

rdual = evaluate([
    "v135",
    "v134",
])


print("=" * 95)
print("ROLE DIRECT — LEAVE-ONE-PAIR-OUT CALIBRATION")
print("=" * 95)

print()
print("V13.5 ONLY")
print(
    "macro role AUC =",
    round(
        r135["macro_role_auc"],
        4,
    )
)

print(
    "macro pair win =",
    round(
        r135["macro_pair_win"],
        4,
    )
)

print(
    "global AUC     =",
    round(
        r135["global_auc"],
        4,
    )
)

print()
print("V13.5 + V13.4")
print(
    "macro role AUC =",
    round(
        rdual["macro_role_auc"],
        4,
    )
)

print(
    "macro pair win =",
    round(
        rdual["macro_pair_win"],
        4,
    )
)

print(
    "global AUC     =",
    round(
        rdual["global_auc"],
        4,
    )
)

print()
print("=" * 95)
print("PER ROLE — V13.5 CALIBRATED")
print("=" * 95)

for role, x in sorted(
    r135["per_role"].items()
):
    print(
        f"{role:26s} "
        f"AUC={x['auc']:.3f} "
        f"pair={x['pair_win']:.3f}"
    )

out = {
    "v135_only":
        r135,

    "v135_plus_v134":
        rdual,
}

(
    ROOT
    / "cv_role_calibration_v1.json"
).write_text(
    json.dumps(
        out,
        indent=2,
    )
)

print()
print(
    "REPORT =",
    ROOT
    / "cv_role_calibration_v1.json"
)
