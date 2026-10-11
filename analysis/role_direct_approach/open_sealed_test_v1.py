#!/usr/bin/env python3

import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean

ROOT = Path(
    "artifacts/role_direct_sealed_v1"
)

SCORES = ROOT / "final_sensor_scores.jsonl"
CALIBRATION = ROOT / "frozen_calibration.json"
OUT = ROOT / "sealed_test_report.json"


def read_jsonl(path):
    return [
        json.loads(x)
        for x in Path(path).read_text().splitlines()
        if x.strip()
    ]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def win(a, b):
    if a > b:
        return 1.0
    if a < b:
        return 0.0
    return 0.5


def aggregate(rows, k):
    g = defaultdict(list)

    for x in rows:
        g[
            (
                str(x["target_role"]),
                str(x["pair_id"]),
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
            key=lambda x: int(x["probe_rank"]),
        )

        assert len(rr) == 3

        cur = rr[:k]

        out.append({
            "role": role,
            "pair_id": pair_id,
            "label": label,
            "score": float(
                mean(
                    float(x["logit"])
                    for x in cur
                )
            ),
        })

    return out


def classification_metrics(rows):
    y = [int(x["label"]) for x in rows]
    p = [int(x["pred"]) for x in rows]

    tp = sum(
        yy == 1 and pp == 1
        for yy, pp in zip(y, p)
    )

    tn = sum(
        yy == 0 and pp == 0
        for yy, pp in zip(y, p)
    )

    fp = sum(
        yy == 0 and pp == 1
        for yy, pp in zip(y, p)
    )

    fn = sum(
        yy == 1 and pp == 0
        for yy, pp in zip(y, p)
    )

    tpr = tp / (tp + fn)
    tnr = tn / (tn + fp)
    fpr = fp / (fp + tn)

    acc = (tp + tn) / len(rows)
    ba = 0.5 * (tpr + tnr)

    return {
        "n": len(rows),
        "accuracy": float(acc),
        "balanced_accuracy": float(ba),
        "tpr": float(tpr),
        "tnr": float(tnr),
        "fpr": float(fpr),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }


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

    for rr in g.values():
        assert len(rr) == 2

        vals.append(
            int(
                all(
                    x["pred"] == x["label"]
                    for x in rr
                )
            )
        )

    return float(mean(vals))


def matched_pair_ranking(rows):
    g = defaultdict(dict)

    for x in rows:
        g[
            (
                x["role"],
                x["pair_id"],
            )
        ][
            int(x["label"])
        ] = float(x["score"])

    assert len(g) == 100

    wins = []

    for d in g.values():
        assert set(d) == {0, 1}
        wins.append(
            win(
                d[1],
                d[0],
            )
        )

    return float(mean(wins)), wins


def bootstrap_primary(rows, n=10000):
    """
    Bootstrap over the 100 matched system pairs.
    Each resampled unit preserves POS and NEG together.
    """

    g = defaultdict(list)

    for x in rows:
        g[
            (
                x["role"],
                x["pair_id"],
            )
        ].append(x)

    keys = sorted(g)

    assert len(keys) == 100

    rng = random.Random(20261009)

    vals = []

    for _ in range(n):
        sampled = [
            keys[
                rng.randrange(len(keys))
            ]
            for _ in keys
        ]

        rr = []

        for key in sampled:
            rr.extend(g[key])

        vals.append(
            classification_metrics(rr)[
                "balanced_accuracy"
            ]
        )

    vals.sort()

    return [
        float(vals[int(0.025 * len(vals))]),
        float(vals[int(0.975 * len(vals))]),
    ]


if OUT.exists():
    raise RuntimeError(
        f"SEALED TEST ALREADY OPENED: {OUT}"
    )


scores = read_jsonl(SCORES)

assert len(scores) == 1200


# CRITICAL: from this point onward we read ONLY sealed_test rows.
sealed = [
    x
    for x in scores
    if x["panel"] == "sealed_test"
]

assert len(sealed) == 600

assert len({
    (
        x["target_role"],
        x["pair_id"],
    )
    for x in sealed
}) == 100


cal = json.loads(
    CALIBRATION.read_text()
)

assert (
    cal["sealed_test_metrics_seen"]
    is False
)

assert cal["primary_endpoint"] == {
    "k": 3,
    "threshold_scheme": "role_specific",
    "metric": "balanced_accuracy",
}


results = {}


for k in [1, 2, 3]:

    aa = aggregate(
        sealed,
        k,
    )

    assert len(aa) == 200

    frozen = cal["k"][str(k)]

    global_t = float(
        frozen["global_threshold"]
    )

    role_t = {
        str(role): float(t)
        for role, t
        in frozen["role_thresholds"].items()
    }

    # --------------------------------------------------------
    # GLOBAL frozen threshold
    # --------------------------------------------------------

    global_rows = []

    for x in aa:
        global_rows.append({
            **x,
            "threshold": global_t,
            "pred": int(
                x["score"] >= global_t
            ),
        })

    gm = classification_metrics(
        global_rows
    )

    gm["exact_matched_pair"] = (
        exact_pair_rate(global_rows)
    )

    grank, _ = matched_pair_ranking(
        global_rows
    )

    gm["matched_pair_ranking"] = (
        grank
    )

    # --------------------------------------------------------
    # ROLE-SPECIFIC frozen thresholds
    # --------------------------------------------------------

    role_rows = []

    for x in aa:
        t = role_t[x["role"]]

        role_rows.append({
            **x,
            "threshold": t,
            "pred": int(
                x["score"] >= t
            ),
        })

    rm = classification_metrics(
        role_rows
    )

    rm["exact_matched_pair"] = (
        exact_pair_rate(role_rows)
    )

    rrank, pair_wins = (
        matched_pair_ranking(
            role_rows
        )
    )

    rm["matched_pair_ranking"] = (
        rrank
    )

    if k == 3:
        rm["balanced_accuracy_ci95"] = (
            bootstrap_primary(
                role_rows
            )
        )

    # --------------------------------------------------------
    # Per-role sealed performance
    # --------------------------------------------------------

    per_role = {}

    for role in sorted(role_t):

        rr = [
            x
            for x in role_rows
            if x["role"] == role
        ]

        # 5 held-out POS + 5 held-out NEG
        assert len(rr) == 10

        m = classification_metrics(rr)

        m["exact_matched_pair"] = (
            exact_pair_rate(
                [
                    x
                    for x in role_rows
                    if x["role"] == role
                ]
            )
            if False
            else None
        )

        per_role[role] = m

    results[str(k)] = {
        "global_threshold":
            gm,

        "role_specific_threshold":
            rm,

        "per_role":
            per_role,
    }


primary = (
    results["3"][
        "role_specific_threshold"
    ]
)


report = {
    "experiment":
        "ROLE DIRECT SENSOR — SEALED EXTERNAL TEST V1",

    "sealed_test_opened":
        True,

    "protocol": {
        "sensor_training":
            "old matched dataset only",

        "threshold_fitting":
            "fresh calibration panel only",

        "evaluation":
            "fresh sealed-test panel only",

        "system_composition_overlap":
            0,

        "query_panel":
            "fresh and distinct from calibration/training",

        "primary_endpoint":
            "K=3 role-specific frozen-threshold balanced accuracy",
    },

    "provenance": {
        "scores_sha256":
            sha256_file(SCORES),

        "frozen_calibration_sha256":
            sha256_file(CALIBRATION),
    },

    "primary_result": {
        "balanced_accuracy":
            primary["balanced_accuracy"],

        "balanced_accuracy_ci95":
            primary[
                "balanced_accuracy_ci95"
            ],

        "accuracy":
            primary["accuracy"],

        "tpr":
            primary["tpr"],

        "fpr":
            primary["fpr"],

        "exact_matched_pair":
            primary[
                "exact_matched_pair"
            ],

        "matched_pair_ranking":
            primary[
                "matched_pair_ranking"
            ],
    },

    "results":
        results,
}


OUT.write_text(
    json.dumps(
        report,
        indent=2,
    )
)


print("=" * 108)
print(
    "ROLE DIRECT SENSOR — SEALED EXTERNAL TEST"
)
print("=" * 108)

print()
print("PRIMARY ENDPOINT")
print(
    "K=3 / ROLE-SPECIFIC FROZEN THRESHOLD"
)

print(
    "BALANCED ACCURACY =",
    f"{primary['balanced_accuracy']:.4f}",
    "CI95=",
    (
        f"[{primary['balanced_accuracy_ci95'][0]:.4f}, "
        f"{primary['balanced_accuracy_ci95'][1]:.4f}]"
    ),
)

print(
    "ACCURACY          =",
    f"{primary['accuracy']:.4f}",
)

print(
    "TPR               =",
    f"{primary['tpr']:.4f}",
)

print(
    "FPR               =",
    f"{primary['fpr']:.4f}",
)

print(
    "PAIR EXACT        =",
    f"{primary['exact_matched_pair']:.4f}",
)

print(
    "MATCHED RANKING   =",
    f"{primary['matched_pair_ranking']:.4f}",
)


print()
print("=" * 108)
print("ALL K")
print("=" * 108)

for k in ["1", "2", "3"]:

    g = results[k][
        "global_threshold"
    ]

    r = results[k][
        "role_specific_threshold"
    ]

    print()
    print(
        f"K={k}"
    )

    print(
        "  GLOBAL "
        f"BA={g['balanced_accuracy']:.3f} "
        f"ACC={g['accuracy']:.3f} "
        f"TPR={g['tpr']:.3f} "
        f"FPR={g['fpr']:.3f} "
        f"PAIR_EXACT={g['exact_matched_pair']:.3f} "
        f"RANK={g['matched_pair_ranking']:.3f}"
    )

    print(
        "  ROLE   "
        f"BA={r['balanced_accuracy']:.3f} "
        f"ACC={r['accuracy']:.3f} "
        f"TPR={r['tpr']:.3f} "
        f"FPR={r['fpr']:.3f} "
        f"PAIR_EXACT={r['exact_matched_pair']:.3f} "
        f"RANK={r['matched_pair_ranking']:.3f}"
    )


print()
print("=" * 108)
print(
    "PER ROLE — PRIMARY K=3"
)
print("=" * 108)

for role, x in (
    results["3"][
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
print("SEALED_TEST_OPEN_PASS")
print("REPORT =", OUT)
