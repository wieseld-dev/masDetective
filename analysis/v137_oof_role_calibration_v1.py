#!/usr/bin/env python3

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


ROOT = Path("artifacts/v137_presence_v1")
PRED = ROOT / "model_v1/dev_predictions.jsonl"
OUT = ROOT / "oof_role_calibration_v1"

OUT.mkdir(parents=True, exist_ok=True)


def readj(path):
    with Path(path).open() as f:
        return [
            json.loads(x)
            for x in f
            if x.strip()
        ]


def sha(path):
    return hashlib.sha256(
        Path(path).read_bytes()
    ).hexdigest()


def sigmoid(x):
    x = np.asarray(x, dtype=float)
    return 1.0 / (1.0 + np.exp(-x))


def poisson_binomial(probs):
    d = np.array([1.0], dtype=float)

    for p in probs:
        nxt = np.zeros(len(d) + 1)
        nxt[:-1] += d * (1.0 - p)
        nxt[1:] += d * p
        d = nxt

    return d


rows = readj(PRED)

if len(rows) != 760:
    raise RuntimeError(
        f"Expected 760 DEV predictions, got {len(rows)}"
    )

if len({
    x["example_id"]
    for x in rows
}) != 760:
    raise RuntimeError(
        "Duplicate DEV example IDs"
    )

roles = sorted({
    x["target_role"]
    for x in rows
})

if len(roles) != 20:
    raise RuntimeError(
        f"Expected 20 roles, got {len(roles)}"
    )

role_index = {
    r: i
    for i, r in enumerate(roles)
}

systems = sorted({
    str(x["system_id"])
    for x in rows
})

if len(systems) != 38:
    raise RuntimeError(
        f"Expected 38 systems, got {len(systems)}"
    )


y = np.asarray(
    [int(x["true_label"]) for x in rows],
    dtype=int,
)

logits = np.asarray(
    [float(x["logit"]) for x in rows],
    dtype=float,
)

system_id = np.asarray(
    [str(x["system_id"]) for x in rows]
)

target_role = np.asarray(
    [x["target_role"] for x in rows]
)


def features(indices, kind):
    indices = np.asarray(indices, dtype=int)

    z = logits[indices]
    rr = target_role[indices]

    if kind == "global":
        return z.reshape(-1, 1)

    if kind == "role_intercept":
        # One common logit slope + one intercept per role.
        X = np.zeros(
            (len(indices), 1 + len(roles)),
            dtype=float,
        )

        X[:, 0] = z

        for row_i, role in enumerate(rr):
            X[
                row_i,
                1 + role_index[role],
            ] = 1.0

        return X

    if kind == "role_slope_intercept":
        # Each role receives its own:
        #   intercept_r + slope_r * raw_logit
        R = len(roles)

        X = np.zeros(
            (len(indices), 2 * R),
            dtype=float,
        )

        for row_i, (role, value) in enumerate(
            zip(rr, z)
        ):
            j = role_index[role]

            X[row_i, j] = 1.0
            X[row_i, R + j] = value

        return X

    raise ValueError(kind)


def make_calibrator(kind):
    if kind == "global":
        return LogisticRegression(
            solver="lbfgs",
            C=1e6,
            max_iter=10000,
            fit_intercept=True,
        )

    return LogisticRegression(
        solver="lbfgs",
        C=1.0,
        max_iter=10000,
        fit_intercept=False,
    )


METHODS = [
    "raw",
    "global_platt",
    "role_intercept",
    "role_slope_intercept",
]

oof = {
    m: np.full(
        len(rows),
        np.nan,
        dtype=float,
    )
    for m in METHODS
}

oof["raw"][:] = sigmoid(logits)


# ------------------------------------------------------------
# Leave-one-SYSTEM-out.
#
# All 20 examples from held-out system stay completely outside
# calibration fitting for that fold.
# ------------------------------------------------------------

for fold_i, sid in enumerate(systems):
    test_idx = np.where(
        system_id == sid
    )[0]

    train_idx = np.where(
        system_id != sid
    )[0]

    if len(test_idx) != 20:
        raise RuntimeError(
            f"{sid}: expected 20 test rows, got {len(test_idx)}"
        )

    specs = [
        ("global_platt", "global"),
        ("role_intercept", "role_intercept"),
        (
            "role_slope_intercept",
            "role_slope_intercept",
        ),
    ]

    for output_name, kind in specs:
        clf = make_calibrator(kind)

        clf.fit(
            features(train_idx, kind),
            y[train_idx],
        )

        p = clf.predict_proba(
            features(test_idx, kind)
        )[:, 1]

        oof[output_name][test_idx] = p

    print(
        f"[fold {fold_i+1:02d}/{len(systems)}] "
        f"held_out={sid}",
        flush=True,
    )


for name, p in oof.items():
    if not np.isfinite(p).all():
        raise RuntimeError(
            f"{name}: incomplete OOF predictions"
        )


def presence_metrics(p):
    pred = (
        np.asarray(p) >= 0.5
    ).astype(int)

    result = {
        "roc_auc":
            float(
                roc_auc_score(
                    y,
                    p,
                )
            ),

        "pr_auc":
            float(
                average_precision_score(
                    y,
                    p,
                )
            ),

        "balanced_accuracy":
            float(
                balanced_accuracy_score(
                    y,
                    pred,
                )
            ),

        "precision":
            float(
                precision_score(
                    y,
                    pred,
                    zero_division=0,
                )
            ),

        "recall":
            float(
                recall_score(
                    y,
                    pred,
                    zero_division=0,
                )
            ),

        "f1":
            float(
                f1_score(
                    y,
                    pred,
                    zero_division=0,
                )
            ),
    }

    per_role_auc = {}

    for role in roles:
        idx = np.where(
            target_role == role
        )[0]

        yy = y[idx]
        pp = np.asarray(p)[idx]

        if len(np.unique(yy)) == 2:
            per_role_auc[role] = float(
                roc_auc_score(
                    yy,
                    pp,
                )
            )

    result["macro_role_auc"] = float(
        np.mean(
            list(
                per_role_auc.values()
            )
        )
    )

    result["per_role_auc"] = (
        per_role_auc
    )

    return result


def k_metrics(p):
    p = np.asarray(p)

    by_system = defaultdict(list)

    for i, row in enumerate(rows):
        by_system[
            str(row["system_id"])
        ].append(i)

    exact = 0
    exact3 = 0
    exact4 = 0

    n3 = 0
    n4 = 0

    abs_errors = []
    exact_role_set = 0

    pred_k_counter = Counter()
    confusion = Counter()

    sum_p_by_true_k = defaultdict(list)

    system_results = []

    for sid in systems:
        idx = by_system[sid]

        if len(idx) != 20:
            raise RuntimeError(
                f"{sid}: expected 20 rows"
            )

        true_k_values = {
            int(
                rows[i]["K_eval_only"]
            )
            for i in idx
        }

        if len(true_k_values) != 1:
            raise RuntimeError(
                f"{sid}: inconsistent K"
            )

        true_k = next(
            iter(true_k_values)
        )

        role_prob = {
            rows[i]["target_role"]:
                float(p[i])
            for i in idx
        }

        true_roles = {
            rows[i]["target_role"]
            for i in idx
            if int(
                rows[i]["true_label"]
            ) == 1
        }

        ordered = sorted(
            role_prob
        )

        probs = [
            role_prob[r]
            for r in ordered
        ]

        posterior = poisson_binomial(
            probs
        )

        pred_k = max(
            range(3, 9),
            key=lambda k:
                posterior[k],
        )

        ok = int(
            pred_k == true_k
        )

        exact += ok
        abs_errors.append(
            abs(
                pred_k
                - true_k
            )
        )

        pred_k_counter[
            pred_k
        ] += 1

        confusion[
            (true_k, pred_k)
        ] += 1

        sum_p_by_true_k[
            true_k
        ].append(
            float(
                sum(probs)
            )
        )

        if true_k == 3:
            n3 += 1
            exact3 += ok

        elif true_k == 4:
            n4 += 1
            exact4 += ok

        ranked = sorted(
            ordered,
            key=lambda r:
                role_prob[r],
            reverse=True,
        )

        pred_roles = set(
            ranked[:pred_k]
        )

        exact_role_set += int(
            pred_roles
            == true_roles
        )

        system_results.append({
            "system_id": sid,
            "true_K": true_k,
            "pred_K": pred_k,
            "sum_probabilities":
                float(
                    sum(probs)
                ),
            "posterior_K": [
                float(z)
                for z in posterior
            ],
        })

    return {
        "systems":
            len(systems),

        "exact_K":
            exact / len(systems),

        "K3_exact":
            exact3 / n3,

        "K4_exact":
            exact4 / n4,

        "K_MAE":
            float(
                np.mean(
                    abs_errors
                )
            ),

        "exact_role_set":
            exact_role_set
            / len(systems),

        "pred_K_distribution": {
            str(k): int(v)
            for k, v
            in sorted(
                pred_k_counter.items()
            )
        },

        "true_to_pred": {
            f"{a}->{b}": int(v)
            for (a, b), v
            in sorted(
                confusion.items()
            )
        },

        "mean_sum_p_by_true_K": {
            str(k):
                float(
                    np.mean(v)
                )
            for k, v
            in sorted(
                sum_p_by_true_k.items()
            )
        },

        "system_results":
            system_results,
    }


results = {}

for method in METHODS:
    results[method] = {
        "presence":
            presence_metrics(
                oof[method]
            ),

        "K":
            k_metrics(
                oof[method]
            ),
    }


# ------------------------------------------------------------
# Majority K=3 baseline.
# ------------------------------------------------------------

majority_exact = (
    24 / 38
)

results[
    "majority_K3_baseline"
] = {
    "exact_K":
        majority_exact,

    "K3_exact":
        1.0,

    "K4_exact":
        0.0,

    "K_MAE":
        14 / 38,
}


# ------------------------------------------------------------
# Fit candidate calibrators on ALL DEV.
#
# These are saved only as candidate parameters.
# They are NOT yet declared final/frozen and internal test
# remains untouched.
# ------------------------------------------------------------

full_idx = np.arange(
    len(rows)
)

candidate_params = {
    "roles":
        roles,

    "dev_predictions_sha256":
        sha(PRED),

    "internal_test_used":
        False,
}

for name, kind in [
    ("global_platt", "global"),
    (
        "role_intercept",
        "role_intercept",
    ),
    (
        "role_slope_intercept",
        "role_slope_intercept",
    ),
]:
    clf = make_calibrator(
        kind
    )

    clf.fit(
        features(
            full_idx,
            kind,
        ),
        y,
    )

    candidate_params[name] = {
        "C":
            float(clf.C),

        "fit_intercept":
            bool(
                clf.fit_intercept
            ),

        "coef":
            [
                float(z)
                for z
                in clf.coef_.reshape(-1)
            ],

        "intercept":
            [
                float(z)
                for z
                in clf.intercept_.reshape(-1)
            ],
    }


# ------------------------------------------------------------
# Save OOF predictions.
# ------------------------------------------------------------

with (
    OUT / "oof_predictions.jsonl"
).open("w") as f:
    for i, row in enumerate(rows):
        out = {
            "example_id":
                row["example_id"],

            "system_id":
                row["system_id"],

            "target_role":
                row["target_role"],

            "true_label":
                int(
                    row["true_label"]
                ),

            "true_K":
                int(
                    row["K_eval_only"]
                ),

            "raw_logit":
                float(
                    row["logit"]
                ),
        }

        for method in METHODS:
            out[
                f"p_{method}"
            ] = float(
                oof[method][i]
            )

        f.write(
            json.dumps(
                out
            )
            + "\n"
        )


(
    OUT / "REPORT.json"
).write_text(
    json.dumps(
        {
            "version":
                "v137_oof_role_calibration_v1",

            "cv":
                "leave-one-system-out",

            "folds":
                len(systems),

            "rows":
                len(rows),

            "systems":
                len(systems),

            "roles":
                len(roles),

            "results":
                results,

            "internal_test_used":
                False,
        },
        indent=2,
        sort_keys=True,
    )
)

(
    OUT /
    "CANDIDATE_FULL_DEV_CALIBRATORS.json"
).write_text(
    json.dumps(
        candidate_params,
        indent=2,
        sort_keys=True,
    )
)


# ------------------------------------------------------------
# Human-readable report.
# ------------------------------------------------------------

lines = []

def P(x=""):
    lines.append(str(x))
    print(x)


P("=" * 100)
P("V13.7 LEAVE-ONE-SYSTEM-OUT ROLE-AWARE CALIBRATION")
P("=" * 100)

P(
    f"DEV rows    : {len(rows)}"
)
P(
    f"DEV systems : {len(systems)}"
)
P(
    f"roles       : {len(roles)}"
)
P(
    "internal test used: NO"
)

P()
P(
    "K=3 MAJORITY BASELINE: "
    f"{majority_exact:.2%}"
)

for method in METHODS:
    pres = (
        results[method][
            "presence"
        ]
    )

    km = (
        results[method]["K"]
    )

    P()
    P("-" * 100)
    P(method.upper())
    P("-" * 100)

    P(
        "Presence ROC-AUC : "
        f"{pres['roc_auc']:.4f}"
    )

    P(
        "Presence PR-AUC  : "
        f"{pres['pr_auc']:.4f}"
    )

    P(
        "Presence BalAcc  : "
        f"{pres['balanced_accuracy']:.4f}"
    )

    P(
        "Macro role AUC   : "
        f"{pres['macro_role_auc']:.4f}"
    )

    P(
        "Exact K          : "
        f"{km['exact_K']:.2%}"
    )

    P(
        "K=3 exact        : "
        f"{km['K3_exact']:.2%}"
    )

    P(
        "K=4 exact        : "
        f"{km['K4_exact']:.2%}"
    )

    P(
        "K MAE            : "
        f"{km['K_MAE']:.4f}"
    )

    P(
        "Exact role set   : "
        f"{km['exact_role_set']:.2%}"
    )

    P(
        "Pred K dist      : "
        f"{km['pred_K_distribution']}"
    )

    P(
        "True -> Pred     : "
        f"{km['true_to_pred']}"
    )

    P(
        "Mean sum(p)|K    : "
        f"{km['mean_sum_p_by_true_K']}"
    )


P()
P("=" * 100)
P("ROLE-SLOPE DETAILS")
P("=" * 100)

role_auc = (
    results[
        "role_slope_intercept"
    ][
        "presence"
    ][
        "per_role_auc"
    ]
)

for role in roles:
    P(
        f"{role:28s} "
        f"AUC={role_auc[role]:.4f}"
    )


P()
P("INTERNAL TEST USED: NO")
P("OOF ROLE-AWARE CALIBRATION: DONE")

(
    OUT / "REPORT.txt"
).write_text(
    "\n".join(lines)
    + "\n"
)
