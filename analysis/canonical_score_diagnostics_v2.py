#!/usr/bin/env python3

import json
import math
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    balanced_accuracy_score,
    confusion_matrix,
    accuracy_score,
)
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


SEED = 1337

ROOT = Path("artifacts/canonical_v136b_target_eval")

INPUTS = {
    "k3": ROOT / "k3" / "scores.jsonl",
    "k4": ROOT / "k4" / "scores.jsonl",
    "high": ROOT / "high" / "scores.jsonl",
}

OUT = Path("artifacts/canonical_score_diagnostics_v2")
OUT.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

REPORT = []


def say(*args):
    s = " ".join(str(x) for x in args)
    print(s, flush=True)
    REPORT.append(s)


def sigmoid(x):
    x = np.asarray(x, dtype=float)
    x = np.clip(x, -50, 50)
    return 1.0 / (1.0 + np.exp(-x))


def binary_metrics(y, p):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    pred = (p >= 0.5).astype(int)

    out = {}

    if len(np.unique(y)) == 2:
        out["auc"] = float(roc_auc_score(y, p))
    else:
        out["auc"] = float("nan")

    out["balanced_acc"] = float(balanced_accuracy_score(y, pred))

    tp = int(np.sum((y == 1) & (pred == 1)))
    fn = int(np.sum((y == 1) & (pred == 0)))
    fp = int(np.sum((y == 0) & (pred == 1)))
    tn = int(np.sum((y == 0) & (pred == 0)))

    out["tpr"] = tp / (tp + fn) if tp + fn else float("nan")
    out["fpr"] = fp / (fp + tn) if fp + tn else float("nan")

    return out


def poisson_binomial_pmf(ps):
    ps = np.asarray(ps, dtype=float)

    dp = np.zeros(len(ps) + 1, dtype=float)
    dp[0] = 1.0

    used = 0

    for p in ps:
        used += 1
        nxt = np.zeros_like(dp)

        for k in range(used + 1):
            if k <= used - 1:
                nxt[k] += dp[k] * (1.0 - p)

            if k > 0:
                nxt[k] += dp[k - 1] * p

        dp = nxt

    return dp


# ---------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------

rows = []

for cohort, path in INPUTS.items():
    if not path.exists():
        raise FileNotFoundError(path)

    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            r = json.loads(line)
            r["cohort"] = cohort

            truth_roles = list(r["truth_roles_training_only"])

            # Training/evaluation labels only. NEVER inference features.
            r["truth_present"] = int(r["target"] in truth_roles)
            r["K_true"] = len(truth_roles)

            rows.append(r)

df = pd.DataFrame(rows)

say("=" * 90)
say("CANONICAL SCORE DIAGNOSTICS V2")
say("=" * 90)

say("rows:", len(df))
say("systems:", df["system_id"].nunique())
say("targets:", df["target"].nunique())

say()
say("rows per cohort:")
for cohort, g in df.groupby("cohort"):
    say(
        f"  {cohort:5s}",
        f"rows={len(g):3d}",
        f"systems={g['system_id'].nunique():2d}",
    )


# ---------------------------------------------------------------------
# Integrity checks
# ---------------------------------------------------------------------

system_cohort_counts = df.groupby("system_id")["cohort"].nunique()

if int(system_cohort_counts.max()) != 1:
    raise RuntimeError("A system_id occurs in multiple cohorts.")

targets = (
    df[["target", "candidate_index"]]
    .drop_duplicates()
    .sort_values(["candidate_index", "target"])
)

# Determine canonical role order by candidate_index.
role_order = (
    df.groupby("target")["candidate_index"]
    .median()
    .sort_values()
    .index
    .tolist()
)

if len(role_order) != 20:
    raise RuntimeError(f"Expected 20 roles, got {len(role_order)}")

say()
say("canonical role order:")
for i, r in enumerate(role_order):
    say(f"  {i:02d} {r}")

for sid, g in df.groupby("system_id"):
    if len(g) != 20:
        raise RuntimeError(f"{sid}: expected 20 rows, got {len(g)}")

    if g["target"].nunique() != 20:
        raise RuntimeError(f"{sid}: targets are not unique")

    if g["K_true"].nunique() != 1:
        raise RuntimeError(f"{sid}: inconsistent K_true")

    truth_sets = {
        tuple(sorted(x))
        for x in g["truth_roles_training_only"]
    }

    if len(truth_sets) != 1:
        raise RuntimeError(f"{sid}: inconsistent truth_roles_training_only")

say()
say("INTEGRITY CHECK: PASS")


# ---------------------------------------------------------------------
# OOF calibration
#
# For each held-out MAS:
#   train calibrators only on OTHER MAS systems.
#
# Two labels are evaluated:
#   1. truth_target_activated  = decode original V13.6b task
#   2. truth_present           = direct target-installed calibration
#
# The second one is better aligned with exact-K.
# ---------------------------------------------------------------------

role_to_idx = {r: i for i, r in enumerate(role_order)}


def global_features(frame):
    x = np.zeros((len(frame), 1 + len(role_order)), dtype=float)

    x[:, 0] = frame["logit"].to_numpy(dtype=float)

    for j, role in enumerate(frame["target"]):
        x[j, 1 + role_to_idx[role]] = 1.0

    return x


def oof_calibrate(label_col):
    probs = np.full(len(df), np.nan, dtype=float)
    method = np.empty(len(df), dtype=object)

    systems = sorted(df["system_id"].unique())

    for heldout in systems:
        train = df[df["system_id"] != heldout]
        test = df[df["system_id"] == heldout]

        # Global fallback: raw logit + target-role one-hot.
        gy = train[label_col].to_numpy(dtype=int)

        global_model = None

        if len(np.unique(gy)) == 2:
            global_model = LogisticRegression(
                max_iter=2000,
                C=1.0,
                solver="lbfgs",
                random_state=SEED,
            )

            global_model.fit(global_features(train), gy)

        for idx, row in test.iterrows():
            role = row["target"]

            role_train = train[train["target"] == role]
            ry = role_train[label_col].to_numpy(dtype=int)

            # Prefer target-specific Platt calibration.
            # Require both classes and at least two examples of each.
            counts = Counter(ry.tolist())

            if (
                len(counts) == 2
                and min(counts.values()) >= 2
            ):
                model = LogisticRegression(
                    max_iter=2000,
                    C=1.0,
                    solver="lbfgs",
                    random_state=SEED,
                )

                Xtr = role_train[["logit"]].to_numpy(dtype=float)
                Xte = np.array([[float(row["logit"])]], dtype=float)

                model.fit(Xtr, ry)

                probs[idx] = model.predict_proba(Xte)[0, 1]
                method[idx] = "per_role_platt"

            elif global_model is not None:
                one = pd.DataFrame([row])
                probs[idx] = global_model.predict_proba(
                    global_features(one)
                )[0, 1]

                method[idx] = "global_role_fallback"

            else:
                # Should almost never happen.
                probs[idx] = float(sigmoid(float(row["logit"])))
                method[idx] = "raw_sigmoid_fallback"

    if np.isnan(probs).any():
        raise RuntimeError(f"NaN OOF probabilities for {label_col}")

    return probs, method


say()
say("=" * 90)
say("OOF CALIBRATION")
say("=" * 90)

df["p_activation_oof"], df["activation_cal_method"] = (
    oof_calibrate("truth_target_activated")
)

df["p_presence_oof"], df["presence_cal_method"] = (
    oof_calibrate("truth_present")
)


def report_row_metrics(label_col, prob_col, title):
    say()
    say(title)

    groups = [
        ("k3", df[df["cohort"] == "k3"]),
        ("k4", df[df["cohort"] == "k4"]),
        ("high", df[df["cohort"] == "high"]),
        ("all", df),
    ]

    for name, g in groups:
        m = binary_metrics(
            g[label_col].to_numpy(),
            g[prob_col].to_numpy(),
        )

        say(
            f"{name:5s}",
            f"n={len(g):3d}",
            f"AUC={m['auc']:.4f}",
            f"BalAcc={m['balanced_acc']:.4f}",
            f"TPR={m['tpr']:.4f}",
            f"FPR={m['fpr']:.4f}",
        )


report_row_metrics(
    "truth_target_activated",
    "p_activation_oof",
    "A) Decode original ACTIVATION target",
)

report_row_metrics(
    "truth_present",
    "p_presence_oof",
    "B) Direct INSTALLED-PRESENCE calibration",
)


# Per-role presence performance.
per_role_rows = []

for role in role_order:
    g = df[df["target"] == role]

    m = binary_metrics(
        g["truth_present"].to_numpy(),
        g["p_presence_oof"].to_numpy(),
    )

    per_role_rows.append(
        {
            "target": role,
            "n": len(g),
            "positives": int(g["truth_present"].sum()),
            **m,
        }
    )

per_role_df = pd.DataFrame(per_role_rows)
per_role_df.to_csv(OUT / "per_role_presence_oof.csv", index=False)

say()
say("Presence calibration — per target:")
for _, r in per_role_df.iterrows():
    say(
        f"{r['target']:26s}",
        f"n={int(r['n']):2d}",
        f"pos={int(r['positives']):2d}",
        f"auc={r['auc']:.3f}",
        f"bal={r['balanced_acc']:.3f}",
    )


# ---------------------------------------------------------------------
# System-level exact K from calibrated presence probabilities
# ---------------------------------------------------------------------

say()
say("=" * 90)
say("POISSON-BINOMIAL K POSTERIOR")
say("=" * 90)

system_rows = []

for sid, g in df.groupby("system_id"):
    g = g.set_index("target").loc[role_order].reset_index()

    K_true = int(g["K_true"].iloc[0])
    cohort = g["cohort"].iloc[0]

    ps = g["p_presence_oof"].to_numpy(dtype=float)

    pmf = poisson_binomial_pmf(ps)

    # RAW diagnostic: do not artificially constrain K.
    Khat_raw = int(np.argmax(pmf))
    conf_raw = float(np.max(pmf))

    # Optional experimental-range posterior, reported separately.
    lo, hi = 3, 8
    sub = pmf[lo : hi + 1]

    if sub.sum() > 0:
        sub_norm = sub / sub.sum()
        Khat_3_8 = int(lo + np.argmax(sub_norm))
        conf_3_8 = float(np.max(sub_norm))
    else:
        Khat_3_8 = -1
        conf_3_8 = float("nan")

    pred_roles = [
        role
        for role, p in zip(role_order, ps)
        if p >= 0.5
    ]

    truth_roles = sorted(g["truth_roles_training_only"].iloc[0])

    hard_K = len(pred_roles)

    row = {
        "system_id": sid,
        "cohort": cohort,
        "K_true": K_true,
        "Khat_hard_05": hard_K,
        "Khat_poibin_raw": Khat_raw,
        "poibin_raw_conf": conf_raw,
        "Khat_poibin_3_8": Khat_3_8,
        "poibin_3_8_conf": conf_3_8,
        "hard_exact_set": int(sorted(pred_roles) == truth_roles),
        "pred_roles": "|".join(sorted(pred_roles)),
        "truth_roles": "|".join(truth_roles),
    }

    for k, value in enumerate(pmf):
        row[f"P_K_{k}"] = float(value)

    system_rows.append(row)

system_df = pd.DataFrame(system_rows)
system_df.to_csv(OUT / "system_k_oof.csv", index=False)


def system_metrics(frame, pred_col):
    y = frame["K_true"].to_numpy(dtype=int)
    p = frame[pred_col].to_numpy(dtype=int)

    return {
        "exact": float(np.mean(y == p)),
        "mae": float(np.mean(np.abs(y - p))),
    }


for cohort in ["k3", "k4", "high", "all"]:
    if cohort == "all":
        g = system_df
    else:
        g = system_df[system_df["cohort"] == cohort]

    say()
    say(f"[{cohort}] systems={len(g)}")

    for col in [
        "Khat_hard_05",
        "Khat_poibin_raw",
        "Khat_poibin_3_8",
    ]:
        m = system_metrics(g, col)

        say(
            f"  {col:20s}",
            f"exactK={m['exact']:.4f}",
            f"MAE={m['mae']:.4f}",
        )

    say(
        "  exact installed role set @0.5 =",
        f"{g['hard_exact_set'].mean():.4f}",
    )


# ---------------------------------------------------------------------
# Direct K fingerprint: K3 vs K4
# ---------------------------------------------------------------------

say()
say("=" * 90)
say("DIRECT K FINGERPRINT: K3 vs K4")
say("=" * 90)

small_ids = sorted(
    df[df["cohort"].isin(["k3", "k4"])]["system_id"].unique()
)

X_rows = []
y_rows = []
id_rows = []


def entropy_bern(p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


for sid in small_ids:
    g = (
        df[df["system_id"] == sid]
        .set_index("target")
        .loc[role_order]
        .reset_index()
    )

    logits = g["logit"].to_numpy(dtype=float)
    probs = sigmoid(logits)

    compact_stats = np.array(
        [
            logits.mean(),
            logits.std(),
            logits.min(),
            logits.max(),
            np.quantile(logits, 0.10),
            np.quantile(logits, 0.25),
            np.quantile(logits, 0.50),
            np.quantile(logits, 0.75),
            np.quantile(logits, 0.90),
            np.sum(logits > 0.0),
            np.sum(logits > -0.5),
            np.sum(logits > -1.0),
            probs.mean(),
            probs.std(),
            entropy_bern(probs).mean(),
        ],
        dtype=float,
    )

    # No system ID, stage, rounds, truth roles, mode, or any other leakage.
    features = np.concatenate([logits, compact_stats])

    X_rows.append(features)
    y_rows.append(int(g["K_true"].iloc[0]))
    id_rows.append(sid)

X = np.vstack(X_rows)
y = np.asarray(y_rows, dtype=int)

say("systems:", len(y))
say("feature_dim:", X.shape[1])
say("class counts:", dict(Counter(y.tolist())))

if set(np.unique(y)) != {3, 4}:
    raise RuntimeError(
        f"Expected direct-K labels {{3,4}}, got {sorted(np.unique(y))}"
    )


def make_models(seed):
    return {
        "logistic": make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=5000,
                C=1.0,
                solver="lbfgs",
                random_state=seed,
            ),
        ),
        "extratrees": ExtraTreesClassifier(
            n_estimators=300,
            random_state=seed,
            class_weight="balanced",
            min_samples_leaf=1,
            n_jobs=-1,
        ),
        "randomforest": RandomForestClassifier(
            n_estimators=300,
            random_state=seed,
            class_weight="balanced",
            min_samples_leaf=1,
            n_jobs=-1,
        ),
    }


cv = RepeatedStratifiedKFold(
    n_splits=4,
    n_repeats=25,
    random_state=SEED,
)

direct_results = {}

for model_name in ["logistic", "extratrees", "randomforest"]:
    fold_acc = []
    predictions_by_sample = defaultdict(list)

    for fold_no, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        model = make_models(SEED + fold_no)[model_name]

        model.fit(X[train_idx], y[train_idx])

        pred = model.predict(X[test_idx])

        fold_acc.append(
            accuracy_score(y[test_idx], pred)
        )

        for idx, phat in zip(test_idx, pred):
            predictions_by_sample[int(idx)].append(int(phat))

    # Each sample is predicted once in each repeat => 25 predictions.
    aggregate_pred = []

    for i in range(len(y)):
        votes = predictions_by_sample[i]

        if len(votes) != 25:
            raise RuntimeError(
                f"sample {i}: expected 25 OOF votes, got {len(votes)}"
            )

        count = Counter(votes)

        # 25 is odd, so no tie.
        aggregate_pred.append(count.most_common(1)[0][0])

    aggregate_pred = np.asarray(aggregate_pred, dtype=int)

    cm = confusion_matrix(
        y,
        aggregate_pred,
        labels=[3, 4],
    )

    result = {
        "mean_fold_accuracy": float(np.mean(fold_acc)),
        "std_fold_accuracy": float(np.std(fold_acc)),
        "aggregate_accuracy": float(
            accuracy_score(y, aggregate_pred)
        ),
        "confusion_matrix": cm.tolist(),
        "aggregate_pred": aggregate_pred.tolist(),
    }

    direct_results[model_name] = result

    say()
    say(model_name)
    say(
        "  repeated-CV fold accuracy =",
        f"{result['mean_fold_accuracy']:.4f}",
        "+/-",
        f"{result['std_fold_accuracy']:.4f}",
    )
    say(
        "  aggregate 25-repeat OOF accuracy =",
        f"{result['aggregate_accuracy']:.4f}",
    )
    say("  confusion [rows=true 3,4; cols=pred 3,4]:")
    say(" ", cm.tolist())


# ---------------------------------------------------------------------
# Permutation sanity test
#
# Use logistic regression because it is the least flexible direct-K model.
# A fixed 4-fold stratified CV is used for both real and permuted labels.
# ---------------------------------------------------------------------

say()
say("=" * 90)
say("LABEL PERMUTATION SANITY — LOGISTIC")
say("=" * 90)

rng = np.random.default_rng(SEED)
N_PERM = 500


def fixed_cv_score(X, y, seed):
    splitter = StratifiedKFold(
        n_splits=4,
        shuffle=True,
        random_state=seed,
    )

    scores = []

    for fold_no, (tr, te) in enumerate(splitter.split(X, y)):
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=5000,
                C=1.0,
                solver="lbfgs",
                random_state=seed + fold_no,
            ),
        )

        model.fit(X[tr], y[tr])
        scores.append(
            accuracy_score(y[te], model.predict(X[te]))
        )

    return float(np.mean(scores))


real_fixed = fixed_cv_score(X, y, SEED)

perm_scores = []

for i in range(N_PERM):
    yp = rng.permutation(y)

    perm_scores.append(
        fixed_cv_score(
            X,
            yp,
            SEED + 1000 + i,
        )
    )

perm_scores = np.asarray(perm_scores, dtype=float)

p_value = (
    1.0 + np.sum(perm_scores >= real_fixed)
) / (
    1.0 + N_PERM
)

say("real fixed-CV accuracy =", f"{real_fixed:.4f}")
say("permutation mean       =", f"{perm_scores.mean():.4f}")
say("permutation std        =", f"{perm_scores.std():.4f}")
say("permutation p          =", f"{p_value:.6f}")


# ---------------------------------------------------------------------
# Save per-system direct-K predictions
# ---------------------------------------------------------------------

direct_out = pd.DataFrame(
    {
        "system_id": id_rows,
        "K_true": y,
    }
)

for model_name, result in direct_results.items():
    direct_out[f"Khat_{model_name}"] = result["aggregate_pred"]

direct_out.to_csv(
    OUT / "direct_k3_k4_predictions.csv",
    index=False,
)


# Save row-level OOF predictions, but omit large truth-role object from CSV.
df[
    [
        "system_id",
        "cohort",
        "candidate_index",
        "target",
        "logit",
        "truth_target_activated",
        "truth_present",
        "K_true",
        "p_activation_oof",
        "p_presence_oof",
        "activation_cal_method",
        "presence_cal_method",
    ]
].to_csv(
    OUT / "row_oof_predictions.csv",
    index=False,
)


summary = {
    "n_rows": int(len(df)),
    "n_systems": int(df["system_id"].nunique()),
    "roles": role_order,
    "direct_k3_k4": direct_results,
    "permutation": {
        "model": "logistic",
        "n_permutations": N_PERM,
        "real_fixed_cv_accuracy": real_fixed,
        "permutation_mean": float(perm_scores.mean()),
        "permutation_std": float(perm_scores.std()),
        "p_value": float(p_value),
    },
}

with (OUT / "summary.json").open("w") as f:
    json.dump(summary, f, indent=2)


say()
say("=" * 90)
say("OUTPUTS")
say("=" * 90)
say(OUT / "row_oof_predictions.csv")
say(OUT / "per_role_presence_oof.csv")
say(OUT / "system_k_oof.csv")
say(OUT / "direct_k3_k4_predictions.csv")
say(OUT / "summary.json")

with (OUT / "REPORT.txt").open("w") as f:
    f.write("\n".join(REPORT) + "\n")

say(OUT / "REPORT.txt")
