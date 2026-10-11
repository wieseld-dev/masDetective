#!/usr/bin/env python3

import json
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import KFold
from sklearn.metrics import pairwise_distances
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIG
# ============================================================

CACHE = Path("artifacts/execution_operator_big/api_runs.jsonl")
OUT = Path("artifacts/contrast_generalization_audit")
OUT.mkdir(parents=True, exist_ok=True)

SEED = 20260916
TRAIN_N = 100
BASE_DIM = 64

SPECIALISTS = "ABC"

PREFIX = {
    k: [
        "".join(p)
        for p in itertools.permutations(SPECIALISTS, k)
    ]
    for k in (1, 2, 3)
}

PREFIX_KEYS = (
    ["R"]
    + ["R" + s for k in (1, 2, 3) for s in PREFIX[k]]
)

DEPTH3 = PREFIX[3]
PAIRS = list(itertools.combinations(range(len(DEPTH3)), 2))

print("=" * 86)
print("CONTRAST GENERALIZATION AUDIT")
print("=" * 86)


# ============================================================
# LOAD SAME 160 CAUSAL PROBES
# ============================================================

rows = []

with CACHE.open(encoding="utf-8") as f:
    for line in f:
        if line.strip():
            rows.append(json.loads(line))

rows = {
    int(r["position"]): r
    for r in rows
}

rows = [
    rows[k]
    for k in sorted(rows)
][:160]

assert len(rows) == 160

TRAIN = np.arange(100)
TEST = np.arange(100, 160)

print("TOTAL:", len(rows))
print("TRAIN:", len(TRAIN))
print("TEST:", len(TEST))


# ============================================================
# EMBEDDING
# ============================================================

enc = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    device="cpu"
)

tasks = [r["task"] for r in rows]

print("Encoding q...")
q_raw = enc.encode(
    tasks,
    normalize_embeddings=True,
    convert_to_numpy=True,
    batch_size=64
)

raw = {}

for key in PREFIX_KEYS:

    print("Encoding", key)

    raw[key] = enc.encode(
        [r["outputs"][key] for r in rows],
        normalize_embeddings=True,
        convert_to_numpy=True,
        batch_size=64
    )


# Same style semantic space as previous experiments.
pool = [q_raw[TRAIN]]

for key in PREFIX_KEYS:
    pool.append(raw[key][TRAIN])

pool = np.vstack(pool)

pca = PCA(
    n_components=BASE_DIM,
    random_state=SEED
)

pca.fit(pool)

scaler = StandardScaler()
scaler.fit(
    pca.transform(pool)
)


def transform(x):

    return scaler.transform(
        pca.transform(x)
    ).astype(np.float64)


q = transform(q_raw)

data = {
    key: transform(raw[key])
    for key in PREFIX_KEYS
}


# ============================================================
# TRUE DEPTH-3 STRUCTURAL CONTRAST TARGET
#
# Y[q,pair] = actual semantic separation of the two
# execution paths for that query.
# ============================================================

Y = np.zeros(
    (len(rows), len(PAIRS)),
    dtype=np.float64
)

pair_names = []

for pidx, (i, j) in enumerate(PAIRS):

    a = DEPTH3[i]
    b = DEPTH3[j]

    pair_names.append(
        f"{a}_vs_{b}"
    )

    da = data["R" + a]
    db = data["R" + b]

    Y[:, pidx] = np.mean(
        (da - db) ** 2,
        axis=1
    )

print("Target shape:", Y.shape)
print("Depth-3 pairs:", len(pair_names))


# ============================================================
# METRICS
# ============================================================

def safe_rho(x, y):

    r = spearmanr(x, y).statistic

    return (
        float(r)
        if np.isfinite(r)
        else 0.0
    )


def lift20(score, actual):

    n = len(score)

    k = max(
        1,
        int(np.ceil(0.2 * n))
    )

    idx = np.argsort(score)[-k:]

    denom = np.mean(actual)

    if abs(denom) < 1e-12:
        return 1.0

    return float(
        np.mean(actual[idx])
        /
        denom
    )


def evaluate(pred, true):

    rhos = []
    lifts = []

    for j in range(true.shape[1]):

        rhos.append(
            safe_rho(
                pred[:, j],
                true[:, j]
            )
        )

        lifts.append(
            lift20(
                pred[:, j],
                true[:, j]
            )
        )

    return {
        "rho_mean": float(np.mean(rhos)),
        "rho_median": float(np.median(rhos)),
        "rho_positive_frac": float(
            np.mean(np.array(rhos) > 0)
        ),
        "lift_mean": float(np.mean(lifts)),
        "lift_median": float(np.median(lifts)),
        "per_pair_rho": np.array(rhos),
        "per_pair_lift": np.array(lifts),
    }


# ============================================================
# CLEAN TARGET STANDARDIZATION
#
# Fit Y normalization ONLY on the corresponding training set.
# ============================================================

def fit_target_scale(y):

    mu = y.mean(
        axis=0,
        keepdims=True
    )

    sd = y.std(
        axis=0,
        keepdims=True
    )

    sd = np.maximum(
        sd,
        1e-8
    )

    return mu, sd


# ============================================================
# RIDGE CV
# ============================================================

RIDGE_ALPHAS = [
    0.01,
    0.1,
    1.0,
    10.0,
    100.0,
    1000.0,
]


def select_ridge_alpha(X, y):

    kf = KFold(
        n_splits=5,
        shuffle=True,
        random_state=SEED
    )

    result = []

    for alpha in RIDGE_ALPHAS:

        fold_rho = []
        fold_lift = []

        for tr, va in kf.split(X):

            mu, sd = fit_target_scale(
                y[tr]
            )

            yz = (
                y[tr] - mu
            ) / sd

            model = Ridge(
                alpha=alpha
            )

            model.fit(
                X[tr],
                yz
            )

            pred = model.predict(
                X[va]
            )

            m = evaluate(
                pred,
                y[va]
            )

            fold_rho.append(
                m["rho_mean"]
            )

            fold_lift.append(
                m["lift_mean"]
            )

        result.append({
            "alpha": alpha,
            "cv_rho": np.mean(fold_rho),
            "cv_lift": np.mean(fold_lift),
        })

    df = pd.DataFrame(result)

    # Select ONLY by TRAIN CV rho.
    best = df.sort_values(
        "cv_rho",
        ascending=False
    ).iloc[0]

    return (
        float(best["alpha"]),
        df
    )


def fit_test_ridge(name, X):

    Xtr = X[TRAIN]
    Xte = X[TEST]

    ytr = Y[TRAIN]
    yte = Y[TEST]

    alpha, cv = select_ridge_alpha(
        Xtr,
        ytr
    )

    mu, sd = fit_target_scale(
        ytr
    )

    model = Ridge(
        alpha=alpha
    )

    model.fit(
        Xtr,
        (ytr - mu) / sd
    )

    ptr = model.predict(Xtr)
    pte = model.predict(Xte)

    train_m = evaluate(
        ptr,
        ytr
    )

    test_m = evaluate(
        pte,
        yte
    )

    return {
        "model": name,
        "selected_alpha": alpha,

        "train_rho": train_m["rho_mean"],
        "train_lift": train_m["lift_mean"],

        "test_rho": test_m["rho_mean"],
        "test_rho_median": test_m["rho_median"],
        "test_positive_frac":
            test_m["rho_positive_frac"],

        "test_lift": test_m["lift_mean"],
        "test_lift_median":
            test_m["lift_median"],

        "prediction": pte,
        "cv": cv,
        "per_pair_rho":
            test_m["per_pair_rho"],
    }


# ============================================================
# RBF KERNEL RIDGE -- q ONLY
#
# Stronger nonlinear sanity baseline.
# Hyperparameters selected ONLY by 5-fold TRAIN CV.
# ============================================================

def select_krr(X, y):

    kf = KFold(
        n_splits=5,
        shuffle=True,
        random_state=SEED
    )

    d2 = pairwise_distances(
        X,
        metric="sqeuclidean"
    )

    positive = d2[
        d2 > 0
    ]

    median_d2 = np.median(
        positive
    )

    base_gamma = (
        1.0
        /
        max(median_d2, 1e-12)
    )

    gammas = [
        0.25 * base_gamma,
        1.0 * base_gamma,
        4.0 * base_gamma,
    ]

    alphas = [
        0.01,
        0.1,
        1.0,
        10.0,
    ]

    result = []

    for gamma in gammas:

        for alpha in alphas:

            fold_rho = []
            fold_lift = []

            for tr, va in kf.split(X):

                mu, sd = fit_target_scale(
                    y[tr]
                )

                yz = (
                    y[tr] - mu
                ) / sd

                model = KernelRidge(
                    kernel="rbf",
                    gamma=gamma,
                    alpha=alpha
                )

                model.fit(
                    X[tr],
                    yz
                )

                pred = model.predict(
                    X[va]
                )

                m = evaluate(
                    pred,
                    y[va]
                )

                fold_rho.append(
                    m["rho_mean"]
                )

                fold_lift.append(
                    m["lift_mean"]
                )

            result.append({
                "gamma": gamma,
                "alpha": alpha,
                "cv_rho":
                    np.mean(fold_rho),
                "cv_lift":
                    np.mean(fold_lift),
            })

    df = pd.DataFrame(result)

    best = df.sort_values(
        "cv_rho",
        ascending=False
    ).iloc[0]

    return (
        float(best["gamma"]),
        float(best["alpha"]),
        df
    )


def fit_test_krr_q():

    Xtr = q[TRAIN]
    Xte = q[TEST]

    ytr = Y[TRAIN]
    yte = Y[TEST]

    gamma, alpha, cv = select_krr(
        Xtr,
        ytr
    )

    mu, sd = fit_target_scale(
        ytr
    )

    model = KernelRidge(
        kernel="rbf",
        gamma=gamma,
        alpha=alpha
    )

    model.fit(
        Xtr,
        (ytr - mu) / sd
    )

    ptr = model.predict(Xtr)
    pte = model.predict(Xte)

    train_m = evaluate(
        ptr,
        ytr
    )

    test_m = evaluate(
        pte,
        yte
    )

    return {
        "model": "q_only_rbf",
        "selected_alpha": alpha,
        "selected_gamma": gamma,

        "train_rho": train_m["rho_mean"],
        "train_lift": train_m["lift_mean"],

        "test_rho": test_m["rho_mean"],
        "test_rho_median":
            test_m["rho_median"],
        "test_positive_frac":
            test_m["rho_positive_frac"],

        "test_lift": test_m["lift_mean"],
        "test_lift_median":
            test_m["lift_median"],

        "prediction": pte,
        "cv": cv,
        "per_pair_rho":
            test_m["per_pair_rho"],
    }


# ============================================================
# FEATURES
# ============================================================

# Realistic static probe ranking:
X_q = q

# Diagnostic oracle:
# true hidden R-state is NOT available to the black-box attacker.
# This asks whether R carries useful structure not predictable
# from q alone.
X_qR = np.concatenate(
    [q, data["R"]],
    axis=1
)

X_R = data["R"]


# ============================================================
# RUN
# ============================================================

results = []

print("\nRunning q-only Ridge...")
results.append(
    fit_test_ridge(
        "q_only_ridge",
        X_q
    )
)

print("Running R-only Ridge (diagnostic oracle)...")
results.append(
    fit_test_ridge(
        "R_only_ridge_ORACLE",
        X_R
    )
)

print("Running q+R Ridge (diagnostic oracle)...")
results.append(
    fit_test_ridge(
        "q_plus_R_ridge_ORACLE",
        X_qR
    )
)

print("Running q-only nonlinear RBF baseline...")
results.append(
    fit_test_krr_q()
)


# ============================================================
# SAVE SUMMARY
# ============================================================

summary_rows = []

for r in results:

    summary_rows.append({
        "model": r["model"],
        "alpha": r.get(
            "selected_alpha",
            np.nan
        ),
        "gamma": r.get(
            "selected_gamma",
            np.nan
        ),
        "train_rho": r["train_rho"],
        "train_lift": r["train_lift"],
        "test_rho": r["test_rho"],
        "test_rho_median":
            r["test_rho_median"],
        "test_positive_frac":
            r["test_positive_frac"],
        "test_lift": r["test_lift"],
        "test_lift_median":
            r["test_lift_median"],
    })

summary = pd.DataFrame(
    summary_rows
)

summary.to_csv(
    OUT / "summary.csv",
    index=False
)


# Per-pair table
pair_rows = []

for r in results:

    for name, rho in zip(
        pair_names,
        r["per_pair_rho"]
    ):

        pair_rows.append({
            "model": r["model"],
            "pair": name,
            "rho": rho
        })

pd.DataFrame(
    pair_rows
).to_csv(
    OUT / "per_pair_rho.csv",
    index=False
)


print("\n" + "=" * 100)
print("DIRECT CONTRAST GENERALIZATION")
print("=" * 100)

print(
    summary.round(5).to_string(
        index=False
    )
)


print("\n" + "=" * 100)
print("INTERPRETATION GATES")
print("=" * 100)

print("""
Reference:

  fixed execution operators:
      structural rho roughly 0.16-0.19

  contrast-trained latent model:
      TEST rho roughly 0.09
      despite near-perfect TRAIN contrast fit


A) q-only direct model reaches ~0.16-0.18 or better
   ------------------------------------------------
   The structural contrast function IS predictable from the query.
   The current learned execution-space formulation is the bottleneck.

   Next:
       latent-dimension / decoder-shortcut / architecture ablation.


B) q-only models remain around ~0.05-0.10
   and oracle q+R is also weak
   ------------------------------------------------
   The main problem is generalizing probe informativeness from
   only 100 training probes in this semantic representation.

   Next:
       representation of probes / more diverse probe bank /
       sample-complexity experiment before adding model capacity.


C) q-only is weak, but q+R oracle is substantially stronger
   ------------------------------------------------
   The execution state contains useful discriminative information
   that is not predictable from q alone.

   This suggests static q-only probe ranking is fundamentally
   missing state information.

   Next:
       history/posterior-conditioned active probing rather than
       a single global q -> informativeness predictor.


D) RBF strongly beats Ridge
   ------------------------------------------------
   The q -> contrast map is learnable but nonlinear.
   Then the next representation model should preserve nonlinear
   probe conditioning without allowing query memorization.


IMPORTANT:
The existing 60 TEST prompts have now been inspected repeatedly.
Use them as DEVELOPMENT HOLDOUT from this point onward.

For a final scientific claim, we will later need a fresh,
untouched causal test set.
""")

print("\nSaved:")
print(OUT / "summary.csv")
print(OUT / "per_pair_rho.csv")
