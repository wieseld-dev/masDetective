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
from sklearn.model_selection import KFold, GroupKFold
from sklearn.metrics import pairwise_distances
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIG
# ============================================================

CACHE = Path("artifacts/execution_operator_big/api_runs.jsonl")
OUT = Path("artifacts/pure_order_spectral")
OUT.mkdir(parents=True, exist_ok=True)

SEED = 20260916

TRAIN_N = 100
N_TOTAL = 160

D = 64
AUG = 2 * D + 1

SPECIALISTS = "ABC"

# Pure order contrasts:
#
# x y t  versus  y x t
#
# Same agents.
# Same terminal specialist.
# Only order of first two changes.
ORDER_PAIRS = [
    ("ABC", "BAC"),
    ("ACB", "CAB"),
    ("BCA", "CBA"),
]

PREFIX = {
    k: [
        "".join(p)
        for p in itertools.permutations(
            SPECIALISTS,
            k
        )
    ]
    for k in (1, 2, 3)
}

ALL_PREFIX_SEQS = (
    [""]
    + PREFIX[1]
    + PREFIX[2]
    + PREFIX[3]
)

PREFIX_KEYS = [
    "R" + s
    for s in ALL_PREFIX_SEQS
]

M_KEYS = [
    ("RM" if s == "" else "R" + s + "M")
    for s in ALL_PREFIX_SEQS
]

RIDGE_ALPHAS = [
    0.01,
    0.1,
    1.0,
    10.0,
    100.0,
    1000.0,
]

KRR_ALPHAS = [
    0.01,
    0.1,
    1.0,
    10.0,
]

BOOTSTRAPS = 1000
NULL_REPS = 2000


print("=" * 100)
print("PURE ORDER SPECTRAL BENCHMARK")
print("=" * 100)
print()
print("Question:")
print("Does non-commutative operator geometry predict")
print("which BENIGN queries best expose PURE execution order?")
print()
print("Pairs:")
for x in ORDER_PAIRS:
    print("  ", x[0], "vs", x[1])


# ============================================================
# LOAD
# ============================================================

rows = []

with CACHE.open(encoding="utf-8") as f:
    for line in f:
        if line.strip():
            rows.append(
                json.loads(line)
            )

rows = {
    int(r["position"]): r
    for r in rows
}

rows = [
    rows[k]
    for k in sorted(rows)
][:N_TOTAL]

assert len(rows) == N_TOTAL

TRAIN = np.arange(TRAIN_N)
TEST = np.arange(
    TRAIN_N,
    N_TOTAL
)

print()
print("TOTAL:", len(rows))
print("TRAIN:", len(TRAIN))
print("DEV-HOLDOUT:", len(TEST))


# ============================================================
# EMBEDDING SPACE
#
# Important:
# PCA/scaler fitted exactly on TRAIN q + prefix states.
# M outputs are transformed afterward, but do NOT influence
# the semantic coordinate system.
# ============================================================

enc = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    device="cpu"
)

tasks = [
    r["task"]
    for r in rows
]

print("\nEncoding q...")

q_raw = enc.encode(
    tasks,
    normalize_embeddings=True,
    convert_to_numpy=True,
    batch_size=64
)

all_needed_keys = list(
    dict.fromkeys(
        PREFIX_KEYS + M_KEYS
    )
)

raw = {}

for key in all_needed_keys:

    print("Encoding", key)

    raw[key] = enc.encode(
        [
            r["outputs"][key]
            for r in rows
        ],
        normalize_embeddings=True,
        convert_to_numpy=True,
        batch_size=64
    )


# Fit semantic basis only with prefix states,
# preserving comparability with the earlier audit.

pool = [
    q_raw[TRAIN]
]

for key in PREFIX_KEYS:

    pool.append(
        raw[key][TRAIN]
    )

pool = np.vstack(pool)

pca = PCA(
    n_components=D,
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
    key: transform(value)
    for key, value in raw.items()
}


# ============================================================
# METRICS
# ============================================================

def safe_rho(x, y):

    r = spearmanr(
        x,
        y
    ).statistic

    if not np.isfinite(r):
        return 0.0

    return float(r)


def lift20(score, actual):

    k = max(
        1,
        int(np.ceil(
            0.20 * len(score)
        ))
    )

    idx = np.argsort(
        score
    )[-k:]

    base = np.mean(actual)

    if abs(base) < 1e-12:
        return 1.0

    return float(
        np.mean(actual[idx])
        /
        base
    )


# ============================================================
# TRUE PURE-ORDER TARGETS
#
# PRE:
#   RABC vs RBAC
#
# POST:
#   RABCM vs RBACM
#
# POST is actual externally visible black-box output.
# ============================================================

PAIR_NAMES = [
    a + "_vs_" + b
    for a, b in ORDER_PAIRS
]

Y_PRE = np.zeros(
    (N_TOTAL, len(ORDER_PAIRS))
)

Y_POST = np.zeros_like(
    Y_PRE
)

for j, (a, b) in enumerate(
    ORDER_PAIRS
):

    ha = data["R" + a]
    hb = data["R" + b]

    Y_PRE[:, j] = np.mean(
        (ha - hb) ** 2,
        axis=1
    )

    ya = data["R" + a + "M"]
    yb = data["R" + b + "M"]

    Y_POST[:, j] = np.mean(
        (ya - yb) ** 2,
        axis=1
    )


# ============================================================
# R MODEL:
#
# q -> hidden R state
#
# Required because at probe-selection time the attacker does NOT
# know true R(q).
# ============================================================

def fit_r_model():

    kf = KFold(
        n_splits=5,
        shuffle=True,
        random_state=SEED
    )

    cv_rows = []

    for alpha in RIDGE_ALPHAS:

        vals = []

        for tr, va in kf.split(TRAIN):

            tr_ids = TRAIN[tr]
            va_ids = TRAIN[va]

            model = Ridge(
                alpha=alpha
            )

            model.fit(
                q[tr_ids],
                data["R"][tr_ids]
            )

            pred = model.predict(
                q[va_ids]
            )

            vals.append(
                np.mean(
                    (
                        pred
                        -
                        data["R"][va_ids]
                    ) ** 2
                )
            )

        cv_rows.append({
            "component": "R",
            "alpha": alpha,
            "cv_mse": np.mean(vals)
        })

    cv = pd.DataFrame(
        cv_rows
    )

    alpha = float(
        cv.sort_values(
            "cv_mse"
        ).iloc[0]["alpha"]
    )

    model = Ridge(
        alpha=alpha
    )

    model.fit(
        q[TRAIN],
        data["R"][TRAIN]
    )

    return model, cv


Rmodel, r_cv = fit_r_model()

Rhat = Rmodel.predict(
    q
)

print(
    "\nSelected R alpha:",
    Rmodel.alpha
)


# ============================================================
# TASK-CONDITIONED AFFINE OPERATORS
#
# h' = A h + B q + c
#
# Convert to ONE fixed homogeneous matrix:
#
#        [ A  B  c ]
# W  =   [ 0  I  0 ]
#        [ 0  0  1 ]
#
# acting on:
#
# s = [h ; q ; 1]
#
# This is important:
# W_A itself no longer changes per prompt.
#
# Prompt dependence enters through s(q).
# ============================================================

def specialist_transitions(agent):

    out = []

    for depth in (1, 2, 3):

        for seq in PREFIX[depth]:

            if seq[-1] != agent:
                continue

            parent_seq = seq[:-1]

            parent = "R" + parent_seq
            child = "R" + seq

            out.append(
                (parent, child)
            )

    return out


def merger_transitions():

    # Deliberately train M only through specialist-count 2.
    #
    # TEST use at count 3 is therefore cross-depth generalization.

    out = []

    for depth in (0, 1, 2):

        seqs = (
            [""]
            if depth == 0
            else PREFIX[depth]
        )

        for seq in seqs:

            parent = "R" + seq

            child = (
                "RM"
                if seq == ""
                else "R" + seq + "M"
            )

            out.append(
                (parent, child)
            )

    return out


def build_transition_dataset(
    transitions
):

    X = []
    Y = []
    groups = []

    for i in TRAIN:

        for parent, child in transitions:

            X.append(
                np.concatenate([
                    data[parent][i],
                    q[i]
                ])
            )

            Y.append(
                data[child][i]
            )

            groups.append(i)

    return (
        np.asarray(X),
        np.asarray(Y),
        np.asarray(groups)
    )


def fit_operator(
    name,
    transitions
):

    X, Y, groups = (
        build_transition_dataset(
            transitions
        )
    )

    gkf = GroupKFold(
        n_splits=5
    )

    cv_rows = []

    for alpha in RIDGE_ALPHAS:

        vals = []

        for tr, va in gkf.split(
            X,
            Y,
            groups
        ):

            model = Ridge(
                alpha=alpha
            )

            model.fit(
                X[tr],
                Y[tr]
            )

            pred = model.predict(
                X[va]
            )

            vals.append(
                np.mean(
                    (
                        pred
                        -
                        Y[va]
                    ) ** 2
                )
            )

        cv_rows.append({
            "component": name,
            "alpha": alpha,
            "cv_mse": np.mean(vals)
        })

    cv = pd.DataFrame(
        cv_rows
    )

    best_alpha = float(
        cv.sort_values(
            "cv_mse"
        ).iloc[0]["alpha"]
    )

    model = Ridge(
        alpha=best_alpha
    )

    model.fit(
        X,
        Y
    )

    # --------------------------------------------
    # Convert affine task-conditioned regression
    # into square homogeneous execution operator.
    # --------------------------------------------

    coef = model.coef_

    A = coef[:, :D]
    B = coef[:, D:]

    c = model.intercept_

    W = np.zeros(
        (AUG, AUG),
        dtype=np.float64
    )

    W[:D, :D] = A
    W[:D, D:2 * D] = B
    W[:D, -1] = c

    # q persists unchanged.
    W[D:2 * D, D:2 * D] = np.eye(D)

    # homogeneous constant persists.
    W[-1, -1] = 1.0

    return (
        model,
        W,
        cv
    )


OPS = {}
OP_MODELS = {}
op_cv_all = []

for agent in SPECIALISTS:

    model, W, cv = fit_operator(
        agent,
        specialist_transitions(
            agent
        )
    )

    OP_MODELS[agent] = model
    OPS[agent] = W

    op_cv_all.append(cv)

    print(
        f"Selected {agent} alpha:",
        model.alpha
    )


Mmodel, WM, mcv = fit_operator(
    "M",
    merger_transitions()
)

OP_MODELS["M"] = Mmodel
OPS["M"] = WM

op_cv_all.append(mcv)

print(
    "Selected M alpha:",
    Mmodel.alpha
)


pd.concat(
    [r_cv] + op_cv_all,
    ignore_index=True
).to_csv(
    OUT / "operator_cv.csv",
    index=False
)


# ============================================================
# COMPOSITION
# ============================================================

def compose(seq):

    M = np.eye(
        AUG,
        dtype=np.float64
    )

    for a in seq:

        M = OPS[a] @ M

    return M


# Feasible states:
#
# realistic: predicted R from q only
# oracle: true hidden R

S_PRED = np.concatenate(
    [
        Rhat,
        q,
        np.ones(
            (N_TOTAL, 1)
        )
    ],
    axis=1
)

S_TRUE = np.concatenate(
    [
        data["R"],
        q,
        np.ones(
            (N_TOTAL, 1)
        )
    ],
    axis=1
)


# ============================================================
# DIRECT q -> INFORMATIVE-NESS BASELINES
# ============================================================

def standardize_targets(
    Y,
    ids
):

    mu = Y[ids].mean(
        axis=0,
        keepdims=True
    )

    sd = Y[ids].std(
        axis=0,
        keepdims=True
    )

    sd = np.maximum(
        sd,
        1e-8
    )

    return mu, sd


def evaluate_target_matrix(
    pred,
    target,
    ids
):

    rhos = []

    for j in range(
        target.shape[1]
    ):

        rhos.append(
            safe_rho(
                pred[:, j],
                target[ids, j]
            )
        )

    return float(
        np.mean(rhos)
    )


def fit_q_ridge(
    target,
    label
):

    kf = KFold(
        n_splits=5,
        shuffle=True,
        random_state=SEED
    )

    cv_rows = []

    for alpha in RIDGE_ALPHAS:

        rs = []

        for tr, va in kf.split(TRAIN):

            tr_ids = TRAIN[tr]
            va_ids = TRAIN[va]

            mu, sd = standardize_targets(
                target,
                tr_ids
            )

            model = Ridge(
                alpha=alpha
            )

            model.fit(
                q[tr_ids],
                (
                    target[tr_ids]
                    -
                    mu
                ) / sd
            )

            pred = model.predict(
                q[va_ids]
            )

            rs.append(
                evaluate_target_matrix(
                    pred,
                    target,
                    va_ids
                )
            )

        cv_rows.append({
            "observable": label,
            "model": "q_only_ridge",
            "alpha": alpha,
            "cv_rho": np.mean(rs)
        })

    cv = pd.DataFrame(
        cv_rows
    )

    alpha = float(
        cv.sort_values(
            "cv_rho",
            ascending=False
        ).iloc[0]["alpha"]
    )

    mu, sd = standardize_targets(
        target,
        TRAIN
    )

    model = Ridge(
        alpha=alpha
    )

    model.fit(
        q[TRAIN],
        (
            target[TRAIN]
            -
            mu
        ) / sd
    )

    pred = model.predict(
        q[TEST]
    )

    return (
        pred,
        cv
    )


def fit_q_rbf(
    target,
    label
):

    X = q[TRAIN]

    d2 = pairwise_distances(
        X,
        metric="sqeuclidean"
    )

    nz = d2[
        d2 > 0
    ]

    med = np.median(nz)

    base_gamma = (
        1.0
        /
        max(med, 1e-12)
    )

    gammas = [
        0.25 * base_gamma,
        1.0 * base_gamma,
        4.0 * base_gamma,
    ]

    kf = KFold(
        n_splits=5,
        shuffle=True,
        random_state=SEED
    )

    cv_rows = []

    for gamma in gammas:

        for alpha in KRR_ALPHAS:

            rs = []

            for tr, va in kf.split(TRAIN):

                tr_ids = TRAIN[tr]
                va_ids = TRAIN[va]

                mu, sd = standardize_targets(
                    target,
                    tr_ids
                )

                model = KernelRidge(
                    kernel="rbf",
                    gamma=gamma,
                    alpha=alpha
                )

                model.fit(
                    q[tr_ids],
                    (
                        target[tr_ids]
                        -
                        mu
                    ) / sd
                )

                pred = model.predict(
                    q[va_ids]
                )

                rs.append(
                    evaluate_target_matrix(
                        pred,
                        target,
                        va_ids
                    )
                )

            cv_rows.append({
                "observable": label,
                "model": "q_only_rbf",
                "alpha": alpha,
                "gamma": gamma,
                "cv_rho": np.mean(rs)
            })

    cv = pd.DataFrame(
        cv_rows
    )

    best = cv.sort_values(
        "cv_rho",
        ascending=False
    ).iloc[0]

    mu, sd = standardize_targets(
        target,
        TRAIN
    )

    model = KernelRidge(
        kernel="rbf",
        gamma=float(
            best["gamma"]
        ),
        alpha=float(
            best["alpha"]
        )
    )

    model.fit(
        q[TRAIN],
        (
            target[TRAIN]
            -
            mu
        ) / sd
    )

    pred = model.predict(
        q[TEST]
    )

    return (
        pred,
        cv
    )


print(
    "\nFitting direct q-only baselines..."
)

Q_RIDGE_PRE, cv_ridge_pre = (
    fit_q_ridge(
        Y_PRE,
        "PRE"
    )
)

Q_RIDGE_POST, cv_ridge_post = (
    fit_q_ridge(
        Y_POST,
        "POST"
    )
)

Q_RBF_PRE, cv_rbf_pre = (
    fit_q_rbf(
        Y_PRE,
        "PRE"
    )
)

Q_RBF_POST, cv_rbf_post = (
    fit_q_rbf(
        Y_POST,
        "POST"
    )
)

pd.concat(
    [
        cv_ridge_pre,
        cv_ridge_post,
        cv_rbf_pre,
        cv_rbf_post,
    ],
    ignore_index=True
).to_csv(
    OUT / "direct_baseline_cv.csv",
    index=False
)


# ============================================================
# SPECTRAL SCORES
#
# D = Delta^T Delta
#
# Full:
#   s^T D s = ||Delta s||^2
#
# Top-k spectral:
#   sum_{j<=k} lambda_j (v_j^T s)^2
#
# This directly tests whether principal eigenvector is enough.
# ============================================================

def get_spectral_scores(
    Delta,
    S
):

    Dmat = (
        Delta.T @ Delta
    )

    eigvals, eigvecs = np.linalg.eigh(
        Dmat
    )

    idx = np.argsort(
        eigvals
    )[::-1]

    eigvals = np.maximum(
        eigvals[idx],
        0.0
    )

    eigvecs = eigvecs[:, idx]

    norm2 = np.sum(
        S ** 2,
        axis=1
    )

    norm2 = np.maximum(
        norm2,
        1e-12
    )

    projection = (
        S @ eigvecs
    )

    weighted = (
        projection ** 2
        *
        eigvals[None, :]
    )

    raw_full = np.sum(
        weighted,
        axis=1
    )

    scores = {
        "operator_raw":
            raw_full,

        "rayleigh_all":
            raw_full / norm2,
    }

    for k in (
        1,
        4,
        16
    ):

        kk = min(
            k,
            len(eigvals)
        )

        scores[
            f"eig{k}"
        ] = (
            np.sum(
                weighted[:, :kk],
                axis=1
            )
            /
            norm2
        )

    trace = np.sum(
        eigvals
    )

    if trace <= 1e-12:

        frac1 = 0.0
        frac4 = 0.0
        frac16 = 0.0

    else:

        frac1 = (
            eigvals[0] / trace
        )

        frac4 = (
            np.sum(
                eigvals[:4]
            ) / trace
        )

        frac16 = (
            np.sum(
                eigvals[:16]
            ) / trace
        )

    diagnostics = {
        "lambda1":
            eigvals[0],

        "trace":
            trace,

        "lambda1_fraction":
            frac1,

        "top4_fraction":
            frac4,

        "top16_fraction":
            frac16,
    }

    return (
        scores,
        diagnostics
    )


# ============================================================
# BUILD SCORES
# ============================================================

score_store = {}
actual_store = {}

spectral_diag = []
comm_checks = []

for pair_idx, (
    seq1,
    seq2
) in enumerate(
    ORDER_PAIRS
):

    name = (
        seq1
        + "_vs_"
        + seq2
    )

    # --------------------------------------------
    # Pure order operator contrast PRE M
    # --------------------------------------------

    F1 = compose(seq1)
    F2 = compose(seq2)

    Delta_pre = (
        F1 - F2
    )

    # Verify exact non-commutative form:
    #
    # seq1 = x y t
    # seq2 = y x t
    #
    # Delta =
    # W_t (W_y W_x - W_x W_y)

    x = seq1[0]
    y = seq1[1]
    t = seq1[2]

    comm_form = (
        OPS[t]
        @ (
            OPS[y] @ OPS[x]
            -
            OPS[x] @ OPS[y]
        )
    )

    comm_error = np.linalg.norm(
        Delta_pre
        -
        comm_form,
        ord="fro"
    )

    # --------------------------------------------
    # POST M
    #
    # M is trained only through depth 2.
    # Applying it here at depth 3 is held-out depth.
    # --------------------------------------------

    Delta_post = (
        OPS["M"]
        @ Delta_pre
    )

    post_comm_form = (
        OPS["M"]
        @ comm_form
    )

    post_comm_error = np.linalg.norm(
        Delta_post
        -
        post_comm_form,
        ord="fro"
    )

    comm_checks.append({
        "pair": name,
        "pre_commutator_error":
            comm_error,
        "post_commutator_error":
            post_comm_error,
    })

    # --------------------------------------------
    # Spectral scores on realistic predicted state
    # --------------------------------------------

    pre_pred, pre_diag = (
        get_spectral_scores(
            Delta_pre,
            S_PRED[TEST]
        )
    )

    post_pred, post_diag = (
        get_spectral_scores(
            Delta_post,
            S_PRED[TEST]
        )
    )

    # ORACLE true R
    pre_true, _ = (
        get_spectral_scores(
            Delta_pre,
            S_TRUE[TEST]
        )
    )

    post_true, _ = (
        get_spectral_scores(
            Delta_post,
            S_TRUE[TEST]
        )
    )

    spectral_diag.append({
        "observable": "PRE",
        "pair": name,
        **pre_diag,
    })

    spectral_diag.append({
        "observable": "POST",
        "pair": name,
        **post_diag,
    })

    # --------------------------------------------
    # Actual targets
    # --------------------------------------------

    actual_store[
        ("PRE", name)
    ] = Y_PRE[
        TEST,
        pair_idx
    ]

    actual_store[
        ("POST", name)
    ] = Y_POST[
        TEST,
        pair_idx
    ]

    # --------------------------------------------
    # Direct q-only baselines
    # --------------------------------------------

    score_store[
        (
            "PRE",
            "q_only_ridge",
            name
        )
    ] = Q_RIDGE_PRE[
        :,
        pair_idx
    ]

    score_store[
        (
            "PRE",
            "q_only_rbf",
            name
        )
    ] = Q_RBF_PRE[
        :,
        pair_idx
    ]

    score_store[
        (
            "POST",
            "q_only_ridge",
            name
        )
    ] = Q_RIDGE_POST[
        :,
        pair_idx
    ]

    score_store[
        (
            "POST",
            "q_only_rbf",
            name
        )
    ] = Q_RBF_POST[
        :,
        pair_idx
    ]

    # --------------------------------------------
    # Operator/spectral realistic scores
    # --------------------------------------------

    for method, values in (
        pre_pred.items()
    ):

        score_store[
            (
                "PRE",
                method + "_predR",
                name
            )
        ] = values

    for method, values in (
        post_pred.items()
    ):

        score_store[
            (
                "POST",
                method + "_predR",
                name
            )
        ] = values

    # --------------------------------------------
    # Oracle only for raw operator separation
    # --------------------------------------------

    score_store[
        (
            "PRE",
            "operator_raw_trueR_ORACLE",
            name
        )
    ] = pre_true[
        "operator_raw"
    ]

    score_store[
        (
            "POST",
            "operator_raw_trueR_ORACLE",
            name
        )
    ] = post_true[
        "operator_raw"
    ]

    # --------------------------------------------
    # Does PRE structural score already predict
    # observable POST separation?
    # --------------------------------------------

    score_store[
        (
            "POST",
            "pre_operator_raw_cross",
            name
        )
    ] = pre_pred[
        "operator_raw"
    ]


pd.DataFrame(
    spectral_diag
).to_csv(
    OUT / "spectral_diagnostics.csv",
    index=False
)

pd.DataFrame(
    comm_checks
).to_csv(
    OUT / "commutator_checks.csv",
    index=False
)


# ============================================================
# PER-PAIR METRICS
# ============================================================

metric_rows = []

for (
    observable,
    method,
    pair
), score in score_store.items():

    actual = actual_store[
        (
            observable,
            pair
        )
    ]

    metric_rows.append({
        "observable": observable,
        "method": method,
        "pair": pair,
        "rho": safe_rho(
            score,
            actual
        ),
        "lift20": lift20(
            score,
            actual
        ),
    })


metrics = pd.DataFrame(
    metric_rows
)

metrics.to_csv(
    OUT / "per_pair_metrics.csv",
    index=False
)


# ============================================================
# MACRO SUMMARY
# ============================================================

macro = (
    metrics
    .groupby(
        [
            "observable",
            "method"
        ]
    )
    .agg(
        rho_mean=(
            "rho",
            "mean"
        ),
        rho_median=(
            "rho",
            "median"
        ),
        lift_mean=(
            "lift20",
            "mean"
        ),
        positive_pairs=(
            "rho",
            lambda x:
                int(
                    np.sum(
                        np.asarray(x)
                        > 0
                    )
                )
        ),
    )
    .reset_index()
)


# ============================================================
# PAIRED BOOTSTRAP OVER QUERIES
#
# Same resampled prompt IDs are used across methods.
# ============================================================

rng = np.random.default_rng(
    SEED + 91
)

boot_indices = [
    rng.integers(
        0,
        len(TEST),
        len(TEST)
    )
    for _ in range(
        BOOTSTRAPS
    )
]


def bootstrap_macro(
    observable,
    method
):

    vals = []

    for ids in boot_indices:

        rr = []

        for pair in PAIR_NAMES:

            score = score_store[
                (
                    observable,
                    method,
                    pair
                )
            ][ids]

            actual = actual_store[
                (
                    observable,
                    pair
                )
            ][ids]

            rr.append(
                safe_rho(
                    score,
                    actual
                )
            )

        vals.append(
            np.mean(rr)
        )

    return np.asarray(vals)


boot_rows = []

for observable in (
    "PRE",
    "POST"
):

    methods = sorted(
        metrics[
            metrics.observable
            == observable
        ].method.unique()
    )

    direct_boot = (
        bootstrap_macro(
            observable,
            "q_only_ridge"
        )
    )

    direct_mean = (
        macro[
            (
                macro.observable
                == observable
            )
            &
            (
                macro.method
                == "q_only_ridge"
            )
        ]
        .iloc[0]
        .rho_mean
    )

    for method in methods:

        b = bootstrap_macro(
            observable,
            method
        )

        diff = (
            b - direct_boot
        )

        method_mean = (
            macro[
                (
                    macro.observable
                    == observable
                )
                &
                (
                    macro.method
                    == method
                )
            ]
            .iloc[0]
            .rho_mean
        )

        boot_rows.append({
            "observable":
                observable,

            "method":
                method,

            "rho_mean":
                method_mean,

            "rho_ci_low":
                np.quantile(
                    b,
                    0.025
                ),

            "rho_ci_high":
                np.quantile(
                    b,
                    0.975
                ),

            "delta_vs_q_ridge":
                (
                    method_mean
                    -
                    direct_mean
                ),

            "delta_ci_low":
                np.quantile(
                    diff,
                    0.025
                ),

            "delta_ci_high":
                np.quantile(
                    diff,
                    0.975
                ),
        })


boot_df = pd.DataFrame(
    boot_rows
)

boot_df.to_csv(
    OUT / "bootstrap_summary.csv",
    index=False
)

macro = macro.merge(
    boot_df[
        [
            "observable",
            "method",
            "rho_ci_low",
            "rho_ci_high",
            "delta_vs_q_ridge",
            "delta_ci_low",
            "delta_ci_high",
        ]
    ],
    on=[
        "observable",
        "method"
    ],
    how="left"
)


# ============================================================
# RANDOM NULL
# ============================================================

null_rows = []

rng_null = np.random.default_rng(
    SEED + 313
)

for observable in (
    "PRE",
    "POST"
):

    rho_null = []
    lift_null = []

    for _ in range(
        NULL_REPS
    ):

        rr = []
        ll = []

        for pair in PAIR_NAMES:

            actual = actual_store[
                (
                    observable,
                    pair
                )
            ]

            random_score = (
                rng_null.random(
                    len(actual)
                )
            )

            rr.append(
                safe_rho(
                    random_score,
                    actual
                )
            )

            ll.append(
                lift20(
                    random_score,
                    actual
                )
            )

        rho_null.append(
            np.mean(rr)
        )

        lift_null.append(
            np.mean(ll)
        )

    null_rows.append({
        "observable":
            observable,

        "rho_null_mean":
            np.mean(rho_null),

        "rho_null_p025":
            np.quantile(
                rho_null,
                .025
            ),

        "rho_null_p975":
            np.quantile(
                rho_null,
                .975
            ),

        "lift_null_mean":
            np.mean(lift_null),

        "lift_null_p025":
            np.quantile(
                lift_null,
                .025
            ),

        "lift_null_p975":
            np.quantile(
                lift_null,
                .975
            ),
    })


null_df = pd.DataFrame(
    null_rows
)

null_df.to_csv(
    OUT / "random_null.csv",
    index=False
)


# Null p for macro rho

pvals = []

for _, row in macro.iterrows():

    obs = row["observable"]

    rng_p = np.random.default_rng(
        SEED
        + (
            1
            if obs == "PRE"
            else 2
        )
    )

    null = []

    for _ in range(
        NULL_REPS
    ):

        rr = []

        for pair in PAIR_NAMES:

            actual = actual_store[
                (
                    obs,
                    pair
                )
            ]

            score = (
                rng_p.random(
                    len(actual)
                )
            )

            rr.append(
                safe_rho(
                    score,
                    actual
                )
            )

        null.append(
            np.mean(rr)
        )

    null = np.asarray(
        null
    )

    p = (
        1
        +
        np.sum(
            null
            >= row["rho_mean"]
        )
    ) / (
        1
        +
        len(null)
    )

    pvals.append(p)

macro[
    "random_null_p"
] = pvals


macro.to_csv(
    OUT / "macro_summary.csv",
    index=False
)


# ============================================================
# OUTPUT
# ============================================================

print("\n" + "=" * 110)
print("COMMUTATOR IDENTITY CHECK")
print("=" * 110)

print(
    pd.DataFrame(
        comm_checks
    )
    .round(12)
    .to_string(
        index=False
    )
)


print("\n" + "=" * 110)
print("SPECTRAL CONCENTRATION")
print("=" * 110)

diag_df = pd.DataFrame(
    spectral_diag
)

print(
    diag_df[
        [
            "observable",
            "pair",
            "lambda1_fraction",
            "top4_fraction",
            "top16_fraction",
        ]
    ]
    .round(5)
    .to_string(
        index=False
    )
)


print("\n" + "=" * 110)
print("PER-PAIR RESULTS")
print("=" * 110)

print(
    metrics
    .pivot_table(
        index=[
            "observable",
            "pair"
        ],
        columns="method",
        values="rho"
    )
    .round(4)
    .to_string()
)


print("\n" + "=" * 110)
print("MACRO PURE-ORDER RESULTS")
print("=" * 110)

display_cols = [
    "observable",
    "method",
    "rho_mean",
    "rho_ci_low",
    "rho_ci_high",
    "lift_mean",
    "positive_pairs",
    "delta_vs_q_ridge",
    "delta_ci_low",
    "delta_ci_high",
    "random_null_p",
]

print(
    macro[
        display_cols
    ]
    .sort_values(
        [
            "observable",
            "rho_mean"
        ],
        ascending=[
            True,
            False
        ]
    )
    .round(5)
    .to_string(
        index=False
    )
)


print("\n" + "=" * 110)
print("RANDOM NULL")
print("=" * 110)

print(
    null_df
    .round(5)
    .to_string(
        index=False
    )
)


print("\n" + "=" * 110)
print("HOW TO READ THIS")
print("=" * 110)

print("""
This benchmark isolates PURE ORDER:

    ABC vs BAC
    ACB vs CAB
    BCA vs CBA

Same agents.
Same number of agents.
Same terminal specialist.
Only execution order differs.


KEY METHODS

q_only_ridge / q_only_rbf
    Strong direct supervised baselines:
    q -> actual order-separation.

operator_raw_predR
    Full predicted operator contrast:
        ||Delta s(q)||^2

    This is algebraically:
        s(q)^T Delta^T Delta s(q)

rayleigh_all_predR
    Same spectral geometry, normalized by state norm.

eig1_predR
    Uses ONLY the principal eigenmode.

eig4_predR / eig16_predR
    Uses progressively more spectral modes.

operator_raw_trueR_ORACLE
    Diagnostic only.
    Uses hidden true R state, unavailable to attacker.

pre_operator_raw_cross
    POST only.
    Tests whether PRE-M order geometry alone ranks
    actual final black-box POST-M separation.


THEORETICAL POINT

For a fixed hypothesis pair:

    D = Delta^T Delta

lambda_max(D) is CONSTANT across prompts.

Therefore lambda_max itself CANNOT rank candidate prompts.

The principal eigenvector gives an unconstrained ideal direction.

For a feasible prompt q, the relevant quantity is:

    s(q)^T D s(q)

or its normalized Rayleigh quotient.


DECISION GATES

1) operator/spectral POST rho >> q-only Ridge
   ------------------------------------------------
   Strong evidence that non-commutative execution geometry
   adds information specifically for pure-order probe selection.

2) eig1 is weak but eig4/eig16/full are strong
   ------------------------------------------------
   Principal eigenvector is NOT enough.
   Structural signal is distributed across spectral modes.

3) eig1 ~= full
   ------------------------------------------------
   A dominant discriminatory eigenmode really exists.

4) PRE works but POST collapses
   ------------------------------------------------
   Order geometry exists internally but is not externally
   observable after the merger.

5) true-R oracle >> predicted-R
   ------------------------------------------------
   Main bottleneck is estimating the initial execution state.

6) q-only ~= operator
   ------------------------------------------------
   Operator algebra currently adds little practical value.

7) everything near random
   ------------------------------------------------
   Pure order is not reliably predictable from unseen benign
   queries under the current 100-query training regime.


IMPORTANT:
The 60 prompts are now DEVELOPMENT HOLDOUT, not a final test set.
No paper-level claim should be made from them.
""")


print("\nSaved:")
for fn in [
    "operator_cv.csv",
    "direct_baseline_cv.csv",
    "commutator_checks.csv",
    "spectral_diagnostics.csv",
    "per_pair_metrics.csv",
    "bootstrap_summary.csv",
    "random_null.csv",
    "macro_summary.csv",
]:
    print(
        OUT / fn
    )
