#!/usr/bin/env python3

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sentence_transformers import SentenceTransformer

from topology_algebra import (
    augment,
    compose_path,
    contrast_score,
)


R1_CACHE = Path(
    "artifacts/router_r1/api_runs.jsonl"
)

ANCHOR_CACHE = Path(
    "artifacts/router_anchor/anchors.jsonl"
)

OUT = Path(
    "artifacts/router_anchor"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)


DEV_N = 60
TOTAL_N = 217
DIM = 16

RIDGE_ALPHA = 1.0

N_NULL = 2000
N_BOOT = 2000

BUDGETS = [1, 2, 3, 5, 10, 20]

PAIRS = [
    "RA", "RB", "RC",
    "AM", "BM", "CM",
]

PATHS = [
    "RAM",
    "RBM",
    "RCM",
]

CONTRASTS = [
    ("AB", "RAM", "RBM"),
    ("AC", "RAM", "RCM"),
    ("BC", "RBM", "RCM"),
]


def rho_safe(x, y):

    r = spearmanr(x, y).statistic

    if not np.isfinite(r):
        return 0.0

    return float(r)


def top_lift(score, actual, frac=0.20):

    k = max(
        1,
        math.ceil(
            frac * len(actual)
        )
    )

    ix = np.argsort(score)[-k:]

    return float(
        actual[ix].mean()
        /
        max(actual.mean(), 1e-12)
    )


def load_jsonl(path):

    rows = []

    for line in path.open(encoding="utf-8"):

        if line.strip():
            rows.append(
                json.loads(line)
            )

    rows = {
        int(r["id"]): r
        for r in rows
    }

    return [
        rows[k]
        for k in sorted(rows)
    ]


def operator_from_residual_ridge(
    X,
    Y,
    alpha=1.0,
    affine=True,
):
    """
    Learn a transition:

        Y ~= X + f(X)

    For affine=True:

        y = A x + b

    represented in homogeneous coordinates:

        [y]   [A b] [x]
        [1] = [0 1] [1]

    This preserves ordinary matrix composition.
    """

    d = X.shape[1]

    target_delta = Y - X

    reg = Ridge(
        alpha=alpha,
        fit_intercept=affine
    )

    reg.fit(
        X,
        target_delta
    )

    A = (
        np.eye(d)
        +
        reg.coef_
    )

    if affine:
        b = np.asarray(
            reg.intercept_
        )
    else:
        b = np.zeros(d)

    W = np.eye(d + 1)

    W[:d, :d] = A
    W[:d, d] = b

    return W


def fit_anchored_operators(
    q,
    anchor,
    z,
    affine
):
    """
    Learn primitive transitions directly.

    R:
        q -> R

    A:
        q -> A
        R -> RA

    B:
        q -> B
        R -> RB

    C:
        q -> C
        R -> RC

    M:
        A -> AM
        B -> BM
        C -> CM

    No RAM/RBM/RCM target enters training.
    """

    ops = {}

    # R acts on the original prompt state.
    ops["R"] = operator_from_residual_ridge(
        q[:DEV_N],
        anchor["R"],
        RIDGE_ALPHA,
        affine
    )

    # Specialists observed in two contexts:
    # raw prompt and after R.
    for role in "ABC":

        X = np.vstack([
            q[:DEV_N],
            anchor["R"],
        ])

        Y = np.vstack([
            anchor[role],
            z["R" + role][:DEV_N],
        ])

        ops[role] = operator_from_residual_ridge(
            X,
            Y,
            RIDGE_ALPHA,
            affine
        )

    # M is explicitly shared across three specialist states.
    X_M = np.vstack([
        anchor["A"],
        anchor["B"],
        anchor["C"],
    ])

    Y_M = np.vstack([
        z["AM"][:DEV_N],
        z["BM"][:DEV_N],
        z["CM"][:DEV_N],
    ])

    ops["M"] = operator_from_residual_ridge(
        X_M,
        Y_M,
        RIDGE_ALPHA,
        affine
    )

    return ops


def predicted_scores(
    operators,
    q_test,
    p1,
    p2,
):

    F1 = compose_path(
        operators,
        p1
    )

    F2 = compose_path(
        operators,
        p2
    )

    return contrast_score(
        F1,
        F2,
        q_test
    )


def spectrum_random_scores(
    F1,
    F2,
    q_test,
    rng
):
    """
    Stronger null than Frobenius-matched random matrix.

    Preserve singular values of the topology contrast
    but randomize its input directions.
    """

    d = q_test.shape[1]

    K = (
        F1 - F2
    )[:d, :]

    U, s, _ = np.linalg.svd(
        K,
        full_matrices=False
    )

    # Random orthogonal basis in augmented input space.
    G = rng.normal(
        size=(d + 1, d + 1)
    )

    Q, _ = np.linalg.qr(G)

    Vt_random = Q[:d, :]

    Knull = (
        U
        @ np.diag(s)
        @ Vt_random
    )

    qa = augment(q_test)

    delta = qa @ Knull.T

    return np.sum(
        delta**2,
        axis=1
    )


###############################################################################
# Load
###############################################################################

r1 = load_jsonl(
    R1_CACHE
)

anchors = load_jsonl(
    ANCHOR_CACHE
)

if len(r1) < TOTAL_N:
    raise RuntimeError(
        f"R1 cache only has {len(r1)} rows"
    )

if len(anchors) < DEV_N:
    raise RuntimeError(
        f"Anchor cache only has {len(anchors)} rows"
    )

r1 = r1[:TOTAL_N]
anchors = anchors[:DEV_N]

DEV = np.arange(DEV_N)
TEST = np.arange(
    DEV_N,
    TOTAL_N
)

print("=" * 72)
print("ANCHORED TOPOLOGY ALGEBRA")
print("=" * 72)

print("R1:", len(r1))
print("ANCHORS:", len(anchors))
print("DEV:", len(DEV))
print("TEST:", len(TEST))


###############################################################################
# Embed
###############################################################################

enc = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    device="cpu"
)

tasks = [
    r["task"]
    for r in r1
]

print("\nEncoding prompts...")

q0 = enc.encode(
    tasks,
    normalize_embeddings=True,
    convert_to_numpy=True,
    batch_size=64
)


anchor0 = {}

for role in "RABC":

    print(
        "Encoding anchor",
        role
    )

    anchor0[role] = enc.encode(
        [
            r["outputs"][role]
            for r in anchors
        ],
        normalize_embeddings=True,
        convert_to_numpy=True,
        batch_size=64
    )


z0 = {}

for comp in PAIRS + PATHS:

    print(
        "Encoding",
        comp
    )

    z0[comp] = enc.encode(
        [
            r["outputs"][comp]
            for r in r1
        ],
        normalize_embeddings=True,
        convert_to_numpy=True,
        batch_size=64
    )


###############################################################################
# PCA
#
# IMPORTANT:
# RAM/RBM/RCM are NOT included in PCA fitting.
###############################################################################

pca_fit = [
    q0[DEV],
]

for role in "RABC":
    pca_fit.append(
        anchor0[role]
    )

for comp in PAIRS:
    pca_fit.append(
        z0[comp][DEV]
    )

pca = PCA(
    n_components=DIM,
    random_state=0
)

pca.fit(
    np.vstack(pca_fit)
)

# Deliberately no per-example L2 normalization after PCA.
# Affine operator composition is defined in one common coordinate system.
q = pca.transform(q0)

anchor = {
    role: pca.transform(anchor0[role])
    for role in "RABC"
}

z = {
    comp: pca.transform(z0[comp])
    for comp in PAIRS + PATHS
}


###############################################################################
# Fit two versions:
#
# 1. Strict linear homogeneous operators
# 2. Affine homogeneous operators
###############################################################################

linear_ops = fit_anchored_operators(
    q,
    anchor,
    z,
    affine=False
)

affine_ops = fit_anchored_operators(
    q,
    anchor,
    z,
    affine=True
)


###############################################################################
# Evaluation
###############################################################################

rows_out = []

all_arrays = {}

rng_master = np.random.default_rng(
    20260916
)


for name, p1, p2 in CONTRASTS:

    actual = np.sum(
        (
            z[p1][TEST]
            -
            z[p2][TEST]
        )**2,
        axis=1
    )

    score_linear = predicted_scores(
        linear_ops,
        q[TEST],
        p1,
        p2
    )

    score_affine = predicted_scores(
        affine_ops,
        q[TEST],
        p1,
        p2
    )

    # Strong direct baseline:
    # explicitly receives DEV triple differences.
    direct = Ridge(
        alpha=3.0,
        fit_intercept=True
    )

    direct.fit(
        q[DEV],
        z[p1][DEV]
        -
        z[p2][DEV]
    )

    direct_delta = direct.predict(
        q[TEST]
    )

    score_direct = np.sum(
        direct_delta**2,
        axis=1
    )

    lr = rho_safe(
        score_linear,
        actual
    )

    ar = rho_safe(
        score_affine,
        actual
    )

    dr = rho_safe(
        score_direct,
        actual
    )

    ll = top_lift(
        score_linear,
        actual
    )

    al = top_lift(
        score_affine,
        actual
    )

    dl = top_lift(
        score_direct,
        actual
    )

    # ---------------------------------------------
    # Shuffle null
    # ---------------------------------------------

    shuffle_rhos = []

    for _ in range(N_NULL):

        yy = rng_master.permutation(
            actual
        )

        shuffle_rhos.append(
            rho_safe(
                score_affine,
                yy
            )
        )

    # ---------------------------------------------
    # Spectrum-matched topology null
    # ---------------------------------------------

    F1 = compose_path(
        affine_ops,
        p1
    )

    F2 = compose_path(
        affine_ops,
        p2
    )

    spectrum_rhos = []

    for _ in range(N_NULL):

        ss = spectrum_random_scores(
            F1,
            F2,
            q[TEST],
            rng_master
        )

        spectrum_rhos.append(
            rho_safe(
                ss,
                actual
            )
        )

    p_shuffle = (
        1
        +
        np.sum(
            np.asarray(shuffle_rhos)
            >= ar
        )
    ) / (
        N_NULL + 1
    )

    p_spectrum = (
        1
        +
        np.sum(
            np.asarray(spectrum_rhos)
            >= ar
        )
    ) / (
        N_NULL + 1
    )

    rows_out.append({
        "contrast": name,

        "linear_rho": lr,
        "linear_lift": ll,

        "affine_rho": ar,
        "affine_lift": al,

        "direct_rho": dr,
        "direct_lift": dl,

        "shuffle_rho":
            np.mean(shuffle_rhos),

        "spectrum_null_rho":
            np.mean(spectrum_rhos),

        "p_shuffle": p_shuffle,
        "p_spectrum": p_spectrum,
    })

    all_arrays[name] = {
        "actual": actual,
        "linear": score_linear,
        "affine": score_affine,
        "direct": score_direct,
    }


df = pd.DataFrame(
    rows_out
)

df.to_csv(
    OUT / "anchored_results.csv",
    index=False
)


###############################################################################
# Active budget test
###############################################################################

budget_rows = []

for name, _, _ in CONTRASTS:

    a = all_arrays[name]

    actual = a["actual"]

    for b in BUDGETS:

        for method in [
            "linear",
            "affine",
            "direct",
        ]:

            order = np.argsort(
                a[method]
            )[::-1]

            chosen = order[:b]

            selected_gain = float(
                actual[chosen].mean()
            )

            random_gain = []

            for _ in range(1000):

                ri = rng_master.choice(
                    len(actual),
                    size=b,
                    replace=False
                )

                random_gain.append(
                    actual[ri].mean()
                )

            rg = float(
                np.mean(random_gain)
            )

            budget_rows.append({
                "contrast": name,
                "budget": b,
                "method": method,
                "vs_random":
                    selected_gain
                    /
                    max(rg, 1e-12)
            })


bdf = pd.DataFrame(
    budget_rows
)

bdf.to_csv(
    OUT / "anchored_budget_results.csv",
    index=False
)


###############################################################################
# Probe-level bootstrap
#
# Same sampled probe indices are used across AB/AC/BC to preserve dependence.
###############################################################################

boot_macro_affine_rho = []
boot_macro_linear_rho = []
boot_macro_direct_rho = []

boot_macro_affine_lift = []
boot_macro_linear_lift = []
boot_macro_direct_lift = []

boot_delta_affine_direct_rho = []
boot_delta_affine_direct_lift = []


n_test = len(TEST)

for _ in range(N_BOOT):

    ix = rng_master.integers(
        0,
        n_test,
        size=n_test
    )

    rs_aff = []
    rs_lin = []
    rs_dir = []

    ls_aff = []
    ls_lin = []
    ls_dir = []

    for name, _, _ in CONTRASTS:

        a = all_arrays[name]

        actual = a["actual"][ix]

        sa = a["affine"][ix]
        sl = a["linear"][ix]
        sd = a["direct"][ix]

        rs_aff.append(
            rho_safe(sa, actual)
        )

        rs_lin.append(
            rho_safe(sl, actual)
        )

        rs_dir.append(
            rho_safe(sd, actual)
        )

        ls_aff.append(
            top_lift(sa, actual)
        )

        ls_lin.append(
            top_lift(sl, actual)
        )

        ls_dir.append(
            top_lift(sd, actual)
        )

    ma_r = np.mean(rs_aff)
    ml_r = np.mean(rs_lin)
    md_r = np.mean(rs_dir)

    ma_l = np.mean(ls_aff)
    ml_l = np.mean(ls_lin)
    md_l = np.mean(ls_dir)

    boot_macro_affine_rho.append(ma_r)
    boot_macro_linear_rho.append(ml_r)
    boot_macro_direct_rho.append(md_r)

    boot_macro_affine_lift.append(ma_l)
    boot_macro_linear_lift.append(ml_l)
    boot_macro_direct_lift.append(md_l)

    boot_delta_affine_direct_rho.append(
        ma_r - md_r
    )

    boot_delta_affine_direct_lift.append(
        ma_l - md_l
    )


def ci(x):

    return np.percentile(
        np.asarray(x),
        [2.5, 97.5]
    )


###############################################################################
# Print report
###############################################################################

print("\n" + "=" * 72)
print("ZERO-SHOT HIDDEN-PATH RESULTS")
print("=" * 72)


for _, r in df.iterrows():

    print(
        f"\n[{r['contrast']}]"
    )

    print(
        "Anchored LINEAR "
        f"rho={r['linear_rho']:+.3f} "
        f"lift={r['linear_lift']:.3f}x"
    )

    print(
        "Anchored AFFINE "
        f"rho={r['affine_rho']:+.3f} "
        f"lift={r['affine_lift']:.3f}x"
    )

    print(
        "DIRECT triple-supervised "
        f"rho={r['direct_rho']:+.3f} "
        f"lift={r['direct_lift']:.3f}x"
    )

    print(
        "Shuffle null "
        f"rho={r['shuffle_rho']:+.3f} "
        f"p={r['p_shuffle']:.4f}"
    )

    print(
        "Spectrum-matched null "
        f"rho={r['spectrum_null_rho']:+.3f} "
        f"p={r['p_spectrum']:.4f}"
    )


print("\n" + "=" * 72)
print("MACRO")
print("=" * 72)

print(
    "Anchored LINEAR rho  = "
    f"{df.linear_rho.mean():+.3f}"
)

print(
    "Anchored LINEAR lift = "
    f"{df.linear_lift.mean():.3f}x"
)

print(
    "Anchored AFFINE rho  = "
    f"{df.affine_rho.mean():+.3f}"
)

print(
    "Anchored AFFINE lift = "
    f"{df.affine_lift.mean():.3f}x"
)

print(
    "DIRECT rho           = "
    f"{df.direct_rho.mean():+.3f}"
)

print(
    "DIRECT lift          = "
    f"{df.direct_lift.mean():.3f}x"
)


print("\n" + "=" * 72)
print("PROBE-LEVEL BOOTSTRAP 95% CI")
print("=" * 72)

print(
    "AFFINE rho CI:",
    np.round(
        ci(boot_macro_affine_rho),
        3
    )
)

print(
    "LINEAR rho CI:",
    np.round(
        ci(boot_macro_linear_rho),
        3
    )
)

print(
    "DIRECT rho CI:",
    np.round(
        ci(boot_macro_direct_rho),
        3
    )
)

print(
    "AFFINE lift CI:",
    np.round(
        ci(boot_macro_affine_lift),
        3
    )
)

print(
    "AFFINE-DIRECT rho difference CI:",
    np.round(
        ci(boot_delta_affine_direct_rho),
        3
    )
)

print(
    "AFFINE-DIRECT lift difference CI:",
    np.round(
        ci(boot_delta_affine_direct_lift),
        3
    )
)


print("\n" + "=" * 72)
print("ACTIVE PROBE UTILITY")
print("=" * 72)

summary = (
    bdf
    .groupby(
        ["budget", "method"]
    )
    ["vs_random"]
    .mean()
    .unstack()
)

print(
    summary.to_string()
)


###############################################################################
# Compare with old failed factorized R1 if available.
###############################################################################

old_file = Path(
    "artifacts/router_r1/frozen_results.csv"
)

if old_file.exists():

    old = pd.read_csv(
        old_file
    )

    print("\n" + "=" * 72)
    print("OLD FACTORIZED R1 VS NEW ANCHORED MODEL")
    print("=" * 72)

    print(
        "Old factorized spectral rho = "
        f"{old.spectral_rho.mean():+.3f}"
    )

    print(
        "Old factorized lift         = "
        f"{old.spectral_lift.mean():.3f}x"
    )

    print(
        "New anchored affine rho     = "
        f"{df.affine_rho.mean():+.3f}"
    )

    print(
        "New anchored affine lift    = "
        f"{df.affine_lift.mean():.3f}x"
    )


print("\nSaved:")
print(
    OUT / "anchored_results.csv"
)

print(
    OUT / "anchored_budget_results.csv"
)

print()
print(
    "NOTE: Because the 157 probes were already inspected "
    "during R1 diagnosis, this is an exploratory follow-up."
)

print(
    "If anchored operators succeed, use a NEW untouched "
    "probe set for the final confirmatory experiment."
)
