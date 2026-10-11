#!/usr/bin/env python3

import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import accuracy_score
from sentence_transformers import SentenceTransformer


CACHE = Path("artifacts/router_r1/api_runs.jsonl")

DEV_N = 60
DIM = 16

PAIRS = ["RA","RB","RC","AM","BM","CM"]
PATHS = ["RAM","RBM","RCM"]

ROUTES = [
    ("A","RA","RAM"),
    ("B","RB","RBM"),
    ("C","RC","RCM"),
]

CONTRASTS = [
    ("AB","RA","RB","RAM","RBM"),
    ("AC","RA","RC","RAM","RCM"),
    ("BC","RB","RC","RBM","RCM"),
]


def l2(x):
    return x / np.maximum(
        np.linalg.norm(x, axis=1, keepdims=True),
        1e-12
    )


def rho(x,y):
    r = spearmanr(x,y).statistic
    return float(r) if np.isfinite(r) else 0.0


def lift(score, actual, frac=.20):
    k = max(1, math.ceil(frac * len(actual)))
    ix = np.argsort(score)[-k:]
    return float(
        actual[ix].mean() /
        max(actual.mean(), 1e-12)
    )


# ----------------------------------------------------------
# Load
# ----------------------------------------------------------

rows = []

for line in CACHE.open(encoding="utf-8"):
    if line.strip():
        rows.append(json.loads(line))

rows = {
    int(r["id"]): r
    for r in rows
}

rows = [rows[k] for k in sorted(rows)]

assert len(rows) >= 217

rows = rows[:217]

DEV = np.arange(DEV_N)
TEST = np.arange(DEV_N, 217)

tasks = [r["task"] for r in rows]

print("TOTAL:", len(rows))
print("DEV:", len(DEV))
print("TEST:", len(TEST))


# ----------------------------------------------------------
# Embeddings
# ----------------------------------------------------------

enc = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    device="cpu"
)

print("Encoding probes...")

q0 = enc.encode(
    tasks,
    normalize_embeddings=True,
    convert_to_numpy=True,
    batch_size=64
)

z0 = {}

for c in PAIRS + PATHS:

    print("Encoding", c)

    z0[c] = enc.encode(
        [r["outputs"][c] for r in rows],
        normalize_embeddings=True,
        convert_to_numpy=True,
        batch_size=64
    )


# ----------------------------------------------------------
# Same PCA definition as R1
# Fit only on DEV + pair systems.
# ----------------------------------------------------------

pdata = [q0[DEV]]

for c in PAIRS:
    pdata.append(z0[c][DEV])

pca = PCA(
    n_components=DIM,
    random_state=0
)

pca.fit(np.vstack(pdata))

q = l2(pca.transform(q0))

z = {
    c: l2(pca.transform(z0[c]))
    for c in PAIRS + PATHS
}


print("\n" + "="*72)
print("1. DOES SPECIALIST SEPARATION SURVIVE THE MERGER?")
print("="*72)


for name, pre1, pre2, post1, post2 in CONTRASTS:

    dpre = np.sum(
        (z[pre1][TEST] - z[pre2][TEST])**2,
        axis=1
    )

    dpost = np.sum(
        (z[post1][TEST] - z[post2][TEST])**2,
        axis=1
    )

    print(f"\n[{name}]")

    print(
        "rho(pre separation, post separation) = "
        f"{rho(dpre,dpost):+.3f}"
    )

    print(
        "top-20% PRE probes -> POST lift        = "
        f"{lift(dpre,dpost):.3f}x"
    )

    print(
        "mean pre separation                   = "
        f"{dpre.mean():.4f}"
    )

    print(
        "mean post separation                  = "
        f"{dpost.mean():.4f}"
    )


# ----------------------------------------------------------
# 2. Can q predict PRE specialist separation?
#
# Compare with POST, which failed in R1.
# ----------------------------------------------------------

print("\n" + "="*72)
print("2. DIRECT PREDICTABILITY: PRE-MERGER VS POST-MERGER")
print("="*72)


for name, pre1, pre2, post1, post2 in CONTRASTS:

    pre_delta_train = (
        z[pre1][DEV] -
        z[pre2][DEV]
    )

    post_delta_train = (
        z[post1][DEV] -
        z[post2][DEV]
    )

    rpre = Ridge(
        alpha=3.0,
        fit_intercept=False
    )

    rpost = Ridge(
        alpha=3.0,
        fit_intercept=False
    )

    rpre.fit(
        q[DEV],
        pre_delta_train
    )

    rpost.fit(
        q[DEV],
        post_delta_train
    )

    pred_pre = rpre.predict(q[TEST])
    pred_post = rpost.predict(q[TEST])

    score_pre = np.sum(pred_pre**2, axis=1)
    score_post = np.sum(pred_post**2, axis=1)

    actual_pre = np.sum(
        (z[pre1][TEST] - z[pre2][TEST])**2,
        axis=1
    )

    actual_post = np.sum(
        (z[post1][TEST] - z[post2][TEST])**2,
        axis=1
    )

    print(f"\n[{name}]")

    print(
        f"DIRECT PRE  rho = "
        f"{rho(score_pre,actual_pre):+.3f}"
    )

    print(
        f"DIRECT POST rho = "
        f"{rho(score_post,actual_post):+.3f}"
    )


# ----------------------------------------------------------
# 3. Learn the merger DIRECTLY:
#
# RA -> RAM
# RB -> RBM
# RC -> RCM
#
# This bypasses R/A/B/C factorization.
# ----------------------------------------------------------

print("\n" + "="*72)
print("3. CAN ONE SHARED LINEAR MERGER MAP PRE -> POST?")
print("="*72)


Xtr = []
Ytr = []

for _, pre, post in ROUTES:

    Xtr.append(z[pre][DEV])
    Ytr.append(z[post][DEV])

Xtr = np.vstack(Xtr)
Ytr = np.vstack(Ytr)

shared_M = Ridge(
    alpha=1.0,
    fit_intercept=False
)

shared_M.fit(Xtr, Ytr)


shared_pred = {}
shared_mse = []


for label, pre, post in ROUTES:

    pred = shared_M.predict(z[pre][TEST])

    shared_pred[label] = pred

    mse = np.mean(
        (pred - z[post][TEST])**2
    )

    shared_mse.append(mse)

    print(
        f"{label}: shared-M output MSE = "
        f"{mse:.6f}"
    )


print(
    "\nMacro shared-M MSE = "
    f"{np.mean(shared_mse):.6f}"
)


print("\nShared-M separation ranking:")

for name, pre1, pre2, post1, post2 in CONTRASTS:

    a = name[0]
    b = name[1]

    predicted_sep = np.sum(
        (
            shared_pred[a] -
            shared_pred[b]
        )**2,
        axis=1
    )

    actual_sep = np.sum(
        (
            z[post1][TEST] -
            z[post2][TEST]
        )**2,
        axis=1
    )

    print(
        f"{name}: rho={rho(predicted_sep,actual_sep):+.3f} "
        f"lift={lift(predicted_sep,actual_sep):.3f}x"
    )


# ----------------------------------------------------------
# 4. Shared merger vs path-specific merger
# Diagnostic only.
# ----------------------------------------------------------

print("\n" + "="*72)
print("4. SHARED MERGER VS PATH-SPECIFIC MERGERS")
print("="*72)


sep_mse = []

for label, pre, post in ROUTES:

    reg = Ridge(
        alpha=1.0,
        fit_intercept=False
    )

    reg.fit(
        z[pre][DEV],
        z[post][DEV]
    )

    pred = reg.predict(
        z[pre][TEST]
    )

    mse = np.mean(
        (pred-z[post][TEST])**2
    )

    sep_mse.append(mse)

    print(
        f"{label}: separate-M MSE = "
        f"{mse:.6f}"
    )


print(
    "\nMacro separate-M MSE = "
    f"{np.mean(sep_mse):.6f}"
)

print(
    "Shared / Separate MSE ratio = "
    f"{np.mean(shared_mse)/max(np.mean(sep_mse),1e-12):.3f}x"
)


# ----------------------------------------------------------
# 5. Does merger erase route identity?
# ----------------------------------------------------------

print("\n" + "="*72)
print("5. ROUTE IDENTITY BEFORE VS AFTER MERGER")
print("="*72)


def route_accuracy(comps):

    Xtr = []
    Xte = []
    ytr = []
    yte = []

    for label, c in enumerate(comps):

        Xtr.append(z0[c][DEV])
        Xte.append(z0[c][TEST])

        ytr.extend(
            [label] * len(DEV)
        )

        yte.extend(
            [label] * len(TEST)
        )

    Xtr = np.vstack(Xtr)
    Xte = np.vstack(Xte)

    ytr = np.array(ytr)
    yte = np.array(yte)

    clf = LogisticRegression(
        max_iter=3000,
        C=1.0
    )

    clf.fit(Xtr, ytr)

    return accuracy_score(
        yte,
        clf.predict(Xte)
    )


pre_acc = route_accuracy(
    ["RA","RB","RC"]
)

post_acc = route_accuracy(
    ["RAM","RBM","RCM"]
)


print(
    f"PRE-merger  route classification = "
    f"{pre_acc*100:.2f}%"
)

print(
    f"POST-merger route classification = "
    f"{post_acc*100:.2f}%"
)

print("Chance = 33.33%")


print("\n" + "="*72)
print("DIAGNOSTIC COMPLETE")
print("="*72)
