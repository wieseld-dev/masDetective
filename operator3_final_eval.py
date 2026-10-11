#!/usr/bin/env python3

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer

import operator3_poc as op


CACHE = Path("artifacts/operator3_poc/api_runs.jsonl")
OUT = Path("artifacts/operator3_final")
OUT.mkdir(parents=True, exist_ok=True)

DEV_N = 60
DIM = 16
NSEEDS = 20
NULL_REPS = 500
BUDGETS = [1, 2, 3, 5, 10, 20]

TESTS = [
    ("AB", "ABC", "BAC", "C"),
    ("AC", "ACB", "CAB", "B"),
    ("BC", "BCA", "CBA", "A"),
]


def l2(x):
    return x / np.maximum(
        np.linalg.norm(x, axis=1, keepdims=True),
        1e-12
    )


def rho_safe(x, y):
    r = spearmanr(x, y).statistic
    return float(r) if np.isfinite(r) else 0.0


def lift(score, actual, frac=.20):
    k = max(1, math.ceil(frac * len(actual)))
    ids = np.argsort(score)[-k:]
    return float(
        actual[ids].mean() /
        max(actual.mean(), 1e-12)
    )


def get_K(model, pair):
    W = {
        k: v.detach().cpu().numpy()
        for k, v in model.W().items()
    }

    if pair == "AB":
        return W["C"] @ (
            W["B"] @ W["A"] -
            W["A"] @ W["B"]
        )

    if pair == "AC":
        return W["B"] @ (
            W["C"] @ W["A"] -
            W["A"] @ W["C"]
        )

    return W["A"] @ (
        W["C"] @ W["B"] -
        W["B"] @ W["C"]
    )


def score_K(K, q):
    return np.sum((q @ K.T) ** 2, axis=1)


def random_K_like(K, rng):
    R = rng.normal(size=K.shape)
    R *= np.linalg.norm(K) / max(
        np.linalg.norm(R), 1e-12
    )
    return R


def fit_direct_difference(q, z, train_idx, c1, c2):
    """
    Generic direct baseline:
       q -> z_c1 - z_c2

    This baseline is deliberately allowed to see triple deltas
    on DEV. It therefore has stronger supervision than the
    pair-only operator model.
    """
    from sklearn.linear_model import Ridge

    target = z[c1][train_idx] - z[c2][train_idx]

    reg = Ridge(
        alpha=3.0,
        fit_intercept=False
    )
    reg.fit(q[train_idx], target)

    return reg


def identification_accuracy(
    selected,
    z,
    c1,
    c2,
    dev_idx,
    test_idx,
):
    """
    Prototype classifier.

    Build one output prototype per candidate chain from DEV.
    On selected TEST probes, classify which hidden chain
    generated the observed outputs.

    Each test probe is treated as one independent black-box
    identification episode; budget means number of selected
    probe observations aggregated before decision.
    """

    proto1 = z[c1][dev_idx].mean(axis=0)
    proto2 = z[c2][dev_idx].mean(axis=0)

    # Evaluate both possible hidden systems.
    correct = 0
    total = 0

    for truth in [c1, c2]:

        # selected is a set of TEST indices.
        obs = z[truth][selected]

        d1 = np.mean(
            np.sum((obs - proto1) ** 2, axis=1)
        )
        d2 = np.mean(
            np.sum((obs - proto2) ** 2, axis=1)
        )

        guess = c1 if d1 < d2 else c2

        correct += int(guess == truth)
        total += 1

    return correct / total


# ============================================================
# DATA
# ============================================================

rows = []

with CACHE.open(encoding="utf-8") as f:
    for line in f:
        if line.strip():
            rows.append(json.loads(line))

rows.sort(key=lambda r: int(r["id"]))

# Defensive de-dup by id.
tmp = {}
for r in rows:
    tmp[int(r["id"])] = r
rows = [tmp[k] for k in sorted(tmp)]

print("TOTAL CACHED:", len(rows))

if len(rows) < 217:
    raise RuntimeError(
        f"Need 217 probes; only {len(rows)} cached."
    )

rows = rows[:217]

tasks = [r["task"] for r in rows]

all_comps = (
    ["A", "B", "C"]
    + op.PAIRS
    + op.TRIPLES
)

device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("DEVICE:", device)

encoder = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    device=device
)

print("Encoding inputs...")

q0 = encoder.encode(
    tasks,
    normalize_embeddings=True,
    convert_to_numpy=True,
    batch_size=64,
)

z0 = {}

for c in all_comps:
    print("Encoding", c)

    z0[c] = encoder.encode(
        [r["outputs"][c] for r in rows],
        normalize_embeddings=True,
        convert_to_numpy=True,
        batch_size=64,
    )


DEV = np.arange(0, DEV_N)
TEST = np.arange(DEV_N, len(rows))

print()
print("DEV:", len(DEV))
print("LOCKED TEST:", len(TEST))


all_results = []
budget_results = []


# ============================================================
# MULTI-INITIALIZATION:
#
# Same frozen 60/157 split every time.
# Seeds change ONLY neural initialization.
# This is intentionally different from the old split test.
# ============================================================

for seed in range(1, NSEEDS + 1):

    np.random.seed(seed)
    torch.manual_seed(seed)

    # PCA fit on DEV pair information ONLY.
    pdata = [q0[DEV]]

    for c in op.PAIRS:
        pdata.append(z0[c][DEV])

    pca = PCA(
        n_components=DIM,
        random_state=seed
    )

    pca.fit(np.vstack(pdata))

    q = l2(pca.transform(q0))

    z = {
        c: l2(pca.transform(z0[c]))
        for c in all_comps
    }

    # Pair-only operator learning.
    model, train_loss = op.fit_model(
        q,
        z,
        DEV,
        op.PAIRS,
        device=device,
        steps=3000
    )

    rng = np.random.default_rng(
        100000 + seed
    )

    seed_rhos = []

    for pair, c1, c2, terminal in TESTS:

        K = get_K(model, pair)

        spectral = score_K(
            K,
            q[TEST]
        )

        actual = np.sum(
            (
                z[c1][TEST]
                - z[c2][TEST]
            ) ** 2,
            axis=1
        )

        rho = rho_safe(
            spectral,
            actual
        )

        lf = lift(
            spectral,
            actual
        )

        # ----------------------------------------
        # Random-K null
        # ----------------------------------------

        random_rhos = []
        random_lifts = []

        for _ in range(NULL_REPS):

            RK = random_K_like(
                K,
                rng
            )

            rs = score_K(
                RK,
                q[TEST]
            )

            random_rhos.append(
                rho_safe(rs, actual)
            )

            random_lifts.append(
                lift(rs, actual)
            )

        # ----------------------------------------
        # Probe-output shuffle null
        # ----------------------------------------

        shuffle_rhos = []
        shuffle_lifts = []

        for _ in range(NULL_REPS):

            ya = rng.permutation(actual)

            shuffle_rhos.append(
                rho_safe(
                    spectral,
                    ya
                )
            )

            shuffle_lifts.append(
                lift(
                    spectral,
                    ya
                )
            )

        # ----------------------------------------
        # Direct supervised difference baseline
        # ----------------------------------------

        direct = fit_direct_difference(
            q,
            z,
            DEV,
            c1,
            c2
        )

        direct_delta = direct.predict(
            q[TEST]
        )

        direct_score = np.sum(
            direct_delta ** 2,
            axis=1
        )

        direct_rho = rho_safe(
            direct_score,
            actual
        )

        direct_lift = lift(
            direct_score,
            actual
        )

        seed_rhos.append(rho)

        all_results.append({
            "seed": seed,
            "pair": pair,

            "spectral_rho": rho,
            "spectral_lift": lf,

            "direct_rho": direct_rho,
            "direct_lift": direct_lift,

            "shuffle_rho":
                np.mean(shuffle_rhos),

            "shuffle_lift":
                np.mean(shuffle_lifts),

            "randomK_rho":
                np.mean(random_rhos),

            "randomK_lift":
                np.mean(random_lifts),

            "p_vs_shuffle":
                (
                    1 + np.sum(
                        np.asarray(shuffle_rhos)
                        >= rho
                    )
                ) / (NULL_REPS + 1),

            "p_vs_randomK":
                (
                    1 + np.sum(
                        np.asarray(random_rhos)
                        >= rho
                    )
                ) / (NULL_REPS + 1),

            "K_norm":
                np.linalg.norm(K),

            "train_loss":
                train_loss,
        })

        # ========================================
        # ACTIVE QUERY BUDGET
        #
        # Compare:
        # 1. spectral top-k
        # 2. direct top-k
        # 3. random k
        #
        # Primary utility metric here:
        # mean structural separation exposed by
        # the selected queries.
        # ========================================

        order_spec = np.argsort(
            spectral
        )[::-1]

        order_direct = np.argsort(
            direct_score
        )[::-1]

        for budget in BUDGETS:

            b = min(
                budget,
                len(TEST)
            )

            s_local = order_spec[:b]
            d_local = order_direct[:b]

            s_global = TEST[s_local]
            d_global = TEST[d_local]

            spec_gain = float(
                actual[s_local].mean()
            )

            direct_gain = float(
                actual[d_local].mean()
            )

            # Robust random baseline:
            # average over 500 random selections.
            random_gains = []

            for _ in range(500):

                ridx = rng.choice(
                    len(TEST),
                    size=b,
                    replace=False
                )

                random_gains.append(
                    actual[ridx].mean()
                )

            rand_gain = float(
                np.mean(random_gains)
            )

            budget_results.append({
                "seed": seed,
                "pair": pair,
                "budget": budget,

                "spectral_gain":
                    spec_gain,

                "direct_gain":
                    direct_gain,

                "random_gain":
                    rand_gain,

                "spectral_vs_random":
                    spec_gain /
                    max(rand_gain, 1e-12),

                "direct_vs_random":
                    direct_gain /
                    max(rand_gain, 1e-12),
            })

    print(
        f"seed={seed:02d} | "
        f"macro spectral rho="
        f"{np.mean(seed_rhos):+.3f}"
    )


# ============================================================
# SUMMARIES
# ============================================================

df = pd.DataFrame(all_results)
bf = pd.DataFrame(budget_results)

df.to_csv(
    OUT / "frozen_external_results.csv",
    index=False
)

bf.to_csv(
    OUT / "active_budget_results.csv",
    index=False
)


print()
print("=" * 60)
print("FINAL FROZEN EXTERNAL VALIDATION")
print("=" * 60)

for pair in ["AB", "AC", "BC"]:

    d = df[df.pair == pair]

    print(f"\n[{pair}]")

    print(
        f"SPECTRAL rho  = "
        f"{d.spectral_rho.mean():+.3f} "
        f"± {d.spectral_rho.std(ddof=1):.3f}"
    )

    print(
        f"SPECTRAL lift = "
        f"{d.spectral_lift.mean():.3f}x"
    )

    print(
        f"DIRECT rho    = "
        f"{d.direct_rho.mean():+.3f}"
    )

    print(
        f"DIRECT lift   = "
        f"{d.direct_lift.mean():.3f}x"
    )

    print(
        f"SHUFFLE rho   = "
        f"{d.shuffle_rho.mean():+.3f}"
    )

    print(
        f"RANDOM-K rho  = "
        f"{d.randomK_rho.mean():+.3f}"
    )

    print(
        f"p<.05 vs shuffle: "
        f"{(d.p_vs_shuffle < .05).mean()*100:.1f}%"
    )

    print(
        f"p<.05 vs randomK: "
        f"{(d.p_vs_randomK < .05).mean()*100:.1f}%"
    )


macro = df.groupby("seed").mean(
    numeric_only=True
)

print()
print("[MACRO]")

print(
    "Spectral rho = "
    f"{macro.spectral_rho.mean():+.3f} "
    f"± {macro.spectral_rho.std(ddof=1):.3f}"
)

print(
    "Spectral lift = "
    f"{macro.spectral_lift.mean():.3f}x"
)

print(
    "Direct rho = "
    f"{macro.direct_rho.mean():+.3f}"
)

print(
    "Direct lift = "
    f"{macro.direct_lift.mean():.3f}x"
)

print(
    "Shuffle rho = "
    f"{macro.shuffle_rho.mean():+.3f}"
)

print(
    "Random-K rho = "
    f"{macro.randomK_rho.mean():+.3f}"
)


print()
print("=" * 60)
print("ACTIVE QUERY-BUDGET TEST")
print("=" * 60)

summary = (
    bf.groupby("budget")
    .agg(
        spectral_vs_random=(
            "spectral_vs_random",
            "mean"
        ),
        direct_vs_random=(
            "direct_vs_random",
            "mean"
        ),
    )
)

print(summary.to_string())

print()
print("Saved:")
print(
    OUT /
    "frozen_external_results.csv"
)
print(
    OUT /
    "active_budget_results.csv"
)
