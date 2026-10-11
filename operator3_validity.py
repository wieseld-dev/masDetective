#!/usr/bin/env python3

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer

import operator3_poc as op

CACHE = Path("artifacts/operator3_poc/api_runs.jsonl")
OUT = Path("artifacts/operator3_poc/validity")
OUT.mkdir(parents=True, exist_ok=True)

NSEEDS = 20
DIM = 16
TRAIN_FRAC = 0.70
NULL_REPS = 200

# pair-name, triple1, triple2, final agent
TESTS = [
    ("AB", "ABC", "BAC", "C"),
    ("AC", "ACB", "CAB", "B"),
    ("BC", "BCA", "CBA", "A"),
]


def l2(x):
    return x / np.maximum(
        np.linalg.norm(x, axis=1, keepdims=True), 1e-12
    )


def rho_safe(x, y):
    r = spearmanr(x, y).statistic
    return float(r) if np.isfinite(r) else 0.0


def lift20(score, actual):
    k = max(1, math.ceil(.20 * len(actual)))
    top = np.argsort(score)[-k:]
    return float(actual[top].mean() / max(actual.mean(), 1e-12))


def get_Ks(model):
    W = {
        k: v.detach().cpu().numpy()
        for k, v in model.W().items()
    }

    # These exactly correspond to:
    # ABC-BAC = C(B A - A B)
    # ACB-CAB = B(C A - A C)
    # BCA-CBA = A(C B - B C)
    return {
        "AB": W["C"] @ (W["B"] @ W["A"] - W["A"] @ W["B"]),
        "AC": W["B"] @ (W["C"] @ W["A"] - W["A"] @ W["C"]),
        "BC": W["A"] @ (W["C"] @ W["B"] - W["B"] @ W["C"]),
    }


def spectral_score(K, q):
    # ||Kq||^2 = q^T K^T K q
    return np.sum((q @ K.T) ** 2, axis=1)


def random_matrix_like(K, rng):
    R = rng.normal(size=K.shape)
    rn = np.linalg.norm(R)
    kn = np.linalg.norm(K)
    if rn > 0:
        R *= kn / rn
    return R


# ---------------------------------------------------------
# Load cached black-box data
# ---------------------------------------------------------

rows = []
with CACHE.open(encoding="utf-8") as f:
    for line in f:
        if line.strip():
            rows.append(json.loads(line))

rows.sort(key=lambda r: int(r["id"]))

if len(rows) < 60:
    raise RuntimeError(f"Expected >=60 cached probes, got {len(rows)}")

# Keep exactly the original 60-probe experiment.
rows = rows[:60]

tasks = [r["task"] for r in rows]
all_comps = ["A", "B", "C"] + op.PAIRS + op.TRIPLES

print(f"Cached probes: {len(rows)}")

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

encoder = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    device=device
)

print("Encoding probes...")
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


records = []
null_records = []

for seed in range(1, NSEEDS + 1):
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = np.random.default_rng(10000 + seed)

    ids = np.arange(len(rows))
    tr, te = train_test_split(
        ids,
        train_size=TRAIN_FRAC,
        random_state=seed,
        shuffle=True,
    )

    # Critical: PCA also sees only TRAIN probe vectors
    # and TRAIN PAIR outputs. No triple leakage.
    pca_data = [q0[tr]]
    for c in op.PAIRS:
        pca_data.append(z0[c][tr])

    pca = PCA(n_components=DIM, random_state=seed)
    pca.fit(np.vstack(pca_data))

    q = l2(pca.transform(q0))
    z = {c: l2(pca.transform(z0[c])) for c in all_comps}

    # Pair-only training. Triple outputs are never supplied here.
    model, train_loss = op.fit_model(
        q, z, tr, op.PAIRS, device=device, steps=3000
    )

    Ks = get_Ks(model)

    pooled_pred = []
    pooled_actual = []

    seed_real_rhos = []

    for pair, c1, c2, terminal in TESTS:
        K = Ks[pair]

        pred = spectral_score(K, q[te])
        actual = np.sum((z[c1][te] - z[c2][te]) ** 2, axis=1)

        rho = rho_safe(pred, actual)
        lift = lift20(pred, actual)

        seed_real_rhos.append(rho)
        pooled_pred.extend(pred.tolist())
        pooled_actual.extend(actual.tolist())

        # -------------------------------------------------
        # NULL 1: shuffle probe -> observed delta mapping.
        # K and score stay fixed; only correspondence is broken.
        # -------------------------------------------------
        shuffle_rhos = []
        shuffle_lifts = []

        for rep in range(NULL_REPS):
            ya = rng.permutation(actual)
            shuffle_rhos.append(rho_safe(pred, ya))
            shuffle_lifts.append(lift20(pred, ya))

        # -------------------------------------------------
        # NULL 2: random matrix with same Frobenius norm as K.
        # -------------------------------------------------
        random_rhos = []
        random_lifts = []

        for rep in range(NULL_REPS):
            R = random_matrix_like(K, rng)
            rs = spectral_score(R, q[te])
            random_rhos.append(rho_safe(rs, actual))
            random_lifts.append(lift20(rs, actual))

        sh_r = np.asarray(shuffle_rhos)
        sh_l = np.asarray(shuffle_lifts)
        rd_r = np.asarray(random_rhos)
        rd_l = np.asarray(random_lifts)

        # Empirical one-sided p:
        # how often does null do at least as well as real?
        p_shuffle_rho = (1 + np.sum(sh_r >= rho)) / (NULL_REPS + 1)
        p_random_rho = (1 + np.sum(rd_r >= rho)) / (NULL_REPS + 1)

        p_shuffle_lift = (1 + np.sum(sh_l >= lift)) / (NULL_REPS + 1)
        p_random_lift = (1 + np.sum(rd_l >= lift)) / (NULL_REPS + 1)

        records.append({
            "seed": seed,
            "pair": pair,
            "terminal": terminal,
            "rho": rho,
            "lift": lift,
            "shuffle_rho_mean": sh_r.mean(),
            "random_rho_mean": rd_r.mean(),
            "shuffle_lift_mean": sh_l.mean(),
            "random_lift_mean": rd_l.mean(),
            "p_shuffle_rho": p_shuffle_rho,
            "p_random_rho": p_random_rho,
            "p_shuffle_lift": p_shuffle_lift,
            "p_random_lift": p_random_lift,
            "K_fro": np.linalg.norm(K),
            "train_loss": train_loss,
        })

        for rep in range(NULL_REPS):
            null_records.append({
                "seed": seed,
                "pair": pair,
                "rep": rep,
                "shuffle_rho": sh_r[rep],
                "shuffle_lift": sh_l[rep],
                "random_rho": rd_r[rep],
                "random_lift": rd_l[rep],
            })

    pooled_pred = np.asarray(pooled_pred)
    pooled_actual = np.asarray(pooled_actual)

    pooled_rho = rho_safe(pooled_pred, pooled_actual)
    pooled_lift = lift20(pooled_pred, pooled_actual)

    print(
        f"seed={seed:02d} | "
        f"AB={seed_real_rhos[0]:+.3f} "
        f"AC={seed_real_rhos[1]:+.3f} "
        f"BC={seed_real_rhos[2]:+.3f} | "
        f"POOLED rho={pooled_rho:+.3f} "
        f"lift={pooled_lift:.3f}x"
    )

df = pd.DataFrame(records)
ndf = pd.DataFrame(null_records)

df.to_csv(OUT / "per_seed_pair.csv", index=False)
ndf.to_csv(OUT / "null_distributions.csv", index=False)

print("\n====================================================")
print("SPECTRAL VALIDITY SUMMARY")
print("====================================================")

for pair in ["AB", "AC", "BC"]:
    d = df[df.pair == pair]

    print(f"\n[{pair}]")
    print(f"real rho      = {d.rho.mean():+.3f} ± {d.rho.std(ddof=1):.3f}")
    print(f"real lift     = {d.lift.mean():.3f} ± {d.lift.std(ddof=1):.3f}x")
    print(f"positive rho  = {(d.rho > 0).mean()*100:.1f}%")
    print(f"lift > 1      = {(d.lift > 1).mean()*100:.1f}%")
    print(f"shuffle rho   = {d.shuffle_rho_mean.mean():+.3f}")
    print(f"random-K rho  = {d.random_rho_mean.mean():+.3f}")
    print(f"shuffle lift  = {d.shuffle_lift_mean.mean():.3f}x")
    print(f"random-K lift = {d.random_lift_mean.mean():.3f}x")
    print(
        "beats shuffle rho (p<.05): "
        f"{(d.p_shuffle_rho < .05).mean()*100:.1f}% seeds"
    )
    print(
        "beats random-K rho (p<.05): "
        f"{(d.p_random_rho < .05).mean()*100:.1f}% seeds"
    )

# Macro treats each agent-pair equally.
macro = df.groupby("seed").agg(
    rho=("rho", "mean"),
    lift=("lift", "mean"),
    shuffle_rho=("shuffle_rho_mean", "mean"),
    random_rho=("random_rho_mean", "mean"),
    shuffle_lift=("shuffle_lift_mean", "mean"),
    random_lift=("random_lift_mean", "mean"),
)

macro.to_csv(OUT / "macro_seed.csv")

print("\n[MACRO ACROSS THREE COMMUTATORS]")
print(f"real rho      = {macro.rho.mean():+.3f} ± {macro.rho.std(ddof=1):.3f}")
print(f"real lift     = {macro.lift.mean():.3f} ± {macro.lift.std(ddof=1):.3f}x")
print(f"positive rho  = {(macro.rho > 0).mean()*100:.1f}%")
print(f"lift > 1      = {(macro.lift > 1).mean()*100:.1f}%")
print(f"shuffle rho   = {macro.shuffle_rho.mean():+.3f}")
print(f"random-K rho  = {macro.random_rho.mean():+.3f}")
print(f"shuffle lift  = {macro.shuffle_lift.mean():.3f}x")
print(f"random-K lift = {macro.random_lift.mean():.3f}x")

print("\nFiles:")
print(OUT / "per_seed_pair.csv")
print(OUT / "macro_seed.csv")
print(OUT / "null_distributions.csv")
