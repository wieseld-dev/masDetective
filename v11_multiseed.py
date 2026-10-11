import json, math
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer

import blackbox_commutator_poc as poc

CACHE = Path("artifacts/commutator_v11_poc/api_runs.jsonl")
NSEEDS = 20
DIM = 16
TRAIN_FRAC = 0.70
RIDGE_ALPHA = 3.0

rows = []
with CACHE.open() as f:
    for line in f:
        if line.strip():
            rows.append(json.loads(line))
rows.sort(key=lambda r: int(r["id"]))

print("Cached V11 probes:", len(rows))

tasks = [r["task"] for r in rows]
out_ab = [r["AB"]["final"] for r in rows]
out_ba = [r["BA"]["final"] for r in rows]

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

enc = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    device=device
)

print("Encoding once...")
q_full = enc.encode(tasks, batch_size=64, convert_to_numpy=True,
                    normalize_embeddings=True)
zab_full = enc.encode(out_ab, batch_size=64, convert_to_numpy=True,
                      normalize_embeddings=True)
zba_full = enc.encode(out_ba, batch_size=64, convert_to_numpy=True,
                      normalize_embeddings=True)


def l2(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def metrics(K, qte, zabte, zbate):
    predicted = np.sum((qte @ K.T) ** 2, axis=1)
    actual = np.sum((zabte - zbate) ** 2, axis=1)

    rho = float(spearmanr(predicted, actual).statistic)

    k = max(1, math.ceil(0.20 * len(actual)))
    top = np.argsort(predicted)[-k:]

    lift = float(
        actual[top].mean() /
        max(actual.mean(), 1e-12)
    )
    return rho, lift


results = []

for seed in range(1, NSEEDS + 1):
    np.random.seed(seed)
    torch.manual_seed(seed)

    idx = np.arange(len(rows))
    tr, te = train_test_split(
        idx,
        train_size=TRAIN_FRAC,
        random_state=seed,
        shuffle=True
    )

    pca_train = np.vstack([
        q_full[tr],
        zab_full[tr],
        zba_full[tr]
    ])

    d = min(DIM, pca_train.shape[0] - 1, pca_train.shape[1])
    pca = PCA(n_components=d, random_state=seed)
    pca.fit(pca_train)

    q = l2(pca.transform(q_full))
    zab = l2(pca.transform(zab_full))
    zba = l2(pca.transform(zba_full))

    qtr, qte = q[tr], q[te]
    zabtr, zabte = zab[tr], zab[te]
    zbatr, zbate = zba[tr], zba[te]

    # Direct spectral
    rab = Ridge(alpha=RIDGE_ALPHA, fit_intercept=False).fit(qtr, zabtr)
    rba = Ridge(alpha=RIDGE_ALPHA, fit_intercept=False).fit(qtr, zbatr)

    K_direct = rab.coef_ - rba.coef_
    drho, dlift = metrics(K_direct, qte, zabte, zbate)

    # Factorized commutator
    model, WA, WB, _ = poc.fit_factorized(
        qtr, zabtr, zbatr,
        device=device,
        steps=2500,
        lr=3e-3,
        weight_decay=1e-3,
    )

    K_comm = WB @ WA - WA @ WB
    crho, clift = metrics(K_comm, qte, zabte, zbate)

    results.append((drho, dlift, crho, clift))

    print(
        f"seed={seed:02d} | "
        f"direct rho={drho:+.3f} lift={dlift:.3f}x | "
        f"comm rho={crho:+.3f} lift={clift:.3f}x"
    )

R = np.array(results)

print("\n================ MULTI-SEED SUMMARY ================")
print(f"Seeds: {NSEEDS}")
print()
print(
    f"DIRECT      rho  = {R[:,0].mean():+.3f} ± {R[:,0].std(ddof=1):.3f}"
)
print(
    f"DIRECT      lift = {R[:,1].mean():.3f} ± {R[:,1].std(ddof=1):.3f}x"
)
print(
    f"DIRECT      positive rho: {(R[:,0] > 0).mean()*100:.1f}%"
)
print(
    f"DIRECT      lift > 1:     {(R[:,1] > 1).mean()*100:.1f}%"
)

print()

print(
    f"COMMUTATOR  rho  = {R[:,2].mean():+.3f} ± {R[:,2].std(ddof=1):.3f}"
)
print(
    f"COMMUTATOR  lift = {R[:,3].mean():.3f} ± {R[:,3].std(ddof=1):.3f}x"
)
print(
    f"COMMUTATOR  positive rho: {(R[:,2] > 0).mean()*100:.1f}%"
)
print(
    f"COMMUTATOR  lift > 1:     {(R[:,3] > 1).mean()*100:.1f}%"
)

print()
print(
    f"COMM - DIRECT mean rho difference: "
    f"{(R[:,2]-R[:,0]).mean():+.3f}"
)
print(
    f"COMM - DIRECT mean lift difference: "
    f"{(R[:,3]-R[:,1]).mean():+.3f}x"
)
