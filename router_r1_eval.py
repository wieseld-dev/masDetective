#!/usr/bin/env python3

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import accuracy_score
from sentence_transformers import SentenceTransformer


CACHE = Path("artifacts/router_r1/api_runs.jsonl")
OUT = Path("artifacts/router_r1")
OUT.mkdir(parents=True, exist_ok=True)

DEV_N = 60
DIM = 16
NSEEDS = 20
STEPS = 3500
NULL_REPS = 500
BUDGETS = [1,2,3,5,10,20]

PAIRS = ["RA","RB","RC","AM","BM","CM"]
PATHS = ["RAM","RBM","RCM"]

CONTRASTS = [
    ("AB","RAM","RBM"),
    ("AC","RAM","RCM"),
    ("BC","RBM","RCM"),
]


def l2(x):
    return x / np.maximum(
        np.linalg.norm(x, axis=1, keepdims=True),
        1e-12
    )


def rho_safe(x,y):
    r = spearmanr(x,y).statistic
    return float(r) if np.isfinite(r) else 0.0


def top_lift(score, actual, frac=.20):
    k = max(1, math.ceil(frac*len(actual)))
    ix = np.argsort(score)[-k:]
    return float(
        actual[ix].mean() /
        max(actual.mean(),1e-12)
    )


class RouterOperators(nn.Module):

    def __init__(self,d):
        super().__init__()
        self.d=d

        self.params=nn.ParameterDict({
            x: nn.Parameter(
                torch.randn(d,d)*0.02
            )
            for x in "RABCM"
        })

    def W(self):
        I=torch.eye(
            self.d,
            device=next(self.parameters()).device
        )

        return {
            k:I+v
            for k,v in self.params.items()
        }

    def compose(self,comp):
        W=self.W()

        M=torch.eye(
            self.d,
            device=next(self.parameters()).device
        )

        for role in comp:
            M=W[role]@M

        return M

    def predict(self,q,comp):
        return q @ self.compose(comp).T


def fit_model(q,z,idx,device,seed):

    torch.manual_seed(seed)

    qt=torch.tensor(
        q[idx],
        dtype=torch.float32,
        device=device
    )

    targets={
        c:torch.tensor(
            z[c][idx],
            dtype=torch.float32,
            device=device
        )
        for c in PAIRS
    }

    model=RouterOperators(q.shape[1]).to(device)

    opt=torch.optim.AdamW(
        model.parameters(),
        lr=3e-3,
        weight_decay=1e-3
    )

    best=float("inf")
    state=None

    for step in range(STEPS):

        opt.zero_grad(set_to_none=True)

        loss=0

        for c in PAIRS:
            pred=model.predict(qt,c)
            loss += ((pred-targets[c])**2).mean()

        loss=loss/len(PAIRS)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),5.0
        )

        opt.step()

        lv=float(loss.detach().cpu())

        if lv<best:
            best=lv
            state={
                k:v.detach().cpu().clone()
                for k,v in model.state_dict().items()
            }

    model.load_state_dict(state)

    return model,best


def path_matrix(W,path):
    M=np.eye(next(iter(W.values())).shape[0])

    for r in path:
        M=W[r]@M

    return M


def random_like(K,rng):
    R=rng.normal(size=K.shape)
    R *= (
        np.linalg.norm(K) /
        max(np.linalg.norm(R),1e-12)
    )
    return R


# --------------------------------------------------
# Load data
# --------------------------------------------------

rows=[]

for line in CACHE.open(encoding="utf-8"):
    if line.strip():
        rows.append(json.loads(line))

dedup={int(r["id"]):r for r in rows}
rows=[dedup[k] for k in sorted(dedup)]

print("TOTAL:",len(rows))

if len(rows)<217:
    raise RuntimeError(
        f"Need 217 rows, found {len(rows)}"
    )

rows=rows[:217]

DEV=np.arange(DEV_N)
TEST=np.arange(DEV_N,217)

tasks=[r["task"] for r in rows]

device="cuda" if torch.cuda.is_available() else "cpu"
print("DEVICE:",device)
print("DEV:",len(DEV))
print("LOCKED TEST:",len(TEST))

enc=SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    device=device
)

print("Encoding probes...")

q0=enc.encode(
    tasks,
    normalize_embeddings=True,
    convert_to_numpy=True,
    batch_size=64
)

z0={}

for c in PAIRS+PATHS:

    print("Encoding",c)

    z0[c]=enc.encode(
        [r["outputs"][c] for r in rows],
        normalize_embeddings=True,
        convert_to_numpy=True,
        batch_size=64
    )


results=[]
budgets=[]


for seed in range(1,NSEEDS+1):

    np.random.seed(seed)
    rng=np.random.default_rng(10000+seed)

    # CRITICAL:
    # representation is fit using DEV pair systems only.
    # RAM/RBM/RCM never enter PCA.
    pdata=[q0[DEV]]

    for c in PAIRS:
        pdata.append(z0[c][DEV])

    pca=PCA(
        n_components=DIM,
        random_state=seed
    )

    pca.fit(np.vstack(pdata))

    q=l2(pca.transform(q0))

    z={
        c:l2(pca.transform(z0[c]))
        for c in PAIRS+PATHS
    }

    model,loss=fit_model(
        q,z,DEV,device,seed
    )

    W={
        k:v.detach().cpu().numpy()
        for k,v in model.W().items()
    }

    F={
        p:path_matrix(W,p)
        for p in PATHS
    }

    seed_rhos=[]

    for name,p1,p2 in CONTRASTS:

        # ------------------------------------------
        # Spectral PATH contrast
        # ------------------------------------------

        K=F[p1]-F[p2]

        spectral=np.sum(
            (q[TEST]@K.T)**2,
            axis=1
        )

        actual=np.sum(
            (z[p1][TEST]-z[p2][TEST])**2,
            axis=1
        )

        sr=rho_safe(spectral,actual)
        sl=top_lift(spectral,actual)

        seed_rhos.append(sr)

        # ------------------------------------------
        # DIRECT baseline:
        # gets direct DEV triple-difference labels.
        # Stronger supervision than our operators.
        # ------------------------------------------

        reg=Ridge(
            alpha=3.0,
            fit_intercept=False
        )

        reg.fit(
            q[DEV],
            z[p1][DEV]-z[p2][DEV]
        )

        dd=reg.predict(q[TEST])

        direct=np.sum(dd**2,axis=1)

        dr=rho_safe(direct,actual)
        dl=top_lift(direct,actual)

        # ------------------------------------------
        # NULLS
        # ------------------------------------------

        shuffle_r=[]
        random_r=[]

        for _ in range(NULL_REPS):

            shuffle_r.append(
                rho_safe(
                    spectral,
                    rng.permutation(actual)
                )
            )

            RK=random_like(K,rng)

            rs=np.sum(
                (q[TEST]@RK.T)**2,
                axis=1
            )

            random_r.append(
                rho_safe(rs,actual)
            )

        results.append({
            "seed":seed,
            "contrast":name,
            "spectral_rho":sr,
            "spectral_lift":sl,
            "direct_rho":dr,
            "direct_lift":dl,
            "shuffle_rho":np.mean(shuffle_r),
            "randomK_rho":np.mean(random_r),
            "p_shuffle":(
                1+np.sum(np.array(shuffle_r)>=sr)
            )/(NULL_REPS+1),
            "p_randomK":(
                1+np.sum(np.array(random_r)>=sr)
            )/(NULL_REPS+1),
            "train_loss":loss
        })

        # ------------------------------------------
        # Active selection utility
        # ------------------------------------------

        spec_order=np.argsort(spectral)[::-1]
        direct_order=np.argsort(direct)[::-1]

        for b in BUDGETS:

            si=spec_order[:b]
            di=direct_order[:b]

            random_gain=[]

            for _ in range(500):

                ri=rng.choice(
                    len(TEST),
                    size=b,
                    replace=False
                )

                random_gain.append(
                    actual[ri].mean()
                )

            rg=float(np.mean(random_gain))

            budgets.append({
                "seed":seed,
                "contrast":name,
                "budget":b,
                "spectral_vs_random":
                    actual[si].mean()/max(rg,1e-12),
                "direct_vs_random":
                    actual[di].mean()/max(rg,1e-12)
            })

    print(
        f"seed={seed:02d} "
        f"macro_rho={np.mean(seed_rhos):+.3f}"
    )


df=pd.DataFrame(results)
bf=pd.DataFrame(budgets)

df.to_csv(
    OUT/"frozen_results.csv",
    index=False
)

bf.to_csv(
    OUT/"budget_results.csv",
    index=False
)


print("\n"+"="*65)
print("R1 HIDDEN-PATH FROZEN VALIDATION")
print("="*65)

for c in ["AB","AC","BC"]:

    d=df[df.contrast==c]

    print(f"\n[{c}]")

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
        "p<.05 shuffle = "
        f"{(d.p_shuffle<.05).mean()*100:.1f}%"
    )

    print(
        "p<.05 randomK = "
        f"{(d.p_randomK<.05).mean()*100:.1f}%"
    )


macro=df.groupby("seed").mean(
    numeric_only=True
)

print("\n[MACRO]")

print(
    f"Spectral rho  = "
    f"{macro.spectral_rho.mean():+.3f}"
)

print(
    f"Spectral lift = "
    f"{macro.spectral_lift.mean():.3f}x"
)

print(
    f"Direct rho    = "
    f"{macro.direct_rho.mean():+.3f}"
)

print(
    f"Direct lift   = "
    f"{macro.direct_lift.mean():.3f}x"
)

print(
    f"Shuffle rho   = "
    f"{macro.shuffle_rho.mean():+.3f}"
)

print(
    f"Random-K rho  = "
    f"{macro.randomK_rho.mean():+.3f}"
)


print("\n"+"="*65)
print("ACTIVE PATH-PROBE UTILITY")
print("="*65)

summary=(
    bf.groupby("budget")
      .agg(
        spectral_vs_random=(
            "spectral_vs_random","mean"
        ),
        direct_vs_random=(
            "direct_vs_random","mean"
        )
      )
)

print(summary.to_string())


# ============================================================
# SEMANTIC LEAKAGE CONTROL
#
# Can final answer alone reveal which specialist path ran?
# Train classifier on DEV path outputs; test on locked TEST.
# ============================================================

X_train=[]
y_train=[]

X_test=[]
y_test=[]

for label,path in enumerate(PATHS):

    X_train.append(z0[path][DEV])
    y_train.extend([label]*len(DEV))

    X_test.append(z0[path][TEST])
    y_test.extend([label]*len(TEST))

X_train=np.vstack(X_train)
X_test=np.vstack(X_test)

y_train=np.array(y_train)
y_test=np.array(y_test)

clf=LogisticRegression(
    max_iter=3000,
    C=1.0
)

clf.fit(X_train,y_train)

pred=clf.predict(X_test)

sem_acc=accuracy_score(y_test,pred)

print("\n"+"="*65)
print("SEMANTIC LEAKAGE CONTROL")
print("="*65)

print(
    f"3-way final-output route classification "
    f"accuracy = {sem_acc*100:.2f}%"
)

print(
    "Chance = 33.33%"
)

print("\nSaved:")
print(OUT/"frozen_results.csv")
print(OUT/"budget_results.csv")
