import json, math
import sys
from pathlib import Path

REPO_ROOT = Path.home() / "mas_rl_detective"
sys.path.insert(0, str(REPO_ROOT))
from collections import Counter, defaultdict
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from adaptive_v11_active.belief import FrozenBeliefBundle, k_features
from adaptive_v11_active.io import read_jsonl
from adaptive_v11_active.schema import PublicBeliefState
from v11.role_universe import TRAIN_ROLES

ROOT = Path("artifacts/adaptive_v11_k_bootstrap_v1")
BELIEF = Path("artifacts/adaptive_v11_active_v1/frozen_belief.joblib")
KS = (3,4,5,6)
EPS = 1e-9

manifest = json.loads((ROOT/"manifest_privileged.json").read_text())
truth = {
    r["session_id"]: int(r["k"])
    for r in manifest["systems"]
    if r["split"] == "train"
}
bundle = FrozenBeliefBundle.load(BELIEF)

def role_probs(history):
    raw = bundle._raw_role_vector_with(bundle.role_model, history)
    p = bundle._apply_platt(bundle.role_platt, raw)
    return {r: float(p[i]) for i,r in enumerate(bundle.roles)}

def make_model(C):
    return Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(
            C=C,
            class_weight="balanced",
            solver="lbfgs",
            max_iter=5000,
        )),
    ])

def aligned(model, X):
    raw = model.predict_proba(X)
    cls = [int(x) for x in model.named_steps["clf"].classes_]
    out = np.full((len(X),4), EPS)
    where = {k:i for i,k in enumerate(KS)}
    for j,k in enumerate(cls):
        if k in where:
            out[:,where[k]] = raw[:,j]
    out /= out.sum(axis=1, keepdims=True)
    return out

def temp(p,t):
    z=np.log(np.clip(p,EPS,1.0))/t
    z-=z.max(axis=1,keepdims=True)
    q=np.exp(z)
    return q/q.sum(axis=1,keepdims=True)

def metrics(p,y):
    yi=np.array([KS.index(int(v)) for v in y])
    nll=float(np.mean(-np.log(np.clip(p[np.arange(len(y)),yi],EPS,1))))
    one=np.zeros_like(p)
    one[np.arange(len(y)),yi]=1
    brier=float(np.mean(np.sum((p-one)**2,axis=1)))
    pred=np.array([KS[i] for i in np.argmax(p,axis=1)])
    acc=float(np.mean(pred==y))
    mae=float(np.mean(np.abs(pred-y)))
    return nll,brier,acc,mae

def balanced_folds(ids, kvals, nfold=6, seed=917777):
    rng=np.random.default_rng(seed)
    byk=defaultdict(list)
    for sid,k in zip(ids,kvals):
        byk[int(k)].append(sid)
    buckets=[[] for _ in range(nfold)]
    for k in KS:
        a=list(byk[k]); rng.shuffle(a)
        for i,sid in enumerate(a):
            buckets[i % nfold].append(sid)
    return [set(x) for x in buckets]

CGRID=(0.003,0.01,0.03,0.1,0.3,1.0)
TGRID=(0.75,1.0,1.5,2.0,3.0,5.0)

print("UNIFORM: NLL=%.6f BRIER=0.750000 ACC=0.250" % math.log(4))
print()

for rnd in range(1,7):
    states = {
        x["session_id"]: PublicBeliefState.from_public_dict(x)
        for x in read_jsonl(ROOT/f"states_train_round_{rnd:02d}.jsonl")
    }

    rows=[]
    for sid,k in truth.items():
        st=states[sid]
        if len(st.history) < rnd:
            continue
        h=list(st.history[:rnd])
        rp=role_probs(h)
        rows.append((sid,k,k_features(h,rp,TRAIN_ROLES)))

    print("="*72)
    print("ROUND",rnd,"systems_with_exact_prefix",len(rows))

    if len(rows) < 40:
        print("TOO FEW; skip")
        continue

    ids=np.array([r[0] for r in rows],dtype=object)
    y=np.array([r[1] for r in rows],dtype=int)
    X=np.vstack([r[2] for r in rows])

    print("K counts:",dict(Counter(y.tolist())))

    folds=balanced_folds(ids,y)
    best=None

    for C in CGRID:
        oof=np.zeros((len(rows),4))
        covered=np.zeros(len(rows),dtype=bool)

        for va_ids in folds:
            va=np.array([sid in va_ids for sid in ids])
            tr=~va
            if len(set(y[tr].tolist())) < 4:
                continue
            m=make_model(C)
            m.fit(X[tr],y[tr])
            oof[va]=aligned(m,X[va])
            covered[va]=True

        if not covered.all():
            continue

        for T in TGRID:
            p=temp(oof,T)
            vals=metrics(p,y)
            cand=(vals[0],C,T,*vals[1:])
            if best is None or cand[0] < best[0]:
                best=cand

    if best is None:
        print("CV FAILED")
        continue

    nll,C,T,brier,acc,mae=best
    print(
        f"BEST: NLL={nll:.6f} Brier={brier:.6f} "
        f"Acc={acc:.3f} MAE={mae:.3f} C={C} T={T}"
    )
    print(
        f"DELTA vs uniform: NLL={nll-math.log(4):+.6f} "
        f"Brier={brier-0.75:+.6f}"
    )
