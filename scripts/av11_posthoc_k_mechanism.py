from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from adaptive_v11_active.belief import FrozenBeliefBundle, k_features
from adaptive_v11_active.schema import PublicBeliefState
from v11.role_universe import TRAIN_ROLES

ARMS = [
    "A_original_static",
    "B_original_adaptive",
    "C_grpo_static",
    "D_grpo_adaptive",
]
KS_DEFAULT = (3, 4, 5, 6)
EPS = 1e-9


def mean(xs):
    xs = list(xs)
    return float(statistics.mean(xs)) if xs else float("nan")


def read_jsonl(path: Path):
    out = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields = []
    seen = set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k); fields.append(k)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)


def state_map(path: Path):
    return {str(r["session_id"]): PublicBeliefState.from_public_dict(r) for r in read_jsonl(path)}


def entropy01(p: float) -> float:
    p = min(1-EPS, max(EPS, float(p)))
    return -(p*math.log(p) + (1-p)*math.log(1-p))


def role_only_features(state: PublicBeliefState, roles: list[str]) -> np.ndarray:
    p = np.asarray([float(state.role_probs[r]) for r in roles], dtype=np.float64)
    s = np.sort(p)[::-1]
    q25, med, q75 = np.quantile(p, [0.25, 0.5, 0.75])
    top8 = [float(s[i]) if i < len(s) else 0.0 for i in range(8)]
    gaps7 = [float(s[i]-s[i+1]) if i+1 < len(s) else 0.0 for i in range(7)]
    x = [
        math.log1p(len(state.history)), float(p.sum()), float(p.mean()), float(p.std()),
        float(p.min()), float(q25), float(med), float(q75), float(p.max()),
        float(np.mean([entropy01(x) for x in p])),
    ]
    x += [float(np.sum(p >= t)) for t in (0.20,0.30,0.40,0.50,0.60,0.70,0.80)]
    x += top8 + gaps7
    return np.asarray(x, dtype=np.float32)


def poisson_binomial(probs: Iterable[float]) -> np.ndarray:
    probs = np.clip(np.asarray(list(probs), dtype=np.float64), EPS, 1-EPS)
    dist = np.zeros(len(probs)+1, dtype=np.float64); dist[0] = 1.0
    for p in probs:
        nxt = np.zeros_like(dist)
        nxt[:-1] += dist[:-1]*(1-p)
        nxt[1:] += dist[:-1]*p
        dist = nxt
    z = dist.sum()
    return dist/z if z else dist


def role_cardinality_probs(state: PublicBeliefState, roles: list[str], ks: tuple[int,...]):
    full = poisson_binomial([state.role_probs[r] for r in roles])
    v = np.asarray([full[k] if k < len(full) else EPS for k in ks], dtype=np.float64)
    v = np.clip(v, EPS, None); v /= v.sum()
    return v


def vector_from_kdict(kp: dict, ks: tuple[int,...]):
    v = np.asarray([float(kp.get(k, kp.get(str(k), EPS))) for k in ks], dtype=np.float64)
    v = np.clip(v, EPS, None); v /= v.sum()
    return v


def pred_metrics(probs: np.ndarray, true_k: int, ks: tuple[int,...]):
    ki = {k:i for i,k in enumerate(ks)}
    y = np.zeros(len(ks), dtype=np.float64); y[ki[true_k]] = 1.0
    p = np.clip(np.asarray(probs, dtype=np.float64), EPS, 1.0); p /= p.sum()
    khat = int(ks[int(np.argmax(p))])
    ek = float(sum(k*p[i] for i,k in enumerate(ks)))
    return {
        "k_nll": float(-math.log(p[ki[true_k]])),
        "k_brier": float(np.sum((p-y)**2)),
        "k_acc": float(khat == true_k),
        "k_mae_argmax": float(abs(khat-true_k)),
        "k_mae_expected": float(abs(ek-true_k)),
        "k_hat": khat,
        "k_expected": ek,
        "p_true_k": float(p[ki[true_k]]),
        **{f"pK{k}": float(p[i]) for i,k in enumerate(ks)},
    }


def bootstrap_ci(ds, n=20000, seed=20260917):
    ds = list(map(float, ds))
    if not ds: return (float("nan"), float("nan"))
    rng = random.Random(seed); m = len(ds); vals=[]
    for _ in range(n): vals.append(mean(ds[rng.randrange(m)] for __ in range(m)))
    vals.sort()
    return vals[int(.025*n)], vals[min(n-1,int(.975*n))]


def signflip_p(ds, n=100000, seed=20260918):
    ds = list(map(float, ds))
    if not ds: return float("nan")
    obs = abs(mean(ds)); rng=random.Random(seed); ext=1
    for _ in range(n):
        x = mean(d if rng.random()<.5 else -d for d in ds)
        if abs(x) >= obs-1e-15: ext += 1
    return ext/(n+1)


def paired_system_delta(rows: list[dict], a: str, b: str, metric: str, lower=True):
    by = {a:defaultdict(list), b:defaultdict(list)}
    for r in rows:
        arm = r.get("arm")
        if arm in by:
            by[arm][r["session_id"]].append(float(r[metric]))
    ids = sorted(set(by[a]) & set(by[b]))
    ds=[]
    for sid in ids:
        ds.append(mean(by[b][sid]) - mean(by[a][sid]))
    lo,hi = bootstrap_ci(ds); p=signflip_p(ds)
    wins = sum(d < 0 if lower else d > 0 for d in ds)
    return {"n":len(ds),"delta":mean(ds),"ci_lo":lo,"ci_hi":hi,"p2":p,"wins":wins,"losses":len(ds)-wins}


def uniform_row(true_k: int, ks: tuple[int,...]):
    return pred_metrics(np.ones(len(ks))/len(ks), true_k, ks)


def balanced_system_folds(truth: dict[str,dict], ks: tuple[int,...]):
    byk = {k:sorted([sid for sid,t in truth.items() if int(t["k"])==k]) for k in ks}
    n = min(len(v) for v in byk.values())
    if n < 2:
        raise RuntimeError(f"not enough systems per K for grouped CV: { {k:len(v) for k,v in byk.items()} }")
    folds=[]
    # Each fold holds out one complete system of every K. All seeds/arms for that system stay together.
    for i in range(n):
        folds.append({byk[k][i] for k in ks})
    return folds


def remap_proba(model, X, ks: tuple[int,...]):
    raw = model.predict_proba(X)
    classes = [int(x) for x in model.named_steps["clf"].classes_]
    out = np.full((len(raw),len(ks)), EPS, dtype=np.float64)
    k2i={k:i for i,k in enumerate(ks)}
    for j,k in enumerate(classes):
        if k in k2i: out[:,k2i[k]]=raw[:,j]
    out /= out.sum(axis=1,keepdims=True)
    return out


def cv_k(rows: list[dict], truth: dict[str,dict], ks: tuple[int,...], feature_key: str, C: float):
    folds = balanced_system_folds(truth,ks)
    out=[]
    for arm in ARMS:
        for rnd in (1,2,3):
            rr=[r for r in rows if r["arm"]==arm and int(r["round"])==rnd]
            for fi,test_ids in enumerate(folds):
                tr=[r for r in rr if r["session_id"] not in test_ids]
                te=[r for r in rr if r["session_id"] in test_ids]
                Xtr=np.vstack([r[feature_key] for r in tr]); ytr=np.asarray([r["true_k"] for r in tr],dtype=int)
                Xte=np.vstack([r[feature_key] for r in te])
                model=Pipeline([
                    ("scale",StandardScaler()),
                    ("clf",LogisticRegression(C=C,class_weight="balanced",max_iter=5000,solver="lbfgs")),
                ])
                model.fit(Xtr,ytr)
                pp=remap_proba(model,Xte,ks)
                for r,p in zip(te,pp):
                    m=pred_metrics(p,int(r["true_k"]),ks)
                    out.append({
                        "feature_set":"full_public" if feature_key=="full_features" else "role_posterior_only",
                        "fold":fi,"arm":arm,"seed":r["seed"],"session_id":r["session_id"],
                        "round":rnd,"true_k":r["true_k"],**m,
                    })
    return out


def summarize_prediction_rows(rows: list[dict], title: str):
    print("\n"+"="*100); print(title); print("="*100)
    for rnd in (1,2,3):
        print(f"ROUND {rnd}")
        for arm in ARMS:
            z=[r for r in rows if r["arm"]==arm and int(r["round"])==rnd]
            if not z: continue
            print(f" {arm:22s} NLL={mean(r['k_nll'] for r in z):.4f} Brier={mean(r['k_brier'] for r in z):.4f} Acc={mean(r['k_acc'] for r in z):.3f} E|K-true|={mean(r['k_mae_expected'] for r in z):.3f} Ptrue={mean(r['p_true_k'] for r in z):.3f}")


def mechanism_rows(all_states, truth, roles):
    rows=[]; examples=[]
    for seed, arms in all_states.items():
        for arm, rounds in arms.items():
            s1,s2,s3=rounds[1],rounds[2],rounds[3]
            for sid,t in truth.items():
                if sid not in s1 or sid not in s3: continue
                st1,st3=s1[sid],s3[sid]; st2=s2.get(sid,st1)
                tr=set(map(str,t["roles"])); k=int(t["k"])
                p1,p2,p3=st1.role_probs,st2.role_probs,st3.role_probs
                true_delta=mean(p3[r]-p1[r] for r in tr)
                false=[r for r in roles if r not in tr]
                false_delta=mean(p3[r]-p1[r] for r in false)
                top1=sorted(roles,key=lambda r:p1[r],reverse=True)[:k]
                top3=sorted(roles,key=lambda r:p3[r],reverse=True)[:k]
                f1=[r for r in top1 if r not in tr]
                removed=[r for r in f1 if r not in top3]
                missed=[r for r in tr if r not in top1]
                recovered=[r for r in missed if r in top3]
                leader1=max(roles,key=lambda r:p1[r]); leader3=max(roles,key=lambda r:p3[r])
                row={
                    "seed":seed,"arm":arm,"session_id":sid,"true_k":k,
                    "true_delta_r3_r1":true_delta,"false_delta_r3_r1":false_delta,
                    "separation_gain":true_delta-false_delta,
                    "false_negative_update_rate":mean(float(p3[r]<p1[r]) for r in false),
                    "true_positive_update_rate":mean(float(p3[r]>p1[r]) for r in tr),
                    "topk_recall_r1":len(set(top1)&tr)/k,"topk_recall_r3":len(set(top3)&tr)/k,
                    "topk_recall_gain":len(set(top3)&tr)/k-len(set(top1)&tr)/k,
                    "false_topk_r1":len(f1),"false_topk_removed":len(removed),
                    "false_topk_removed_rate":len(removed)/len(f1) if f1 else 1.0,
                    "missed_true_r1":len(missed),"missed_true_recovered":len(recovered),
                    "missed_true_recovered_rate":len(recovered)/len(missed) if missed else 1.0,
                    "wrong_leader_r1":float(leader1 not in tr),
                    "wrong_leader_corrected_r3":float((leader1 not in tr) and (leader3 in tr)),
                }
                rows.append(row)
                qs=[h.query for h in st3.history]
                for r in f1:
                    drop=float(p1[r]-p3[r])
                    examples.append({
                        "seed":seed,"arm":arm,"session_id":sid,"true_k":k,"false_role":r,
                        "p_r1":float(p1[r]),"p_r2":float(p2[r]),"p_r3":float(p3[r]),"drop_r1_r3":drop,
                        "removed_from_topk":int(r not in top3),
                        "query_r1":qs[0] if len(qs)>0 else "","query_r2":qs[1] if len(qs)>1 else "","query_r3":qs[2] if len(qs)>2 else "",
                    })
    examples.sort(key=lambda x:(x["removed_from_topk"],x["drop_r1_r3"]),reverse=True)
    return rows,examples


def summarize_mechanism(rows):
    print("\n"+"="*100); print("BELIEF CORRECTION / REVERSAL AUDIT"); print("="*100)
    for arm in ARMS:
        z=[r for r in rows if r["arm"]==arm]
        print(f"{arm:22s} trueΔ={mean(r['true_delta_r3_r1'] for r in z):+.4f} falseΔ={mean(r['false_delta_r3_r1'] for r in z):+.4f} separation_gain={mean(r['separation_gain'] for r in z):+.4f} topK_gain={mean(r['topk_recall_gain'] for r in z):+.4f} false-remove={mean(r['false_topk_removed_rate'] for r in z):.3f} true-recover={mean(r['missed_true_recovered_rate'] for r in z):.3f}")
    for metric,lower in [("separation_gain",False),("topk_recall_gain",False),("false_topk_removed_rate",False)]:
        ba=paired_system_delta(rows,"A_original_static","B_original_adaptive",metric,lower=lower)
        dc=paired_system_delta(rows,"C_grpo_static","D_grpo_adaptive",metric,lower=lower)
        print(f"  B-A {metric}: delta={ba['delta']:+.5f} CI=[{ba['ci_lo']:+.5f},{ba['ci_hi']:+.5f}] p={ba['p2']:.4f} systems={ba['n']}")
        print(f"  D-C {metric}: delta={dc['delta']:+.5f} CI=[{dc['ci_lo']:+.5f},{dc['ci_hi']:+.5f}] p={dc['p2']:.4f} systems={dc['n']}")


def summary_by_method(rows, feature_set=None):
    out=[]
    for rnd in (1,2,3):
        for arm in ARMS:
            z=[r for r in rows if r["arm"]==arm and int(r["round"])==rnd and (feature_set is None or r.get("feature_set")==feature_set)]
            if z:
                out.append({"arm":arm,"round":rnd,"n":len(z),"k_nll":mean(r["k_nll"] for r in z),"k_brier":mean(r["k_brier"] for r in z),"k_acc":mean(r["k_acc"] for r in z),"k_mae_expected":mean(r["k_mae_expected"] for r in z),"p_true_k":mean(r["p_true_k"] for r in z)})
    return out


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",default="artifacts/adaptive_v11_factorial_heldout32")
    ap.add_argument("--belief",default="artifacts/adaptive_v11_active_v1/frozen_belief.joblib")
    ap.add_argument("--out",default="artifacts/adaptive_v11_factorial_heldout32/posthoc_k_mechanism")
    ap.add_argument("--cv-C",type=float,default=0.20,help="Locked to the existing K-head regularization by default; not tuned on heldout labels.")
    args=ap.parse_args()

    root=Path(args.root); outdir=Path(args.out); outdir.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((root/"manifest_privileged.json").read_text())
    truth={str(x["session_id"]):x for x in manifest["systems"]}
    roles=list(TRAIN_ROLES)
    ks=tuple(sorted({int(x["k"]) for x in truth.values()})) or KS_DEFAULT
    seeds=sorted((root/"runs").glob("seed_*"))
    if not seeds: raise SystemExit(f"No seed directories under {root/'runs'}")
    print(f"[load] systems={len(truth)} seeds={[x.name for x in seeds]} ks={ks} roles={len(roles)}")

    all_states={}; flat=[]
    for sd in seeds:
        seed=int(sd.name.split("_")[-1]); all_states[seed]={}
        for arm in ARMS:
            rr=sd/arm
            all_states[seed][arm]={}
            for rnd in (0,1,2,3):
                p=rr/f"states_dev_round_{rnd:02d}.jsonl"
                if not p.exists(): raise SystemExit(f"Missing {p}")
                all_states[seed][arm][rnd]=state_map(p)
            for rnd in (1,2,3):
                sm=all_states[seed][arm][rnd]
                for sid,t in truth.items():
                    st=sm[sid]
                    flat.append({
                        "seed":seed,"arm":arm,"session_id":sid,"round":rnd,"true_k":int(t["k"]),
                        "state":st,
                        "role_features":role_only_features(st,roles),
                        "full_features":k_features(st.history,st.role_probs,roles),
                    })

    # Experiment 1: zero-training cardinality from calibrated role marginals.
    rolecard=[]
    for r in flat:
        p=role_cardinality_probs(r["state"],roles,ks); m=pred_metrics(p,r["true_k"],ks)
        rolecard.append({k:v for k,v in r.items() if k not in ("state","role_features","full_features")} | m)
    summarize_prediction_rows(rolecard,"K EXPERIMENT 1 — POISSON-BINOMIAL CARDINALITY FROM ROLE POSTERIORS (NO NEW TRAINING)")
    write_csv(outdir/"k_role_cardinality_predictions.csv",rolecard)

    # Experiment 2: existing frozen K head, a fully disjoint/training-frozen evaluation.
    frozen=[]; belief_path=Path(args.belief)
    if belief_path.exists():
        belief=FrozenBeliefBundle.load(belief_path)
        print(f"[frozen-belief] version={getattr(belief,'VERSION','?')} fusion_alpha={getattr(belief,'fusion_alpha','?')} fit_summary={getattr(belief,'fit_summary',{})}")
        for r in flat:
            kp,_=belief.predict(r["state"].history); p=vector_from_kdict(kp,ks); m=pred_metrics(p,r["true_k"],ks)
            frozen.append({k:v for k,v in r.items() if k not in ("state","role_features","full_features")} | m)
        summarize_prediction_rows(frozen,"K EXPERIMENT 2 — EXISTING FROZEN K HEAD ON ADAPTIVE/STATIC HELD-OUT HISTORIES")
        write_csv(outdir/"k_frozen_head_predictions.csv",frozen)
    else:
        print(f"[warn] frozen belief missing: {belief_path}; skipping experiment 2")

    # Experiment 3: grouped system-disjoint CV as a LEARNABILITY diagnostic, not an untouched-test claim.
    cv_role=cv_k(flat,truth,ks,"role_features",args.cv_C)
    cv_full=cv_k(flat,truth,ks,"full_features",args.cv_C)
    cv=cv_role+cv_full
    for fs,name in [("role_posterior_only","K EXPERIMENT 3A — SYSTEM-DISJOINT CV, ROLE-POSTERIOR FEATURES"),("full_public","K EXPERIMENT 3B — SYSTEM-DISJOINT CV, FULL PUBLIC JUDGE+ROLE FEATURES")]:
        z=[r for r in cv if r["feature_set"]==fs]; summarize_prediction_rows(z,name)
    write_csv(outdir/"k_groupcv_predictions.csv",cv)

    # Primary K comparisons at R3 full-public CV.
    z=[r for r in cv if r["feature_set"]=="full_public" and int(r["round"])==3]
    print("\n"+"="*100); print("K R3 ADAPTIVITY CONTRASTS — SYSTEM-DISJOINT CV, FULL PUBLIC FEATURES"); print("="*100)
    for a,b,label in [(ARMS[0],ARMS[1],"B-A Original: adaptive-static"),(ARMS[2],ARMS[3],"D-C GRPO: adaptive-static"),(ARMS[0],ARMS[3],"D-A Full-original static")]:
        d=paired_system_delta(z,a,b,"k_nll",lower=True)
        print(f"{label:32s} ΔNLL={d['delta']:+.5f} CI=[{d['ci_lo']:+.5f},{d['ci_hi']:+.5f}] p={d['p2']:.4f} wins={d['wins']}/{d['n']}")

    # Compare each arm to uniform K prior at R3.
    uniform_nll=math.log(len(ks)); uniform_brier=1-1/len(ks)
    print(f"\nUniform baseline: NLL={uniform_nll:.6f} Brier={uniform_brier:.6f} Acc={1/len(ks):.3f} Expected-K MAE=1.000 for balanced K=3..6")
    for arm in ARMS:
        zz=[r for r in z if r["arm"]==arm]
        by=defaultdict(list)
        for r in zz: by[r["session_id"]].append(float(r["k_nll"])-uniform_nll)
        ds=[mean(v) for v in by.values()]; lo,hi=bootstrap_ci(ds); p=signflip_p(ds)
        print(f" {arm:22s} ΔNLL-vs-uniform={mean(ds):+.5f} CI=[{lo:+.5f},{hi:+.5f}] p={p:.4f}")

    # Experiment 4: mechanism / negative belief updates.
    mech,examples=mechanism_rows(all_states,truth,roles)
    summarize_mechanism(mech)
    write_csv(outdir/"belief_correction_metrics.csv",mech)
    write_csv(outdir/"belief_reversal_examples.csv",examples[:200])

    # Compact machine-readable summaries.
    summary={
        "systems":len(truth),"seeds":[x.name for x in seeds],"ks":list(ks),
        "uniform":{"nll":uniform_nll,"brier":uniform_brier,"accuracy":1/len(ks),"expected_k_mae_balanced":1.0},
        "role_cardinality":summary_by_method(rolecard),
        "frozen_head":summary_by_method(frozen) if frozen else [],
        "groupcv_role":summary_by_method(cv,"role_posterior_only"),
        "groupcv_full":summary_by_method(cv,"full_public"),
    }
    (outdir/"summary.json").write_text(json.dumps(summary,indent=2))

    # Evidence gate. This does not turn CV into an untouched test; it only decides the next experiment.
    b=[r for r in z if r["arm"]=="B_original_adaptive"]
    b_nll=mean(r["k_nll"] for r in b); b_acc=mean(r["k_acc"] for r in b)
    f_b=[r for r in frozen if r["arm"]=="B_original_adaptive" and int(r["round"])==3]
    print("\n"+"="*100); print("NEXT-STEP GATE"); print("="*100)
    if f_b:
        fn=mean(r["k_nll"] for r in f_b); fa=mean(r["k_acc"] for r in f_b)
        print(f"Frozen disjoint K head, B R3: NLL={fn:.4f} Acc={fa:.3f} (uniform NLL={uniform_nll:.4f}, Acc=.250)")
    print(f"System-disjoint CV learnability, B R3: NLL={b_nll:.4f} Acc={b_acc:.3f}")
    if f_b and mean(r["k_nll"] for r in f_b) < uniform_nll:
        print("GATE: Existing disjoint K head has some held-out signal. Next: quantify significance/calibration and consider freezing it for a final untouched replication.")
    elif b_nll < uniform_nll:
        print("GATE: K signal is learnable from adaptive public trajectories, but the current frozen head does not transfer. Next: train a fresh K head only on disjoint TRAIN systems, freeze it, then evaluate once on these 32 held-out systems.")
    else:
        print("GATE: No useful K signal yet. Next: do NOT fit harder on these held-out labels; train K-aware probing/head on disjoint TRAIN systems and run a fresh held-out K benchmark.")
    print(f"Artifacts written to {outdir}")


if __name__=="__main__":
    main()
