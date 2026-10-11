from __future__ import annotations
import argparse, json, math
from pathlib import Path
import joblib
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from scipy.stats import spearmanr

from adaptive_v11_active.io import read_jsonl
from v11.role_universe import TRAIN_ROLES
from scripts.av11_query_utility import build_matrix, total_entropy


def load_states(root: Path, split: str, rnd: int):
    p = root / f"states_{split}_round_{rnd:02d}.jsonl"
    return {str(x["session_id"]): x for x in read_jsonl(p)} if p.exists() else {}

def load_candidates(root: Path, split: str, rnd: int):
    p = root / f"candidates_{split}_round_{rnd:02d}.jsonl"
    return {str(x["session_id"]): x for x in read_jsonl(p)} if p.exists() else {}

def truth_map(root: Path, split: str):
    p = root / "manifest_privileged.json"
    if not p.exists(): return {}
    obj=json.loads(p.read_text())
    return {str(x["session_id"]): set(map(str,x.get("roles",[]))) for x in obj.get("systems",[]) if x.get("split")==split}

def logloss(state: dict, truth: set[str]):
    vals=[]
    for r in TRAIN_ROLES:
        p=min(1-1e-9,max(1e-9,float(state["role_probs"][r])))
        y=1.0 if r in truth else 0.0
        vals.append(-(y*math.log(p)+(1-y)*math.log(1-p)))
    return float(np.mean(vals))

def add_root(rows, root: Path, split: str, rounds: range, source: str):
    tm=truth_map(root, split)
    for rnd in rounds:
        pre=load_states(root,split,rnd-1); post=load_states(root,split,rnd); cand=load_candidates(root,split,rnd)
        for sid,c in cand.items():
            if sid not in pre or sid not in post or not c.get("valid") or not str(c.get("query","")).strip(): continue
            h0=total_entropy(pre[sid]["role_probs"], list(TRAIN_ROLES)); h1=total_entropy(post[sid]["role_probs"], list(TRAIN_ROLES))
            rec={"session_id":sid,"round":rnd,"state":pre[sid],"query":str(c["query"]),"target_role":str(c.get("target_role", "")),"entropy_gain":h0-h1,"source":source}
            if sid in tm: rec["logloss_gain"] = logloss(pre[sid],tm[sid])-logloss(post[sid],tm[sid])
            rows.append(rec)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--kboot-root", default="artifacts/adaptive_v11_k_bootstrap_v1")
    ap.add_argument("--controller-root", default="artifacts/adaptive_v11_controller_v1")
    ap.add_argument("--output", required=True)
    args=ap.parse_args()
    rows=[]
    add_root(rows, Path(args.kboot_root), "train", range(1,7), "kboot_train")
    cr=Path(args.controller_root)
    add_root(rows, cr/"common", "dev", range(1,2), "old_common")
    add_root(rows, cr/"static", "dev", range(2,7), "old_static")
    add_root(rows, cr/"adaptive", "dev", range(2,7), "old_maxent")
    if len(rows)<300: raise SystemExit(f"too few training transitions: {len(rows)}")
    y=np.asarray([x["entropy_gain"] for x in rows],dtype=float)
    groups=np.asarray([x["source"]+"::"+x["session_id"] for x in rows])
    X0,mu,sd=build_matrix(rows,list(TRAIN_ROLES))
    alphas=[0.1,1.0,10.0,100.0]
    gkf=GroupKFold(n_splits=5)
    best=None
    for a in alphas:
        oof=np.zeros(len(rows),dtype=float)
        for tr,va in gkf.split(np.zeros(len(rows)),y,groups):
            m=Ridge(alpha=a,solver="lsqr")
            m.fit(X0[tr],y[tr]); oof[va]=m.predict(X0[va])
        mse=float(np.mean((oof-y)**2)); rho=float(spearmanr(oof,y).statistic)
        corr=float(np.corrcoef(oof,y)[0,1]) if np.std(oof)>0 and np.std(y)>0 else 0.0
        rec=(mse,-rho,a,oof,corr)
        if best is None or rec[:2]<best[:2]: best=rec
    mse,neg_rho,alpha,oof,corr=best; rho=-neg_rho
    model=Ridge(alpha=alpha,solver="lsqr").fit(X0,y)
    pred=model.predict(X0)
    bundle={"version":"av11-query-utility-v2","roles":list(TRAIN_ROLES),"model":model,"num_mean":mu,"num_std":sd,"alpha":alpha,"training_rows":len(rows),"sources":sorted(set(x["source"] for x in rows)),"oof_mse":mse,"oof_spearman":rho,"oof_pearson":corr,"target_mean":float(y.mean()),"target_std":float(y.std())}
    out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True); joblib.dump(bundle,out)
    diag={k:v for k,v in bundle.items() if k not in {"model","num_mean","num_std","roles"}}
    diag["positive_gain_fraction"]=float(np.mean(y>0)); diag["oof_sign_accuracy"]=float(np.mean((oof>0)==(y>0)))
    if all("logloss_gain" in x for x in rows):
        ll=np.asarray([x["logloss_gain"] for x in rows]); diag["entropy_gain_vs_logloss_gain_spearman"]=float(spearmanr(y,ll).statistic)
    (out.with_suffix(".json")).write_text(json.dumps(diag,indent=2,sort_keys=True))
    print(json.dumps(diag,indent=2,sort_keys=True)); print(f"[done] {out}")
if __name__=="__main__": main()
