from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np

from analysis.adaptive_fanout_count_v12 import cosine, sigmoid

def _unit(v: np.ndarray) -> np.ndarray:
    x=np.asarray(v,dtype=np.float32).reshape(-1)
    return x/max(float(np.linalg.norm(x)),1e-8)


def _residual(q: np.ndarray, o: np.ndarray) -> np.ndarray:
    return _unit(np.asarray(o,dtype=np.float32)-np.asarray(q,dtype=np.float32))


def _style(text: str) -> np.ndarray:
    import re
    t=str(text or ""); words=re.findall(r"\b[\w'-]+\b",t,flags=re.UNICODE); chars=max(len(t),1)
    return np.asarray([
        math.log1p(len(t)), math.log1p(len(words)), math.log1p(t.count("\n")),
        math.log1p(t.count("- ")+t.count("* ")), float("```" in t),
        sum(ch.isdigit() for ch in t)/chars, t.count(":")/chars, t.count("|")/chars,
        (t.count("{")+t.count("}"))/chars,
    ],dtype=np.float32)


def load_fingerprint_examples(paths: Sequence[tuple[str, float]], allowed_roles: set[str]) -> list[dict]:
    """TRAIN-only counterfactual examples. Positive outputs must actually route to target.

    Truth/selected-role fields are training labels only. They are never runtime features.
    Grouping by pair_id keeps the same matched system out of both train and OOF fold.
    """
    out=[]; seen=set()
    for path,source_weight in paths:
        p=Path(path)
        if not p.exists():
            continue
        for line_no,line in enumerate(p.open(encoding="utf-8"),1):
            if not line.strip(): continue
            try: rec=json.loads(line)
            except Exception: continue
            role=str(rec.get("target_role","")); query=str(rec.get("query","")).strip()
            if rec.get("phase")!="train" or rec.get("status")!="ok" or role not in allowed_roles or not query:
                continue
            rb=rec.get("reward_breakdown_training_only") or {}
            strict=bool(rb.get("strict_specific_pass",False)); confirmed=bool(rec.get("confirmation_pass_training_only",False))
            base_w=float(source_weight)*(1.0+(0.75 if strict else 0.0)+(0.75 if confirmed else 0.0))
            branches=list(rec.get("branches_training_only") or [])+list(rec.get("confirmation_branches_training_only") or [])
            for b in branches:
                pid=str(b.get("pair_id") or f"line{line_no}")
                plus_sel=set(map(str,b.get("plus_selected") or [])); sham_sel=set(map(str,b.get("sham_selected") or []))
                candidates=[]
                # Presence is only useful as positive behavioral evidence when the target was activated.
                if role in plus_sel: candidates.append(("plus_output",1,"plus"))
                if role in sham_sel: candidates.append(("sham_output",1,"sham"))
                candidates.append(("minus_output",0,"minus"))
                for field,y,branch in candidates:
                    text=str(b.get(field,"") or "").strip()
                    if not text: continue
                    key=(role,query,y,hashlib.sha1(text.encode()).hexdigest())
                    if key in seen: continue
                    seen.add(key)
                    out.append({
                        "role":role,"query":query,"output":text,"y":int(y),
                        "weight":base_w,"group":f"{role}|{pid}","pair_id":pid,"branch":branch,
                        "source":p.name,
                    })
    return out


class BehavioralFingerprintDetectorV13:
    """Per-role cross-fitted behavioral detector using output-minus-query residual prototypes.

    The role-specific prototype is learned only from TRAIN counterfactual branches. Runtime
    inputs are candidate role + public query + public external output.
    """
    def __init__(self, embedder, role_descriptions: dict[str,str], c: float=0.05):
        self.embedder=embedder; self.role_descriptions=role_descriptions; self.c=float(c)
        self.models={}; self.prototypes={}; self.thresholds={}; self.report={}

    def _embed_record(self,x:dict)->dict:
        q=self.embedder.encode(x["query"]).astype(np.float32)
        o=self.embedder.encode(x["output"]).astype(np.float32)
        return {**x,"qv":q,"ov":o,"rv":_residual(q,o)}

    def _protos(self, rows:list[dict])->tuple[np.ndarray,np.ndarray]:
        pos=[x["rv"] for x in rows if x["y"]==1]; neg=[x["rv"] for x in rows if x["y"]==0]
        if not pos or not neg: raise RuntimeError("need both classes for fingerprint prototypes")
        return _unit(np.mean(pos,axis=0)), _unit(np.mean(neg,axis=0))

    def _features(self, role:str, qv:np.ndarray, ov:np.ndarray, text:str, pp:np.ndarray, pn:np.ndarray)->np.ndarray:
        rv=_residual(qv,ov); rd=self.embedder.encode(self.role_descriptions.get(role,role))
        scal=np.asarray([
            cosine(rv,pp), cosine(rv,pn), cosine(rv,pp)-cosine(rv,pn),
            cosine(ov,rd), cosine(qv,rd), cosine(ov,qv), cosine(rv,rd),
        ],dtype=np.float32)
        return np.concatenate([scal,_style(text)])

    @staticmethod
    def _fit(X,y,w,c):
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        m=make_pipeline(StandardScaler(),LogisticRegression(C=float(c),class_weight="balanced",max_iter=5000,random_state=0))
        m.fit(X,y,logisticregression__sample_weight=w); return m

    @staticmethod
    def _best_threshold(y,scores):
        from sklearn.metrics import balanced_accuracy_score
        cand=np.unique(np.quantile(scores,np.linspace(.05,.95,181))); bt=0.0; bb=-1.0
        for t in cand:
            ba=float(balanced_accuracy_score(y,(scores>=t).astype(int)))
            if ba>bb: bt=float(t); bb=ba
        return bt,bb

    def fit(self, examples:list[dict], roles:Sequence[str], folds:int=5)->dict:
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import GroupKFold
        embedded=[self._embed_record(x) for x in examples]
        by=defaultdict(list)
        for x in embedded: by[x["role"]].append(x)
        role_reports={}; aucs=[]; bas=[]
        for role in roles:
            rows=by.get(role,[])
            y=np.asarray([x["y"] for x in rows],dtype=int)
            groups=np.asarray([x["group"] for x in rows],dtype=object)
            if len(rows)<12 or len(np.unique(y))<2 or len(np.unique(groups))<3:
                raise RuntimeError(f"insufficient V13 detector data for role {role}: n={len(rows)} groups={len(np.unique(groups))}")
            nfold=min(max(2,int(folds)),len(np.unique(groups)))
            oof=np.full(len(rows),np.nan,dtype=float)
            gkf=GroupKFold(n_splits=nfold)
            dummy=np.zeros((len(rows),1),dtype=np.float32)
            for tr,te in gkf.split(dummy,y,groups):
                train=[rows[i] for i in tr]; pp,pn=self._protos(train)
                Xtr=np.stack([self._features(role,x["qv"],x["ov"],x["output"],pp,pn) for x in train])
                Xte=np.stack([self._features(role,rows[i]["qv"],rows[i]["ov"],rows[i]["output"],pp,pn) for i in te])
                w=np.asarray([float(x.get("weight",1.0)) for x in train])
                m=self._fit(Xtr,y[tr],w,self.c); oof[te]=m.decision_function(Xte)
            if not np.isfinite(oof).all(): raise RuntimeError(f"incomplete OOF for {role}")
            auc=float(roc_auc_score(y,oof)); thr,ba=self._best_threshold(y,oof)
            pp,pn=self._protos(rows)
            X=np.stack([self._features(role,x["qv"],x["ov"],x["output"],pp,pn) for x in rows])
            w=np.asarray([float(x.get("weight",1.0)) for x in rows])
            self.models[role]=self._fit(X,y,w,self.c); self.prototypes[role]=(pp,pn); self.thresholds[role]=thr
            role_reports[role]={"examples":len(rows),"groups":len(np.unique(groups)),"positive_rate":float(y.mean()),"oof_auc":auc,"oof_balanced_accuracy":ba,"threshold":thr}
            aucs.append(auc); bas.append(ba)
        self.report={"roles":role_reports,"macro_oof_auc":float(np.mean(aucs)),"macro_oof_balanced_accuracy":float(np.mean(bas)),"examples":len(examples)}
        return self.report

    def score(self,role:str,query:str,output:str)->tuple[float,float]:
        q=self.embedder.encode(query); o=self.embedder.encode(output); pp,pn=self.prototypes[role]
        x=self._features(role,q,o,output,pp,pn).reshape(1,-1)
        raw=float(self.models[role].decision_function(x)[0]); centered=raw-float(self.thresholds[role])
        return centered,sigmoid(centered)


