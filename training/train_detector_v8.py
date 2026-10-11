from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold

from estimator.candidate_presence_contrastive_v64 import load_profile
from estimator.transcript_detector_v8 import TranscriptDetectorV8, load_jsonl


def metrics(y, p):
    y=np.asarray(y,dtype=int);p=np.asarray(p,dtype=float);pred=p>=.5
    return {
        "auc": float(roc_auc_score(y,p)) if len(set(y.tolist()))==2 else float("nan"),
        "balanced_accuracy": float(balanced_accuracy_score(y,pred)),
        "f1": float(f1_score(y,pred,zero_division=0)),
        "brier": float(brier_score_loss(y,p)),
    }


def main():
    ap=argparse.ArgumentParser(description="Train the post-RL V8 black-box transcript detector")
    ap.add_argument("--profile",default="config/application_profile.json")
    ap.add_argument("--public",default="artifacts/v8_detector_train_public.jsonl")
    ap.add_argument("--truth",default="artifacts/v8_detector_train_truth.jsonl")
    ap.add_argument("--output",default="artifacts/v8_transcript_detector.joblib")
    ap.add_argument("--summary",default="artifacts/v8_transcript_detector_summary.json")
    args=ap.parse_args()
    profile=load_profile(args.profile);pub=load_jsonl(args.public);truth=load_jsonl(args.truth)
    tm={(r['system_id'],r['role']):r for r in truth};rows=[];y=[];groups=[]
    for r in pub:
        t=tm[(r['system_id'],r['role'])];rows.append(r);y.append(int(t['label']));groups.append(r['system_id'])
    unique=len(set(groups));n_splits=min(4,unique)
    oof=np.full(len(rows),np.nan,dtype=float)
    if n_splits>=2:
        gkf=GroupKFold(n_splits=n_splits)
        for tr,te in gkf.split(np.arange(len(rows)),groups=groups):
            det=TranscriptDetectorV8(profile['candidate_roles'],profile.get('role_descriptions',{})).fit([rows[i] for i in tr],[y[i] for i in tr])
            oof[te]=det.predict_proba([rows[i] for i in te])
        cv=metrics(y,oof)
        print("SYSTEM-GROUPED OOF CV:",json.dumps(cv,indent=2))
    else:
        cv={}
    det=TranscriptDetectorV8(profile['candidate_roles'],profile.get('role_descriptions',{})).fit(rows,y)
    p=det.predict_proba(rows);train=metrics(y,p)
    print("FULL-FIT TRAIN SANITY (not final evidence):",json.dumps(train,indent=2))
    det.save(args.output)
    summary={"systems":unique,"rows":len(rows),"grouped_cv":cv,"full_fit_train_sanity":train,"rl_reward_used_this_detector":False}
    Path(args.summary).write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print("saved:",args.output);print("saved:",args.summary)
if __name__=="__main__":main()
