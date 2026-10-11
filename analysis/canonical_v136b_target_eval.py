from pathlib import Path
from collections import defaultdict, Counter
import argparse, json
import numpy as np

from sklearn.metrics import roc_auc_score

import torch
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from analysis.train_v13_6b_activation_judge import (
    presence_text,
    ROLE_DESCRIPTIONS,
)

BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
ADAPTER = Path("artifacts/v13_6b_activation_judge/best_adapter")

def readj(p):
    return [json.loads(x) for x in p.open() if x.strip()]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True, choices=["k3","k4","high"])
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    root = Path("artifacts/canonical_parallel_v1") / args.cohort
    outdir = Path("artifacts/canonical_v136b_target_eval") / args.cohort
    outdir.mkdir(parents=True, exist_ok=True)

    meta = json.loads((root / "meta.json").read_text())
    target_by_idx = {
        int(x["candidate_index"]): x["role"]
        for x in meta["probes"]
    }

    rows = readj(root / "responses_train_round_01.jsonl")

    print("="*100)
    print("V13.6B CANONICAL TARGET-ONLY BLACK-BOX EVAL")
    print("cohort =", args.cohort)
    print("rows   =", len(rows))
    print("="*100)

    tok = AutoTokenizer.from_pretrained(str(ADAPTER), use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    base = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=1,
        device_map={"":0},
        dtype=dtype,
    )
    base.config.pad_token_id = tok.pad_token_id
    base.config.use_cache = True

    model = PeftModel.from_pretrained(
        base,
        str(ADAPTER),
        is_trainable=False,
    )
    model.eval()

    texts=[]
    truth_labels=[]
    targets=[]

    for x in rows:
        idx = int(x["candidate_index"])
        target = target_by_idx[idx]

        q = str(x.get("query",""))
        ans = str(
            x.get("final_output")
            or x.get("answer")
            or x.get("output")
            or ""
        )

        txt = presence_text(
            target,
            ROLE_DESCRIPTIONS[target],
            target,
            q,
            ans,
        )

        selected = set(x.get("selected_roles_training_only") or [])

        texts.append(txt)
        truth_labels.append(int(target in selected))
        targets.append(target)

    logits=[]

    with torch.inference_mode():
        for st in range(0,len(texts),args.batch_size):
            enc = tok(
                texts[st:st+args.batch_size],
                padding=True,
                truncation=True,
                max_length=1024,
                return_tensors="pt",
            ).to("cuda:0")

            with torch.autocast(
                device_type="cuda",
                dtype=dtype,
                enabled=True,
            ):
                z = model(**enc).logits[:,0]

            logits.extend(z.float().cpu().numpy().tolist())

    labels=np.asarray(truth_labels)
    scores=np.asarray(logits)

    auc = roc_auc_score(labels,scores)
    pred0 = scores > 0.0

    print()
    print("TARGET ACTIVATION DETECTOR")
    print("--------------------------")
    print("positives =", int(labels.sum()))
    print("negatives =", int((1-labels).sum()))
    print("AUC       =", round(float(auc),4))
    print("acc@0     =", round(float(np.mean(pred0==labels)),4))
    print(
        "TPR@0     =",
        round(float(np.mean(pred0[labels==1])),4)
    )
    print(
        "FPR@0     =",
        round(float(np.mean(pred0[labels==0])),4)
    )

    scored=[]
    for x,t,y,z in zip(rows,targets,labels,scores):
        scored.append({
            "system_id":x.get("system_id") or x.get("session_id"),
            "candidate_index":int(x["candidate_index"]),
            "target":t,
            "truth_target_activated":int(y),
            "logit":float(z),
            "pred_at_0":int(z>0),
            "truth_roles_training_only":
                x.get("truth_roles_training_only") or [],
            "mode":x.get("mode"),
        })

    with (outdir/"scores.jsonl").open("w") as f:
        for x in scored:
            f.write(json.dumps(x)+"\n")

    by=defaultdict(list)
    for x in scored:
        by[x["system_id"]].append(x)

    print()
    print("="*100)
    print("END-TO-END K / ROLE-SET USING FIXED THRESHOLD 0")
    print("="*100)

    exact_k=[]
    exact_set=[]
    maes=[]

    for sid,rr in sorted(by.items()):
        gt=set(rr[0]["truth_roles_training_only"])

        predicted={
            x["target"]
            for x in rr
            if x["logit"] > 0
        }

        khat=len(predicted)
        k=len(gt)

        exact_k.append(khat==k)
        exact_set.append(predicted==gt)
        maes.append(abs(khat-k))

        print(
            sid,
            f"K={k}",
            f"Khat={khat}",
            f"K_OK={khat==k}",
            f"SET_OK={predicted==gt}",
            f"pred={sorted(predicted)}",
            f"truth={sorted(gt)}",
        )

    print()
    print("SYSTEM SUMMARY")
    print("systems   =",len(by))
    print("exact K   =",round(float(np.mean(exact_k)),4))
    print("exact set =",round(float(np.mean(exact_set)),4))
    print("K MAE     =",round(float(np.mean(maes)),4))

    # Diagnostic threshold sweep. This is NOT a deployable threshold selection;
    # it tells us whether the representation contains separable signal.
    print()
    print("="*100)
    print("DIAGNOSTIC THRESHOLD SWEEP")
    print("="*100)

    candidate_thresholds=np.quantile(
        scores,
        np.linspace(0.05,0.95,91)
    )

    best=None

    for th in candidate_thresholds:
        ek=[]
        es=[]

        for sid,rr in by.items():
            gt=set(rr[0]["truth_roles_training_only"])

            pred={
                x["target"]
                for x in rr
                if x["logit"] > th
            }

            ek.append(len(pred)==len(gt))
            es.append(pred==gt)

        item=(
            float(np.mean(es)),
            float(np.mean(ek)),
            float(th),
        )

        if best is None or item > best:
            best=item

    print(
        "ORACLE best exact-set =",
        round(best[0],4),
        "exact-K =",
        round(best[1],4),
        "threshold =",
        round(best[2],4),
    )

    # Per-target diagnostic
    print()
    print("PER TARGET")
    for r in sorted(set(targets)):
        ix=np.asarray([t==r for t in targets])
        yy=labels[ix]
        zz=scores[ix]

        if len(set(yy.tolist()))==2:
            ra=roc_auc_score(yy,zz)
        else:
            ra=float("nan")

        print(
            f"{r:25s}",
            f"n={len(yy):2d}",
            f"pos={int(yy.sum()):2d}",
            f"auc={ra:.3f}" if not np.isnan(ra) else "auc=NA",
        )

if __name__=="__main__":
    main()
