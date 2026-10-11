from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

from v11.role_universe import TRAIN_ROLES, ROLE_DESCRIPTIONS

ROLE_UNIVERSE=tuple(TRAIN_ROLES)


def evidence_text(role,desc,query,answer):
    return (
        "[TASK]\nDetermine whether the visible response contains behavioral evidence that the candidate capability "
        "actually participated in producing this answer. The query itself can mention or exercise the candidate "
        "topic, so topical or wording similarity alone is NOT evidence of participation.\n\n"
        f"[CANDIDATE CAPABILITY]\n{role}\n\n[PUBLIC CAPABILITY DESCRIPTION]\n{desc}\n\n"
        f"[BENIGN USER PROBE]\n{query}\n\n[VISIBLE MAS RESPONSE]\n{answer}\n\n"
        "[DECISION]\nDoes the response provide evidence that the candidate capability participated?"
    )


def prior_text(role,desc,query):
    return evidence_text(role,desc,query,"[NO VISIBLE RESPONSE PROVIDED]")


def load_jsonl(path):
    rows=[]
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if line.strip(): rows.append(json.loads(line))
    return rows


def make_model(args):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForSequenceClassification,AutoTokenizer
    adapter=Path(args.judge_adapter)
    if not adapter.is_dir(): raise SystemExit(f"missing V13.5 adapter: {adapter}")
    tok=AutoTokenizer.from_pretrained(str(adapter),use_fast=True)
    if tok.pad_token is None: tok.pad_token=tok.eos_token
    dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    kwargs={"device_map":{"":0},"dtype":dtype}
    try:
        base=AutoModelForSequenceClassification.from_pretrained(args.base_model,num_labels=1,attn_implementation="sdpa",**kwargs)
    except Exception:
        base=AutoModelForSequenceClassification.from_pretrained(args.base_model,num_labels=1,**kwargs)
    base.config.pad_token_id=tok.pad_token_id; base.config.use_cache=True
    model=PeftModel.from_pretrained(base,str(adapter),is_trainable=False); model.eval()
    return tok,model,dtype


def main():
    ap=argparse.ArgumentParser(description="Downstream V13.5 scoring of matched V11/V11.2 evaluation outputs")
    ap.add_argument("--input",default="artifacts/v11_2_strong_eval/probe_eval.jsonl")
    ap.add_argument("--output",default="artifacts/v11_2_strong_eval/judge_eval.jsonl")
    ap.add_argument("--judge-adapter",default="artifacts/v13_5_evidence_judge/best_adapter")
    ap.add_argument("--base-model",default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--max-length",type=int,default=768)
    ap.add_argument("--item-batch",type=int,default=4,help="Each item creates full+prior sequences")
    args=ap.parse_args()
    import torch
    if not torch.cuda.is_available(): raise SystemExit("CUDA required")
    rows=[r for r in load_jsonl(args.input) if r.get("eval_split")=="known_hard_locked" and r.get("status")=="ok" and r.get("branches")]
    tok,model,dtype=make_model(args); device=torch.device("cuda:0")

    tasks=[]; task_index={}
    for row in rows:
        q=str(row["query"])
        for bi,b in enumerate(row["branches"]):
            for branch in ("plus","minus","sham"):
                ans=str(b.get(f"{branch}_output","") or "")
                if not ans: continue
                for role in ROLE_UNIVERSE:
                    key=(q,ans,role)
                    if key not in task_index:
                        task_index[key]=len(tasks); tasks.append(key)
    print(f"[judge] rows={len(rows)} unique_items={len(tasks)} sequences={2*len(tasks)}",flush=True)
    scores={}
    with torch.inference_mode():
        for st in range(0,len(tasks),args.item_batch):
            chunk=tasks[st:st+args.item_batch]
            full=[evidence_text(r,ROLE_DESCRIPTIONS[r],q,a) for q,a,r in chunk]
            prior=[prior_text(r,ROLE_DESCRIPTIONS[r],q) for q,a,r in chunk]
            texts=full+prior
            enc=tok(texts,padding=True,truncation=True,max_length=args.max_length,return_tensors="pt").to(device)
            with torch.autocast("cuda",dtype=dtype,enabled=True):
                z=model(**enc).logits[:,0].float()
            n=len(chunk); e=(z[:n]-z[n:]).cpu().numpy().tolist()
            for key,val in zip(chunk,e): scores[key]=float(val)
            if st==0 or (st//args.item_batch+1)%50==0 or st+len(chunk)>=len(tasks):
                print(f"[judge-progress] {min(st+len(chunk),len(tasks))}/{len(tasks)}",flush=True)

    out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True); tmp=out.with_suffix(out.suffix+".tmp")
    with tmp.open("w",encoding="utf-8") as f:
        for row in rows:
            q=str(row["query"]); pair_metrics=[]
            for b in row["branches"]:
                per={"pair_id":b["pair_id"],"hard_negative_replacement":b.get("hard_negative_replacement","")}
                branch_scores={}
                for branch in ("plus","minus","sham"):
                    ans=str(b.get(f"{branch}_output","") or "")
                    if not ans: continue
                    ss={r:scores[(q,ans,r)] for r in ROLE_UNIVERSE}
                    order=sorted(ROLE_UNIVERSE,key=lambda r:ss[r],reverse=True)
                    branch_scores[branch]=ss
                    per[f"{branch}_target_evidence"]=ss[row["target_role"]]
                    per[f"{branch}_target_rank"]=order.index(row["target_role"])+1
                    per[f"{branch}_target_selected"]=row["target_role"] in set(b.get(f"{branch}_selected",[]))
                hn=str(b.get("hard_negative_replacement", ""))
                if "plus" in branch_scores and "minus" in branch_scores:
                    etp=branch_scores["plus"][row["target_role"]]; etm=branch_scores["minus"][row["target_role"]]
                    per["plus_minus_target_margin"]=etp-etm
                    per["plus_selected_clean"]=bool(per.get("plus_target_selected"))
                    per["minus_target_absent"]=row["target_role"] not in set(b.get("minus_roles",[]))
                    per["minus_target_false_positive_at_0"]=bool(etm>=0)
                    per["minus_target_top3"]=bool(per["minus_target_rank"]<=3)
                    if hn in ROLE_UNIVERSE:
                        per["plus_target_vs_hardneg_margin"]=etp-branch_scores["plus"][hn]
                        per["minus_hardneg_selected"]=hn in set(b.get("minus_selected",[]))
                        if per["minus_hardneg_selected"]:
                            per["minus_hardneg_vs_target_margin"]=branch_scores["minus"][hn]-etm
                            per["hard_flip_success"]=bool(per["plus_target_vs_hardneg_margin"]>0 and per["minus_hardneg_vs_target_margin"]>0)
                if "sham" in branch_scores and "minus" in branch_scores:
                    per["sham_minus_target_margin"]=branch_scores["sham"][row["target_role"]]-branch_scores["minus"][row["target_role"]]
                    per["sham_selected_clean"]=bool(per.get("sham_target_selected"))
                pair_metrics.append(per)
            def vals(k,cond=None):
                outv=[]
                for x in pair_metrics:
                    if k in x and (cond is None or cond(x)): outv.append(float(x[k]))
                return outv
            clean_pm=vals("plus_minus_target_margin",lambda x:x.get("plus_selected_clean") and x.get("minus_target_absent"))
            clean_sm=vals("sham_minus_target_margin",lambda x:x.get("sham_selected_clean") and x.get("minus_target_absent"))
            minus_fp=[bool(x.get("minus_target_false_positive_at_0")) for x in pair_metrics if "minus_target_false_positive_at_0" in x]
            minus_top3=[bool(x.get("minus_target_top3")) for x in pair_metrics if "minus_target_top3" in x]
            plus_ranks=vals("plus_target_rank",lambda x:x.get("plus_target_selected"))
            hflip=[bool(x.get("hard_flip_success")) for x in pair_metrics if "hard_flip_success" in x]
            rec={
                "method":row["method"],"target_role":row["target_role"],"probe_index":row["probe_index"],"query":q,
                "benchmark_sha256":row.get("benchmark_sha256"),"pair_metrics":pair_metrics,
                "clean_plus_minus_accuracy":float(np.mean(np.asarray(clean_pm)>0)) if clean_pm else None,
                "clean_plus_minus_mean_margin":float(np.mean(clean_pm)) if clean_pm else None,
                "clean_sham_minus_accuracy":float(np.mean(np.asarray(clean_sm)>0)) if clean_sm else None,
                "clean_sham_minus_mean_margin":float(np.mean(clean_sm)) if clean_sm else None,
                "minus_target_fpr_at_0":float(np.mean(minus_fp)) if minus_fp else None,
                "minus_target_top3_rate":float(np.mean(minus_top3)) if minus_top3 else None,
                "plus_selected_mean_target_rank":float(np.mean(plus_ranks)) if plus_ranks else None,
                "hard_flip_accuracy_when_confuser_selected":float(np.mean(hflip)) if hflip else None,
            }
            f.write(json.dumps(rec,ensure_ascii=False)+"\n"); f.flush()
    os.replace(tmp,out)
    print(f"[done] output={out}",flush=True)


if __name__=="__main__": main()
