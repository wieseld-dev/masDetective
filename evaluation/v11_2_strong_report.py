from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def load_jsonl(path):
    rows=[]
    p=Path(path)
    if not p.exists(): return rows
    with p.open(encoding="utf-8") as f:
        for line in f:
            if line.strip(): rows.append(json.loads(line))
    return rows


def mean(xs):
    xs=[float(x) for x in xs if x is not None and np.isfinite(float(x))]
    return float(np.mean(xs)) if xs else None


def q(xs,q):
    xs=[float(x) for x in xs if x is not None and np.isfinite(float(x))]
    return float(np.quantile(xs,q)) if xs else None


def role_macro(rows, getter):
    by=defaultdict(list)
    for r in rows:
        v=getter(r)
        if v is not None and np.isfinite(float(v)): by[r["target_role"]].append(float(v))
    vals={k:float(np.mean(v)) for k,v in by.items() if v}
    return mean(vals.values()),vals


def bootstrap_role_delta(a_by,b_by,seed=914,n=10000):
    roles=sorted(set(a_by)&set(b_by))
    if not roles: return None
    d=np.asarray([b_by[r]-a_by[r] for r in roles],dtype=float)
    rng=np.random.default_rng(seed); boots=[]
    for _ in range(n):
        idx=rng.integers(0,len(d),len(d)); boots.append(float(np.mean(d[idx])))
    return {"n_roles":len(roles),"mean_delta":float(np.mean(d)),"ci95":[float(np.quantile(boots,.025)),float(np.quantile(boots,.975))],"roles_improved":int(np.sum(d>0)),"roles_worse":int(np.sum(d<0))}


def probe_summary(rows):
    total=len(rows); ok=[r for r in rows if r.get("status")=="ok" and r.get("reward_breakdown")]
    strict=[r for r in ok if r["reward_breakdown"].get("strict_specific_pass")]
    delta=[r for r in ok if r["reward_breakdown"].get("strict_delta_pass")]
    roles=sorted({r["target_role"] for r in rows}); byrole=defaultdict(list)
    for r in ok: byrole[r["target_role"]].append(r)
    any_strict=sum(any(x["reward_breakdown"].get("strict_specific_pass") for x in byrole[role]) for role in roles)
    best_rob=[]; worst_conf=[]
    for role in roles:
        rr=byrole.get(role,[])
        if rr: best_rob.append(max(float(x["reward_breakdown"]["robust_specific"]) for x in rr))
        conf=defaultdict(list)
        for x in rr:
            for b in x.get("branches",[]): conf[str(b.get("hard_negative_replacement",""))].append(float(b.get("specific_pairwise",0.0)))
        if conf: worst_conf.append(min(float(np.mean(v)) for v in conf.values() if v))
    macro_rob,rob_by=role_macro(ok,lambda r:r["reward_breakdown"].get("robust_specific"))
    macro_q25,q25_by=role_macro(ok,lambda r:r["reward_breakdown"].get("specific_q25"))
    macro_act,act_by=role_macro(ok,lambda r:r["reward_breakdown"].get("activation_rate"))
    macro_pair,pair_by=role_macro(ok,lambda r:r["reward_breakdown"].get("positive_specific_pair_rate"))
    return {
        "generated":total,"evaluated":len(ok),"valid_eval_rate":len(ok)/max(1,total),
        "strict_specific_per_generated":len(strict)/max(1,total),"strict_specific_per_evaluated":len(strict)/max(1,len(ok)),
        "strict_delta_per_evaluated":len(delta)/max(1,len(ok)),"roles":len(roles),"roles_any_strict":int(any_strict),"roles_any_strict_rate":any_strict/max(1,len(roles)),
        "macro_robust_specific":macro_rob,"macro_specific_q25":macro_q25,"macro_activation":macro_act,"macro_positive_pair_rate":macro_pair,
        "macro_best_of_n_robust":mean(best_rob),"macro_worst_confuser_specific":mean(worst_conf),
        "_by_role":{"robust":rob_by,"q25":q25_by,"activation":act_by,"pair_rate":pair_by},
    }


def judge_summary(rows):
    return {
        "macro_clean_plus_minus_margin":role_macro(rows,lambda r:r.get("clean_plus_minus_mean_margin"))[0],
        "macro_clean_plus_minus_accuracy":role_macro(rows,lambda r:r.get("clean_plus_minus_accuracy"))[0],
        "macro_clean_sham_minus_margin":role_macro(rows,lambda r:r.get("clean_sham_minus_mean_margin"))[0],
        "macro_minus_target_fpr_at_0":role_macro(rows,lambda r:r.get("minus_target_fpr_at_0"))[0],
        "macro_minus_target_top3_rate":role_macro(rows,lambda r:r.get("minus_target_top3_rate"))[0],
        "macro_plus_selected_mean_target_rank":role_macro(rows,lambda r:r.get("plus_selected_mean_target_rank"))[0],
        "macro_hard_flip_accuracy":role_macro(rows,lambda r:r.get("hard_flip_accuracy_when_confuser_selected"))[0],
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--probe-eval",default="artifacts/v11_2_strong_eval/probe_eval.jsonl")
    ap.add_argument("--judge-eval",default="artifacts/v11_2_strong_eval/judge_eval.jsonl")
    ap.add_argument("--historical-v11",default="artifacts/v11_summary.json")
    ap.add_argument("--output",default="artifacts/v11_2_strong_eval/report.json")
    args=ap.parse_args()
    probes=load_jsonl(args.probe_eval); judges=load_jsonl(args.judge_eval)
    report={"version":"v11_2_strong_eval_v1","probe":{},"paired":{},"judge":{},"decision":{}}
    for split in sorted({r.get("eval_split") for r in probes}):
        report["probe"][split]={}
        for method in ("v11","v11_2"):
            rr=[r for r in probes if r.get("eval_split")==split and r.get("method")==method]
            report["probe"][split][method]=probe_summary(rr)
        a=report["probe"][split]["v11"]; b=report["probe"][split]["v11_2"]
        report["paired"][split]={
            "robust_specific":bootstrap_role_delta(a["_by_role"]["robust"],b["_by_role"]["robust"]),
            "specific_q25":bootstrap_role_delta(a["_by_role"]["q25"],b["_by_role"]["q25"]),
            "activation":bootstrap_role_delta(a["_by_role"]["activation"],b["_by_role"]["activation"]),
            "positive_pair_rate":bootstrap_role_delta(a["_by_role"]["pair_rate"],b["_by_role"]["pair_rate"]),
        }
        # remove internal maps from public summary after paired stats
        for method in ("v11","v11_2"): report["probe"][split][method].pop("_by_role",None)

    if judges:
        for method in ("v11","v11_2"):
            report["judge"][method]=judge_summary([r for r in judges if r.get("method")==method])

    hist={}
    p=Path(args.historical_v11)
    if p.exists():
        o=json.loads(p.read_text(encoding="utf-8")); hist={k:o.get(k) for k in ("train_strict_specific_rate","train_confirmed_elites","holdout_total","holdout_valid","holdout_role_any_strict_specific","holdout_role_any_strict_delta","holdout_role_any_activation_ge_075")}
    report["historical_v11_context_not_apples_to_apples"]=hist

    known=report.get("probe",{}).get("known_hard_locked",{})
    if known:
        a=known["v11"]; b=known["v11_2"]
        causal_win=(b.get("macro_robust_specific") or -99)>(a.get("macro_robust_specific") or -99)
        worst_win=(b.get("macro_worst_confuser_specific") or -99)>(a.get("macro_worst_confuser_specific") or -99)
        no_valid_reg=(b.get("valid_eval_rate",0)+.05)>=a.get("valid_eval_rate",0)
        no_role_reg=b.get("roles_any_strict",0)>=a.get("roles_any_strict",0)
        report["decision"]["generator_gate"]={"causal_macro_improved":causal_win,"worst_confuser_improved":worst_win,"valid_rate_not_worse_by_gt_5pp":no_valid_reg,"roles_any_strict_not_worse":no_role_reg,"pass_all":bool(causal_win and worst_win and no_valid_reg and no_role_reg)}
    if report["judge"]:
        a=report["judge"]["v11"]; b=report["judge"]["v11_2"]
        report["decision"]["downstream_judge_gate"]={
            "cf_margin_improved":(b.get("macro_clean_plus_minus_margin") or -99)>(a.get("macro_clean_plus_minus_margin") or -99),
            "absent_fpr_reduced":(b.get("macro_minus_target_fpr_at_0") if b.get("macro_minus_target_fpr_at_0") is not None else 99)<(a.get("macro_minus_target_fpr_at_0") if a.get("macro_minus_target_fpr_at_0") is not None else 99),
            "absent_top3_reduced":(b.get("macro_minus_target_top3_rate") if b.get("macro_minus_target_top3_rate") is not None else 99)<(a.get("macro_minus_target_top3_rate") if a.get("macro_minus_target_top3_rate") is not None else 99),
            "present_rank_improved":(b.get("macro_plus_selected_mean_target_rank") if b.get("macro_plus_selected_mean_target_rank") is not None else 99)<(a.get("macro_plus_selected_mean_target_rank") if a.get("macro_plus_selected_mean_target_rank") is not None else 99),
        }
    out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(report,indent=2,sort_keys=True,ensure_ascii=False),encoding="utf-8")
    print(json.dumps(report,indent=2,sort_keys=True,ensure_ascii=False),flush=True)
    print(f"[done] report={out}",flush=True)


if __name__=="__main__": main()
