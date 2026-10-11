#!/usr/bin/env python3
"""
eval_robust_visible_generator_v1.py

Held-out TEST-system generator-distribution evaluation.

Compares three policies on identical round-1 public belief states:
  A) stage10 cycle01 branch point
  B) existing final stage10 cycle02 policy
  C) robust visible-MI branch

No MAS calls and no actual output are used here.
Each generated query is scored BEFORE execution by expected visible MI about K.

Primary question:
  Did training lift the whole generator distribution enough that RANDOM / FIRST
  candidate from the new generator is strong, reducing the need for a selector?
"""

from __future__ import annotations
import argparse, gc, json, random, sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from adaptive_v11_active.reward import surface_valid
from adaptive_v11_active.schema import PublicBeliefState
from adaptive_v11_active.io import stable_int
from adaptive_v11_curriculum.policy import CurriculumAdaptiveProbePolicyV11

sys.path.insert(0,str(Path("analysis").resolve()))
import observation_model_v1 as om
import visible_adaptive_bridge_v1 as vb


def pct(xs,p):
    return float(np.quantile(np.asarray(xs,float),p)) if xs else None


def summarize(rows):
    vals=[r["visible_mi_k"] for r in rows if r["valid"]]
    by=defaultdict(list)
    for r in rows:
        if r["valid"]: by[r["system_id"]].append(r)
    first=[]; gmean=[]; gworst=[]; gbest=[]; gp25=[]
    unique_rates=[]
    for sid,rs in by.items():
        rs=sorted(rs,key=lambda z:z["candidate_index"])
        vs=[z["visible_mi_k"] for z in rs]
        if not vs: continue
        first.append(vs[0]); gmean.append(float(np.mean(vs)))
        gworst.append(float(np.min(vs))); gbest.append(float(np.max(vs)))
        gp25.append(float(np.quantile(vs,.25)))
        unique_rates.append(len(set(z["query"].strip().lower() for z in rs))/len(rs))
    return {
        "valid_rate":float(np.mean([r["valid"] for r in rows])),
        "all_candidate_mean":float(np.mean(vals)) if vals else None,
        "all_candidate_p10":pct(vals,.10),
        "all_candidate_p25":pct(vals,.25),
        "all_candidate_median":pct(vals,.50),
        "all_candidate_p75":pct(vals,.75),
        "random_candidate_expected_mean":float(np.mean(gmean)) if gmean else None,
        "candidate0_mean":float(np.mean(first)) if first else None,
        "mean_group_worst":float(np.mean(gworst)) if gworst else None,
        "mean_group_p25":float(np.mean(gp25)) if gp25 else None,
        "mean_group_best_of_8":float(np.mean(gbest)) if gbest else None,
        "mean_unique_rate":float(np.mean(unique_rates)) if unique_rates else None,
        "systems":len(by),
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",default="artifacts/adaptive_v11_curriculum_v2")
    ap.add_argument("--obs-dir",default="artifacts/observation_model_v1")
    ap.add_argument("--vis-dir",default="artifacts/visible_output_fingerprint_v1")
    ap.add_argument("--cal-dir",default="artifacts/calibrated_bayes_diag_v1")
    ap.add_argument("--branch-policy",default="artifacts/adaptive_v11_curriculum_v2/checkpoints/stage_10_cycle_01/policy")
    ap.add_argument("--existing-final",default="artifacts/adaptive_v11_curriculum_v2/checkpoints/stage_10_cycle_02/policy")
    ap.add_argument("--robust-policy",default="artifacts/robust_visible_generator_v1/policy")
    ap.add_argument("--out",default="artifacts/robust_visible_generator_v1/eval")
    ap.add_argument("--systems",type=int,default=24)
    ap.add_argument("--candidates",type=int,default=8)
    ap.add_argument("--particles-per-k",type=int,default=16)
    ap.add_argument("--seed",type=int,default=103)
    a=ap.parse_args()

    root=Path(a.root); obsdir=Path(a.obs_dir); visdir=Path(a.vis_dir)
    caldir=Path(a.cal_dir); out=Path(a.out); out.mkdir(parents=True,exist_ok=True)

    examples,roles,_,_=om.collect_examples(root)
    split=json.loads((obsdir/"split_systems.json").read_text())
    test_systems=set(map(str,split["test"]))
    train_systems=set(map(str,split["train"]))
    val_systems=set(map(str,split["val"]))

    # One clean ROUND-1 state per held-out system. No truth is used for selection.
    state_map={}
    for x in examples:
        sid=str(x["system_id"])
        if sid in test_systems and int(x["round"])==1 and sid not in state_map:
            state_map[sid]=x
    sids=sorted(state_map)
    rng=random.Random(a.seed); rng.shuffle(sids); sids=sids[:a.systems]
    if not sids: raise RuntimeError("No held-out round1 systems")

    # Frozen visible channel.
    Qbase=np.load(obsdir/"query_embeddings.npy").astype(np.float32)
    Obase=np.load(visdir/"output_embeddings.npy").astype(np.float32)
    Xbase=np.concatenate([Qbase,Obase],axis=1)
    tri=np.array([i for i,x in enumerate(examples) if str(x["system_id"]) in train_systems],int)
    vai=np.array([i for i,x in enumerate(examples) if str(x["system_id"]) in val_systems],int)
    Y=vb.labels(examples,roles,"selected_roles")
    fp_models,fp_cals,thr,tpr,fpr,_=vb.fit_visible_detector(Xbase,Y,tri,vai)

    # Forward observation model.
    ckpt=torch.load(obsdir/"model.pt",map_location="cpu")
    fwd=om.ObservationNet(
        qdim=ckpt["query_embedding_dim"],n_roles=len(roles),
        hidden=ckpt["hidden"],dropout=ckpt["dropout"]
    )
    fwd.load_state_dict(ckpt["state_dict"]); fwd.eval()
    ri={r:i for i,r in enumerate(roles)}
    pos=np.zeros(len(roles),float)
    for i in tri:
        for r in examples[i]["selected_roles"]: pos[ri[r]]+=1
    posw=np.clip((len(tri)-pos)/np.maximum(pos,1.0),1.0,20.0)
    log_posw=np.log(posw)
    cs=json.loads((caldir/"summary.json").read_text())
    T=float(cs["true_Z_probability_calibration"]["corrected_plus_temperature"]["best_T"])

    policies=[
        ("branch_cycle01",a.branch_policy),
        ("existing_final_cycle02",a.existing_final),
        ("robust_visible",a.robust_policy),
    ]
    generated=[]

    # Generate sequentially to stay memory-safe.
    for pname,ppath in policies:
        print(f"[generate] loading {pname}: {ppath}",flush=True)
        pol=CurriculumAdaptiveProbePolicyV11(
            "Qwen/Qwen2.5-1.5B-Instruct",
            adapter_path=ppath,trainable=False,use_4bit=False,max_seq_length=4096
        )
        for si,sid in enumerate(sids):
            x=state_map[sid]
            st=PublicBeliefState(
                session_id=sid,split="dev",round_index=0,
                k_probs={str(k):float(v) for k,v in x["k_probs"].items()},
                role_probs={r:float(x["role_probs"][r]) for r in roles},
                history=[]
            )
            seen=[]
            for j in range(a.candidates):
                sample=None; q=""; valid=False; why=""
                for retry in range(8):
                    seed=stable_int(a.seed,pname,sid,j,retry)
                    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
                    obs={"belief_state":st.to_public_dict(),"candidate_index":j,"max_history_turns":5}
                    sample=pol.sample_probe(obs,temperature=.85,top_p=.95,max_new_tokens=192)
                    q=str(sample.query or "").strip()
                    ok,why=surface_valid(q)
                    valid=bool(sample.valid) and ok and q not in seen
                    if valid: break
                if valid: seen.append(q)
                generated.append({
                    "policy":pname,"system_id":sid,"candidate_index":j,
                    "query":q,"valid":bool(valid),"invalid_reason":"" if valid else str(why),
                })
            if (si+1)%6==0: print(f"  {pname}: {si+1}/{len(sids)} systems",flush=True)
        del pol; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # Embed ONLY generated queries; this is pre-execution.
    enc=SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2",device="cpu")
    qtexts=[r["query"] if r["valid"] else "" for r in generated]
    Q=enc.encode(qtexts,batch_size=64,show_progress_bar=True,convert_to_numpy=True).astype(np.float32)

    # Score groups.
    max_round=max(int(x["round"]) for x in examples)
    loc=defaultdict(list)
    for i,r in enumerate(generated): loc[(r["policy"],r["system_id"])].append(i)

    for gi,((pname,sid),ids) in enumerate(loc.items()):
        x=state_map[sid]
        belief=np.array([x["role_probs"][r] for r in roles],dtype=np.float32)
        kvec=np.array([x["k_probs"].get(str(k),0.0) for k in [3,4,5,6]],dtype=np.float32)
        rnorm=float(1/max(max_round,1))
        prng=np.random.default_rng(abs(hash((sid,a.seed)))%(2**32))
        Z,W,Klabels,Kprior=vb.stratified_particles(
            x["k_probs"],x["role_probs"],roles,a.particles_per_k,prng
        )
        valid_ids=[i for i in ids if generated[i]["valid"]]
        if not valid_ids: continue
        qg=np.stack([Q[i] for i in valid_ids]).astype(np.float32)
        L=vb.particle_logits(fwd,qg,belief,kvec,rnorm,Z,"cpu",4096)
        pY=vb.calibrated_hidden_probs(L,log_posw,T)
        pE=vb.detector_channel_probs(pY,tpr,fpr)
        mi=vb.mi_k(pE,W,Klabels,Kprior)
        for ii,m in zip(valid_ids,mi): generated[ii]["visible_mi_k"]=float(m)
        for ii in ids:
            if "visible_mi_k" not in generated[ii]: generated[ii]["visible_mi_k"]=0.0

    summ={}
    for pname,_ in policies:
        summ[pname]=summarize([r for r in generated if r["policy"]==pname])

    base=summ["existing_final_cycle02"]; rob=summ["robust_visible"]
    comparison={}
    for k in ["all_candidate_mean","all_candidate_p10","all_candidate_p25","all_candidate_median",
              "random_candidate_expected_mean","candidate0_mean","mean_group_worst","mean_group_p25",
              "mean_group_best_of_8"]:
        b=base.get(k); r=rob.get(k)
        comparison[k]={"existing_final":b,"robust":r,"delta":None if b is None or r is None else float(r-b),
                       "ratio":None if b in (None,0) or r is None else float(r/b)}

    summary={
        "setting":{
            "heldout_test_systems":len(sids),
            "candidates_per_system":a.candidates,
            "actual_MAS_calls":0,
            "actual_candidate_outputs_used":False,
            "score":"expected visible MI about K before execution",
            "primary_no_selector_metric":"random_candidate_expected_mean + candidate0_mean + lower-tail quantiles",
        },
        "policies":summ,
        "robust_vs_existing_final":comparison,
        "success_criterion":{
            "desired":"robust generator raises mean AND p10/p25/worst, so random/first candidate becomes strong",
            "strong_result":"robust random/first approaches or exceeds existing-final best-of-8 surrogate quality",
        }
    }
    (out/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    with (out/"generated_scored.jsonl").open("w",encoding="utf-8") as f:
        for r in generated: f.write(json.dumps(r,ensure_ascii=False)+"\n")

    print("\n"+"="*100)
    print("ROBUST VISIBLE GENERATOR V1 — HELD-OUT PRE-EXECUTION EVAL")
    print("="*100)
    for p in summ:
        print("\n",p)
        for k,v in summ[p].items(): print(f"{k:34s}: {v}")
    print("\nROBUST vs EXISTING FINAL")
    for k,v in comparison.items(): print(f"{k:34s}: {v}")
    print("\n[saved]",out/"summary.json")


if __name__=="__main__":
    main()
