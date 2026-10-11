from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from analysis.adaptive_fanout_count_v12 import (
    JsonlTargetCache,
    aggregate_results,
    cosine,
    entropy01,
    json_hash,
    norm_text,
    sample_probe_fast,
    set_metrics,
    sigmoid,
    stable_int,
    tree_sha256,
)
from v11.role_universe import ROLE_DESCRIPTIONS, TRAIN_ROLES, anchor_probe
from v13.fingerprint import BehavioralFingerprintDetectorV13, load_fingerprint_examples
from v13.hard_negative import (
    build_neighbor_map,
    composition_key,
    excluded_v12_compositions,
    make_fresh_fanout_specs,
)


def recent_majority(evidence:Sequence[float],positive:bool,n:int=3)->bool:
    if len(evidence)<n: return False
    tail=list(evidence)[-n:]
    return sum((x>0) if positive else (x<0) for x in tail)>=max(2,n-1)


def run_session(*,system_id:str,query_fn:Callable[[str],tuple[str,bool]],candidate_roles:Sequence[str],neighbor_map:dict[str,list[str]],detector,policy,gate,embedder,seed:int,min_probes:int,max_probes:int,max_total:int,confidence:float,evidence_discount:float,evidence_clip:float,temperature:float,top_p:float,generation_retries:int)->dict:
    states={r:{"n":0,"log_odds":0.0,"belief":0.5,"queries":[],"outputs":[],"scores":[],"evidence":[],"resolved":False,"decision":None,"exhausted":False,"generation_failures":0} for r in candidate_roles}
    history=[]; total=0

    def resolve(r):
        s=states[r]
        if s["n"]<min_probes: return
        b=float(s["belief"])
        if b>=confidence and recent_majority(s["evidence"],True,3): s["resolved"]=True; s["decision"]=True
        elif b<=1-confidence and recent_majority(s["evidence"],False,3): s["resolved"]=True; s["decision"]=False

    while total<max_total:
        need=[r for r in candidate_roles if not states[r]["exhausted"] and states[r]["n"]<min_probes]
        if need:
            role=min(need,key=lambda r:(states[r]["n"],candidate_roles.index(r)))
        else:
            unresolved=[r for r in candidate_roles if not states[r]["resolved"] and not states[r]["exhausted"] and states[r]["n"]<max_probes]
            if not unresolved: break
            def priority(r):
                own=entropy01(states[r]["belief"])
                neigh=max((entropy01(states[n]["belief"]) for n in neighbor_map.get(r,[])[:4]),default=0.0)
                return own+0.20*neigh-0.03*states[r]["n"]
            role=max(unresolved,key=priority)
        s=states[role]; valid=False; reason=""; raw=""; query=""
        # Dynamic competitors: nearest public roles that remain plausible/uncertain.
        comp=sorted(neighbor_map.get(role,[])[:6],key=lambda n:(states[n]["belief"],entropy01(states[n]["belief"])),reverse=True)[:4]
        for retry in range(max(1,generation_retries)):
            ph=[{"query":q,"output":o} for q,o in zip(s["queries"][-2:],s["outputs"][-2:])]
            obs={
                "candidate_role":role,"candidate_description":ROLE_DESCRIPTIONS[role],"anchor_probe":anchor_probe(role),
                "avoid_probes":s["queries"][-6:],
                "confusable_roles":[{"role":n,"description":ROLE_DESCRIPTIONS[n]} for n in comp],
                "public_history":ph,
            }
            query,valid,reason,raw=sample_probe_fast(policy,obs,seed=stable_int(seed,system_id,role,s["n"]+1,retry),temperature=temperature,top_p=top_p)
            if not valid or not query.strip(): continue
            if norm_text(query) in {norm_text(x) for x in s["queries"]}: valid=False; reason="duplicate"; continue
            if s["queries"]:
                qv=embedder.encode(query); ms=max(cosine(qv,embedder.encode(x)) for x in s["queries"])
                if ms>=.965: valid=False; reason=f"semantic_duplicate:{ms:.3f}"; continue
            g=gate.check(query)
            if not g.allowed: valid=False; reason=f"blocked:{getattr(g,'stage','gate')}:{getattr(g,'category','')}"; continue
            valid=True; break
        if not valid:
            s["generation_failures"]+=1
            if s["generation_failures"]>=3: s["exhausted"]=True
            history.append({"role":role,"event":"generation_failure","reason":reason,"raw":raw}); continue
        output,hit=query_fn(query); score,p=detector.score(role,query,output)
        ev=float(np.clip(score,-evidence_clip,evidence_clip))*float(evidence_discount)
        s["n"]+=1; s["queries"].append(query); s["outputs"].append(output); s["scores"].append(score); s["evidence"].append(ev)
        s["log_odds"]+=ev; s["belief"]=sigmoid(s["log_odds"]); total+=1; resolve(role)
        history.append({"step":total,"role":role,"competitors_public":comp,"query":query,"output":output,"cache_hit":hit,"detector_score":score,"probe_p":p,"evidence":ev,"belief_after":s["belief"],"resolved_after":s["resolved"]})
        print(f"[{system_id}] q={total:03d} role={role:<26} probe_p={p:.3f} belief={s['belief']:.3f} n={s['n']} resolved={s['resolved']} competitors={comp[:2]}",flush=True)

    for r,s in states.items():
        if s["decision"] is None: s["decision"]=bool(s["belief"]>=0.5)
    present=[r for r in candidate_roles if states[r]["decision"]]
    return {"system_id":system_id,"total_queries":total,"predicted_roles":present,"k_hat_workers":len(present),"unresolved_roles":[r for r in candidate_roles if not states[r]["resolved"]],"states":states,"history":history}


def main():
    ap=argparse.ArgumentParser(description="V13 end-to-end adaptive FANOUT role/K inference")
    ap.add_argument("--model-name",default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--adapter",default="artifacts/v13_policy")
    ap.add_argument("--v13-log",default="artifacts/v13_probe_disambiguation.jsonl")
    ap.add_argument("--v11-log",default="artifacts/v11_probe_discovery.jsonl")
    ap.add_argument("--output-dir",default="artifacts/v13_fanout_count")
    ap.add_argument("--train-systems-per-cell",type=int,default=4)
    ap.add_argument("--training-seed",type=int,default=1313)
    ap.add_argument("--systems-per-cell",type=int,default=1)
    ap.add_argument("--seed",type=int,default=1314)
    ap.add_argument("--neighbor-k",type=int,default=5)
    ap.add_argument("--min-probes-per-role",type=int,default=3)
    ap.add_argument("--max-probes-per-role",type=int,default=7)
    ap.add_argument("--max-total-queries",type=int,default=120)
    ap.add_argument("--confidence",type=float,default=.90)
    ap.add_argument("--evidence-discount",type=float,default=.45)
    ap.add_argument("--evidence-clip",type=float,default=1.5)
    ap.add_argument("--temperature",type=float,default=.75)
    ap.add_argument("--top-p",type=float,default=.90)
    ap.add_argument("--generation-retries",type=int,default=6)
    ap.add_argument("--detector-c",type=float,default=.05)
    ap.add_argument("--detector-folds",type=int,default=5)
    ap.add_argument("--gate-mode",choices=["hard","hybrid"],default="hard")
    ap.add_argument("--guard-model",default="gpt-4o-mini")
    ap.add_argument("--no-4bit",action="store_true")
    ap.add_argument("--execute",action="store_true")
    args=ap.parse_args()

    # Reconstruct exclusions: V11 + already-inspected V12 + V13 training compositions.
    excluded=excluded_v12_compositions(TRAIN_ROLES)
    v13train=make_fresh_fanout_specs(TRAIN_ROLES,prefix="v13_train",systems_per_cell=args.train_systems_per_cell,seed=args.training_seed,excluded_compositions=excluded)
    excluded |= {composition_key(s.worker_roles) for s in v13train}
    specs=make_fresh_fanout_specs(TRAIN_ROLES,prefix="v13_eval",systems_per_cell=args.systems_per_cell,seed=args.seed,excluded_compositions=excluded)
    specs = specs[:1]  # PILOT: exactly one fresh system
    manifest=[{"system_id":s.system_id,"mode":s.execution_mode,"k_true_workers_eval_only":len(s.worker_roles),"true_roles_eval_only":list(s.worker_roles)} for s in specs]
    examples=load_fingerprint_examples([(args.v13_log,1.0),(args.v11_log,.35)],set(TRAIN_ROLES))
    config={"adapter_sha256":tree_sha256(args.adapter),"manifest_sha256":json_hash(manifest),"training_seed":args.training_seed,"seed":args.seed,"systems_per_cell":args.systems_per_cell,"min_probes":args.min_probes_per_role,"max_probes":args.max_probes_per_role,"max_total":args.max_total_queries,"confidence":args.confidence,"evidence_discount":args.evidence_discount,"evidence_clip":args.evidence_clip}
    run_hash=json_hash(config)
    print("=== V13 ADAPTIVE BEHAVIORAL-FINGERPRINT FANOUT INFERENCE ===")
    print("frozen V13 adapter:",args.adapter)
    print("adapter sha256:",config["adapter_sha256"])
    print("detector examples:",len(examples),"(V13 hard negatives weight 1.0; V11 breadth weight .35)")
    print("fresh eval systems:",len(specs),"fanout2/fanout3 x K3..6")
    print("manifest sha256:",config["manifest_sha256"])
    print("NO eval truth/K/routing/trace enters policy, detector, controller, or stopping rule.")
    for s in specs: print(f"  EVAL-ONLY {s.system_id}: mode={s.execution_mode} K={len(s.worker_roles)} roles={list(s.worker_roles)}")
    if not args.execute:
        print("PLAN ONLY. Add --execute to run."); return
    if not os.environ.get("OPENAI_API_KEY"): raise SystemExit("OPENAI_API_KEY is not set")
    if not Path(args.adapter).exists(): raise SystemExit(f"V13 adapter missing: {args.adapter}")

    import torch
    if not torch.cuda.is_available(): raise SystemExit("CUDA required")
    from v8.embedding import FrozenOutputEmbedderV8
    from v10.gate import BenignGateV10
    from targets.single_router_fanout_mas import SingleRouterFanoutMAS
    from v13.policy import DisambiguationProbePolicyV13

    outdir=Path(args.output_dir); outdir.mkdir(parents=True,exist_ok=True)
    cfg=outdir/"run_config.json"
    if cfg.exists():
        old=json.loads(cfg.read_text());
        if old.get("run_hash")!=run_hash: raise SystemExit("Existing V13 eval dir has different config; archive it or use another --output-dir")
    else: cfg.write_text(json.dumps({**config,"run_hash":run_hash},indent=2))
    (outdir/"test_manifest_eval_only.json").write_text(json.dumps(manifest,indent=2))

    print("\n=== FIT V13 BEHAVIORAL FINGERPRINT DETECTOR (TRAIN DATA ONLY) ===",flush=True)
    embedder=FrozenOutputEmbedderV8(device="cpu")
    neighbor_map=build_neighbor_map(embedder,TRAIN_ROLES,top_k=args.neighbor_k)
    detector=BehavioralFingerprintDetectorV13(embedder,ROLE_DESCRIPTIONS,c=args.detector_c)
    report=detector.fit(examples,TRAIN_ROLES,folds=args.detector_folds)
    (outdir/"detector_oof.json").write_text(json.dumps(report,indent=2))
    print(f"detector macro grouped-OOF AUC={report['macro_oof_auc']:.3f} BA={report['macro_oof_balanced_accuracy']:.3f}",flush=True)

    gate=BenignGateV10(mode=args.gate_mode,guard_model=args.guard_model)
    policy=DisambiguationProbePolicyV13(args.model_name,adapter_path=args.adapter,trainable=False,use_4bit=not args.no_4bit,max_seq_length=1536)
    cache=JsonlTargetCache(outdir/"target_public_cache.jsonl")
    sessions_path=outdir/"sessions.jsonl"; completed={}
    if sessions_path.exists():
        for line in sessions_path.open():
            if not line.strip(): continue
            try:
                r=json.loads(line)
                if r.get("run_hash")==run_hash and r.get("complete"): completed[r["system_id"]]=r
            except Exception: pass
    results=[]
    for i,spec in enumerate(specs,1):
        if spec.system_id in completed:
            print(f"=== SYSTEM {i}/{len(specs)} {spec.system_id}: resume-skip ===",flush=True); results.append(completed[spec.system_id]); continue
        print(f"\n=== SYSTEM {i}/{len(specs)} {spec.system_id} ===",flush=True)
        mas=SingleRouterFanoutMAS(spec)
        def public_query(q,_sid=spec.system_id,_mas=mas): return cache.run(_sid,q,_mas.query)
        sess=run_session(system_id=spec.system_id,query_fn=public_query,candidate_roles=TRAIN_ROLES,neighbor_map=neighbor_map,detector=detector,policy=policy,gate=gate,embedder=embedder,seed=args.seed,min_probes=args.min_probes_per_role,max_probes=args.max_probes_per_role,max_total=args.max_total_queries,confidence=args.confidence,evidence_discount=args.evidence_discount,evidence_clip=args.evidence_clip,temperature=args.temperature,top_p=args.top_p,generation_retries=args.generation_retries)
        # Truth is attached only AFTER stopping.
        true=set(spec.worker_roles); pred=set(sess["predicted_roles"])
        sess.update({"complete":True,"run_hash":run_hash,"execution_mode_eval_only":spec.execution_mode,"true_roles_eval_only":sorted(true),"k_true_workers_eval_only":len(true),"k_hat_total_nodes":sess["k_hat_workers"]+2,"role_metrics_eval_only":set_metrics(true,pred)})
        with sessions_path.open("a") as f: f.write(json.dumps(sess,ensure_ascii=False)+"\n"); f.flush()
        results.append(sess)
        print(f"RESULT {spec.system_id}: K_true={len(true)} K_hat={sess['k_hat_workers']} true={sorted(true)} pred={sorted(pred)} queries={sess['total_queries']} unresolved={len(sess['unresolved_roles'])}",flush=True)

    summary={"protocol":"v13_hard_negative_policy_plus_behavioral_residual_fingerprint_detector","development_test":True,"not_final_untouched_paper_test":True,"run_hash":run_hash,"adapter_sha256":config["adapter_sha256"],"manifest_sha256":config["manifest_sha256"],"detector_oof":report,"neighbor_map_public":neighbor_map,"metrics":aggregate_results(results,TRAIN_ROLES),"systems":[{"system_id":r["system_id"],"mode":r["execution_mode_eval_only"],"k_true":r["k_true_workers_eval_only"],"k_hat":r["k_hat_workers"],"queries":r["total_queries"],"unresolved":len(r["unresolved_roles"]),"role_f1":r["role_metrics_eval_only"]["f1"]} for r in results]}
    (outdir/"summary.json").write_text(json.dumps(summary,indent=2))
    m=summary["metrics"]
    print("\n"+"="*80); print("V13 FANOUT COUNT DEVELOPMENT SUMMARY"); print("="*80)
    print(f"K exact accuracy      : {m['k_exact_accuracy']:.3f}")
    print(f"K MAE                 : {m['k_mae']:.3f}")
    print(f"K within-1 accuracy   : {m['k_within_1_accuracy']:.3f}")
    print(f"Exact role-set acc    : {m['exact_role_set_accuracy']:.3f}")
    print(f"Mean role-set Jaccard : {m['mean_role_set_jaccard']:.3f}")
    print(f"Micro role F1         : {m['micro_role_f1']:.3f}")
    print(f"Mean queries/system   : {m['mean_queries']:.1f}")
    print("summary:",outdir/"summary.json")


if __name__=="__main__": main()
