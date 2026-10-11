from __future__ import annotations
import argparse, gc, math
from pathlib import Path
import joblib
import torch
from adaptive_v11_active.io import read_jsonl, stable_int, write_jsonl
from adaptive_v11_active.reward import surface_valid
from adaptive_v11_active.schema import PublicBeliefState, assert_policy_safe
from v10.policy import SeededProbePolicyV10
from v11.role_universe import ROLE_DESCRIPTIONS, TRAIN_ROLES, anchor_probe
from scripts.av11_query_utility import h01, predict_one

def static_order(sid,seed): return sorted(TRAIN_ROLES,key=lambda r:stable_int(seed,"static",sid,r))
def prev_targets(root,split,rnd,sid):
    out=[]
    for rr in range(1,rnd):
        p=root/f"candidates_{split}_round_{rr:02d}.jsonl"
        if p.exists():
            for x in read_jsonl(p):
                if str(x.get("session_id"))==sid:
                    if x.get("target_role"): out.append(str(x["target_role"]))
                    break
    return out

def gen_one(policy,state,target,args,retry,previous_queries):
    seed=stable_int(args.seed,args.split,args.round,state.session_id,target,retry)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    obs={"candidate_role":target,"candidate_description":ROLE_DESCRIPTIONS[target],"anchor_probe":anchor_probe(target),"avoid_probes":previous_queries[-6:]}
    s=policy.sample_probe(obs,temperature=args.temperature,top_p=args.top_p,max_new_tokens=args.max_new_tokens)
    q=str(getattr(s,"query","") or "").strip(); ok,why=surface_valid(q)
    valid=bool(getattr(s,"valid",bool(q))) and ok and q not in set(previous_queries)
    return {"query":q,"valid":valid,"invalid_reason":"" if valid else (str(getattr(s,"invalid_reason","")) or why or "duplicate"),"prompt":str(getattr(s,"prompt","")),"completion":str(getattr(s,"completion","")),"raw_completion":str(getattr(s,"raw_completion",getattr(s,"completion",""))),"seed":int(seed),"local_retry":retry}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--round",type=int,required=True); ap.add_argument("--split",choices=["train","dev"],required=True); ap.add_argument("--root",required=True)
    ap.add_argument("--controller",choices=["common","static","maxent","utility"],required=True)
    ap.add_argument("--utility-model",default="artifacts/adaptive_v11_query_eig_v2/query_utility.joblib")
    ap.add_argument("--adapter",default="artifacts/v11_policy"); ap.add_argument("--model-name",default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--candidate-pool",type=int,default=8); ap.add_argument("--temperature",type=float,default=.85); ap.add_argument("--top-p",type=float,default=.95)
    ap.add_argument("--max-new-tokens",type=int,default=192); ap.add_argument("--max-seq-length",type=int,default=4096); ap.add_argument("--local-retries",type=int,default=20); ap.add_argument("--seed",type=int,default=917611); ap.add_argument("--no-4bit",action="store_true")
    args=ap.parse_args()
    if not torch.cuda.is_available(): raise SystemExit("CUDA required")
    root=Path(args.root); states=read_jsonl(root/f"states_{args.split}_round_{args.round-1:02d}.jsonl")
    policy=SeededProbePolicyV10(args.model_name,adapter_path=args.adapter,trainable=False,use_4bit=not args.no_4bit,max_seq_length=args.max_seq_length)
    ub=joblib.load(args.utility_model) if args.controller=="utility" else None
    rows=[]; pools=[]
    for i,raw in enumerate(states):
        assert_policy_safe(raw); st=PublicBeliefState.from_public_dict(raw); st.validate(TRAIN_ROLES,sorted(st.k_probs))
        tested=prev_targets(root,args.split,args.round,st.session_id); order=static_order(st.session_id,args.seed); rank={r:i for i,r in enumerate(order)}
        previous_queries=[str(h.query).strip() for h in st.history if str(h.query).strip()]
        if args.controller=="common": roles=[order[0]]
        elif args.controller=="static": roles=[order[(args.round-1)%len(order)]]
        else:
            untested=[r for r in TRAIN_ROLES if r not in set(tested)] or list(TRAIN_ROLES)
            untested=sorted(untested,key=lambda r:(-h01(st.role_probs[r]),rank[r]))
            roles=untested[: (1 if args.controller=="maxent" else max(2,args.candidate_pool))]
        cand=[]
        for role in roles:
            best=None
            for retry in range(max(1,args.local_retries)):
                z=gen_one(policy,st,role,args,retry,previous_queries)
                best=z
                if z["valid"]: break
            score=None
            if best and best["valid"] and ub is not None:
                score=predict_one(ub,raw,best["query"],role,args.round)
            rec={"session_id":st.session_id,"round":args.round,"controller":args.controller,"target_role":role,"target_role_probability_before":float(st.role_probs[role]),"target_role_entropy_before":h01(st.role_probs[role]),"predicted_entropy_gain":score,**(best or {})}
            cand.append(rec)
        valid=[x for x in cand if x.get("valid")]
        if not valid: selected=cand[0]
        elif args.controller=="utility": selected=max(valid,key=lambda x:float(x.get("predicted_entropy_gain") if x.get("predicted_entropy_gain") is not None else -1e9))
        else: selected=valid[0]
        out=dict(selected); out.update({"split":args.split,"candidate_index":0,"group_id":f"{st.session_id}::r{args.round:02d}","previous_target_roles":tested,"adapter":args.adapter,"phase":"query_utility_controller_v2","pool_size":len(cand)})
        rows.append(out)
        for j,x in enumerate(cand): pools.append({**x,"split":args.split,"pool_candidate_index":j,"selected":x is selected})
        print(f"[eig-gen] {i+1}/{len(states)} mode={args.controller} r={args.round} sid={st.session_id} target={out.get('target_role')} predIG={out.get('predicted_entropy_gain')} valid={out.get('valid')}",flush=True)
    write_jsonl(root/f"candidates_{args.split}_round_{args.round:02d}.jsonl",rows)
    write_jsonl(root/f"candidate_pool_{args.split}_round_{args.round:02d}.jsonl",pools)
    print(f"[done] rows={len(rows)} valid={sum(int(x.get('valid',False)) for x in rows)} pool_rows={len(pools)}",flush=True)
    del policy; gc.collect(); torch.cuda.empty_cache()
if __name__=="__main__": main()
