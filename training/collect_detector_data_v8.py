from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from estimator.candidate_presence_contrastive_v64 import load_profile
from probes.role_probe_bank import primary_probe
from rl.generative_probe_env_v7 import ResponseCacheV7
from targets.single_router_zoo_v65 import get_v65_split
from v8.counterfactual import get_mode, get_roles, split_base_specs
from v8.execution import execute_cached
from v8.gate import BenignGateV8
from v8.policy import GroupProbePolicyV8


def main():
    ap=argparse.ArgumentParser(description="Collect black-box transcripts for the post-RL V8 PRESENT/ABSENT detector")
    ap.add_argument("--model-name",default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--adapter",default="artifacts/v8_grpo_policy")
    ap.add_argument("--profile",default="config/application_profile.json")
    ap.add_argument("--seed",type=int,default=808)
    ap.add_argument("--gate-mode",choices=["hard","hybrid"],default="hybrid")
    ap.add_argument("--guard-model",default="gpt-4o-mini")
    ap.add_argument("--output-public",default="artifacts/v8_detector_train_public.jsonl")
    ap.add_argument("--output-truth",default="artifacts/v8_detector_train_truth.jsonl")
    ap.add_argument("--response-public-cache",default="artifacts/v8_detector_response_public.jsonl")
    ap.add_argument("--response-truth-cache",default="artifacts/v8_detector_response_truth.jsonl")
    ap.add_argument("--execute",action="store_true")
    args=ap.parse_args()
    profile=load_profile(args.profile); roles=tuple(profile["candidate_roles"])
    _, specs=split_base_specs(list(get_v65_split("train")),seed=args.seed)
    interactions=len(specs)*len(roles)*2
    print("=== V8 DETECTOR-DATA COLLECTION PLAN ===")
    print("reserved base systems:",len(specs),"roles:",len(roles))
    print("external interactions: up to",interactions,"(seed + one trained follow-up per candidate)")
    print("RL training systems are excluded by base-system split")
    if not args.execute:
        print("PLAN ONLY. Re-run with --execute.");return
    if not os.environ.get("OPENAI_API_KEY"): raise SystemExit("OPENAI_API_KEY is not set")
    if not Path(args.adapter).exists(): raise SystemExit(f"missing adapter: {args.adapter}")
    policy=GroupProbePolicyV8(args.model_name,adapter_path=args.adapter,trainable=False,use_4bit=True,max_seq_length=1408)
    gate=BenignGateV8(mode=args.gate_mode,guard_model=args.guard_model)
    cache=ResponseCacheV7(args.response_public_cache,args.response_truth_cache)
    pubp=Path(args.output_public); truthp=Path(args.output_truth); pubp.parent.mkdir(parents=True,exist_ok=True)
    pf=pubp.open("w",encoding="utf-8"); tf=truthp.open("w",encoding="utf-8")
    for si,spec in enumerate(specs,1):
        true=set(get_roles(spec))
        for role in roles:
            seed_q=primary_probe(role); seed_res=execute_cached(spec,seed_q,cache)
            obs={"candidate_role":role,"candidate_description":profile.get("role_descriptions",{}).get(role,role),"seed_query":seed_q,"seed_output":seed_res.output}
            sample=policy.sample_question(obs,deterministic=True)
            follow_q=sample.query if sample.valid else ""
            follow_out=""; allowed=False; gate_reason=sample.invalid_reason
            if sample.valid:
                g=gate.check(follow_q); allowed=bool(g.allowed); gate_reason=g.reason
                if allowed: follow_out=execute_cached(spec,follow_q,cache).output
            pf.write(json.dumps({"system_id":getattr(spec,"system_id",""),"role":role,"seed_query":seed_q,"seed_output":seed_res.output,"follow_query":follow_q,"follow_output":follow_out,"follow_allowed":allowed,"gate_reason":gate_reason},ensure_ascii=False)+"\n")
            tf.write(json.dumps({"system_id":getattr(spec,"system_id",""),"role":role,"label":int(role in true),"mode":get_mode(spec),"K":len(true)},ensure_ascii=False)+"\n")
        print(f"[{si}/{len(specs)}] {getattr(spec,'system_id','')}",flush=True)
    pf.close();tf.close();print("saved:",pubp,truthp)
if __name__=="__main__":main()
