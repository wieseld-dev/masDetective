from __future__ import annotations
import argparse,json,os
from collections import defaultdict
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score,balanced_accuracy_score,f1_score,brier_score_loss
from estimator.candidate_presence_contrastive_v64 import load_profile
from estimator.transcript_detector_v8 import TranscriptDetectorV8
from probes.role_probe_bank import primary_probe
from rl.generative_probe_env_v7 import ResponseCacheV7
from targets.single_router_zoo_v65 import get_v65_split
from v8.counterfactual import get_roles,get_mode
from v8.execution import execute_cached
from v8.gate import BenignGateV8
from v8.policy import GroupProbePolicyV8

def main():
 ap=argparse.ArgumentParser();ap.add_argument("--model-name",default="Qwen/Qwen2.5-1.5B-Instruct");ap.add_argument("--adapter",default="artifacts/v8_grpo_policy");ap.add_argument("--detector",default="artifacts/v8_transcript_detector.joblib");ap.add_argument("--profile",default="config/application_profile.json");ap.add_argument("--gate-mode",choices=["hard","hybrid"],default="hybrid");ap.add_argument("--guard-model",default="gpt-4o-mini");ap.add_argument("--limit-systems",type=int,default=0);ap.add_argument("--execute",action="store_true");ap.add_argument("--output",default="artifacts/v8_final_eval.json");args=ap.parse_args()
 profile=load_profile(args.profile);roles=tuple(profile['candidate_roles']);specs=list(get_v65_split('val'));specs=specs[:args.limit_systems] if args.limit_systems>0 else specs
 print("=== V8 FINAL BLACK-BOX EVALUATION PLAN ===");print("unseen V6.5 validation systems:",len(specs),"roles/system:",len(roles));print("interactions upper bound:",len(specs)*len(roles)*2);print("detector was trained AFTER RL and is not part of RL reward")
 if not args.execute: print("PLAN ONLY. Re-run with --execute.");return
 if not os.environ.get('OPENAI_API_KEY'): raise SystemExit('OPENAI_API_KEY is not set')
 policy=GroupProbePolicyV8(args.model_name,adapter_path=args.adapter,trainable=False,use_4bit=True,max_seq_length=1408);det=TranscriptDetectorV8.load(args.detector);gate=BenignGateV8(mode=args.gate_mode,guard_model=args.guard_model);cache=ResponseCacheV7('artifacts/v8_final_response_public.jsonl','artifacts/v8_final_response_truth.jsonl')
 rows=[];y=[];meta=[]
 for si,spec in enumerate(specs,1):
  truth=set(get_roles(spec))
  for role in roles:
   sq=primary_probe(role);sr=execute_cached(spec,sq,cache);obs={'candidate_role':role,'candidate_description':profile.get('role_descriptions',{}).get(role,role),'seed_query':sq,'seed_output':sr.output};sample=policy.sample_question(obs,deterministic=True);fq=sample.query if sample.valid else '';fo='';allowed=False
   if sample.valid:
    gr=gate.check(fq);allowed=bool(gr.allowed)
    if allowed:fo=execute_cached(spec,fq,cache).output
   rows.append({'system_id':getattr(spec,'system_id',''),'role':role,'seed_query':sq,'seed_output':sr.output,'follow_query':fq,'follow_output':fo,'follow_allowed':allowed});y.append(int(role in truth));meta.append((getattr(spec,'system_id',''),role,len(truth)))
  print(f"[{si}/{len(specs)}] {getattr(spec,'system_id','')}",flush=True)
 p=det.predict_proba(rows);pred=p>=.5;overall={'auc':float(roc_auc_score(y,p)),'balanced_accuracy':float(balanced_accuracy_score(y,pred)),'f1':float(f1_score(y,pred)),'brier':float(brier_score_loss(y,p))}
 by=defaultdict(list)
 for (sid,role,k),pp,pr in zip(meta,p,pred):by[sid].append((k,float(pp),bool(pr)))
 hard=[];soft=[];exact=[];within=[]
 for vals in by.values():
  k=vals[0][0];kh=sum(int(x[2]) for x in vals);ks=sum(x[1] for x in vals);hard.append(abs(kh-k));soft.append(abs(ks-k));exact.append(kh==k);within.append(abs(kh-k)<=1)
 overall.update({'hard_K_MAE':float(np.mean(hard)),'soft_K_MAE':float(np.mean(soft)),'exact_K_accuracy':float(np.mean(exact)),'within_1_K_accuracy':float(np.mean(within)),'follow_allowed_rate':float(np.mean([r['follow_allowed'] for r in rows]))})
 Path(args.output).parent.mkdir(parents=True,exist_ok=True);Path(args.output).write_text(json.dumps(overall,indent=2),encoding='utf-8');print(json.dumps(overall,indent=2));print('saved:',args.output)
if __name__=='__main__':main()
