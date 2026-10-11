from __future__ import annotations
import argparse, gc, random
from pathlib import Path
from adaptive_v11_active.io import write_jsonl, stable_int
from v10.policy import SeededProbePolicyV10
from v11.role_universe import TRAIN_ROLES, ROLE_DESCRIPTIONS, anchor_probe

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--adapter',default='artifacts/v11_policy'); ap.add_argument('--model-name',default='Qwen/Qwen2.5-1.5B-Instruct'); ap.add_argument('--output',default='artifacts/adaptive_v11_curriculum_v1/v11_retention.jsonl'); ap.add_argument('--variants',type=int,default=3); ap.add_argument('--seed',type=int,default=99173); ap.add_argument('--max-seq-length',type=int,default=2048); a=ap.parse_args()
    import torch
    pol=SeededProbePolicyV10(a.model_name,adapter_path=a.adapter,trainable=False,use_4bit=False,max_seq_length=a.max_seq_length)
    rows=[]
    for role in TRAIN_ROLES:
        avoid=[]
        for v in range(a.variants):
            seed=stable_int(a.seed,role,v); random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
            obs={'candidate_role':role,'candidate_description':ROLE_DESCRIPTIONS[role],'anchor_probe':anchor_probe(role),'avoid_probes':avoid}
            sample=pol.sample_probe(obs,temperature=.80,top_p=.95,max_new_tokens=128)
            if sample.valid and sample.completion:
                rows.append({'role':role,'variant':v,'prompt':str(sample.prompt),'completion':str(sample.completion),'query':str(sample.query)})
                avoid.append(str(sample.query))
    if len(rows)<len(TRAIN_ROLES): raise RuntimeError(f"retention buffer too small: {len(rows)}")
    write_jsonl(a.output,rows); print(f"[retention] rows={len(rows)} -> {a.output}")
    del pol; gc.collect(); torch.cuda.empty_cache()
if __name__=='__main__': main()
