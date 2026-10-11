from __future__ import annotations
import argparse,json,random
from pathlib import Path
from v11.role_universe import TRAIN_ROLES,ROLE_DESCRIPTIONS,anchor_probe
from v11_2.counterfactual import PREFERRED_CONFUSERS
from v11_2.policy import HardConfuserProbePolicyV112


def main():
    ap=argparse.ArgumentParser(description='Generate a frozen benign action library from final V11.2')
    ap.add_argument('--adapter',default='artifacts/v11_2_split/policy')
    ap.add_argument('--output',default='artifacts/adaptive_v1/action_library.jsonl')
    ap.add_argument('--variants',type=int,default=1)
    ap.add_argument('--temperature',type=float,default=.85)
    ap.add_argument('--seed',type=int,default=314159)
    ap.add_argument('--no-4bit',action='store_true')
    args=ap.parse_args()
    if not Path(args.adapter).is_dir():raise SystemExit(f'missing final V11.2 adapter: {args.adapter}')
    import torch
    random.seed(args.seed);torch.manual_seed(args.seed)
    pol=HardConfuserProbePolicyV112(adapter_path=args.adapter,trainable=False,use_4bit=not args.no_4bit,max_seq_length=1536)
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True);tmp=out.with_suffix(out.suffix+'.tmp')
    n=0
    with tmp.open('w',encoding='utf-8') as f:
        for role in TRAIN_ROLES:
            avoid=[]
            hard=[{'role':c,'description':ROLE_DESCRIPTIONS.get(c,c)} for c in PREFERRED_CONFUSERS.get(role,())[:4]]
            for v in range(args.variants):
                obs={'candidate_role':role,'candidate_description':ROLE_DESCRIPTIONS[role],'anchor_probe':anchor_probe(role),'avoid_probes':avoid,'hard_confusers':hard}
                sample=pol.sample_probe(obs,temperature=args.temperature,top_p=.95,max_new_tokens=96,deterministic=False)
                rec={'target':role,'competitor':None,'hard_confusers':[x['role'] for x in hard],'variant':v,'query':sample.query,'valid':bool(sample.valid),'invalid_reason':sample.invalid_reason,'raw_completion':sample.raw_completion}
                f.write(json.dumps(rec,ensure_ascii=False)+'\n');f.flush();n+=1
                if sample.valid:avoid.append(sample.query)
                print(f"[generate] {n}/{len(TRAIN_ROLES)*args.variants} role={role:<24} v={v} valid={sample.valid} q={sample.query!r}",flush=True)
    tmp.replace(out);print(f'[done] action_library={out} rows={n}',flush=True)
if __name__=='__main__':main()
