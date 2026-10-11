from __future__ import annotations
import argparse,json
from pathlib import Path
from adaptive_v1.judge import V135Scorer
from v11.role_universe import TRAIN_ROLES,ROLE_DESCRIPTIONS


def load_jsonl(p):
    out=[]
    with Path(p).open(encoding='utf-8') as f:
        for line in f:
            if line.strip():out.append(json.loads(line))
    return out


def main():
    ap=argparse.ArgumentParser(description='Score adaptive replay responses with frozen V13.5 all-role evidence Judge')
    ap.add_argument('--input',default='artifacts/adaptive_v1/public_observations.jsonl')
    ap.add_argument('--output',default='artifacts/adaptive_v1/replay_bank.jsonl')
    ap.add_argument('--judge-adapter',default='artifacts/v13_5_evidence_judge/best_adapter')
    ap.add_argument('--item-batch',type=int,default=4)
    ap.add_argument('--max-length',type=int,default=768)
    args=ap.parse_args()
    rows=load_jsonl(args.input);sc=V135Scorer(args.judge_adapter,TRAIN_ROLES,ROLE_DESCRIPTIONS,item_batch=args.item_batch,max_length=args.max_length)
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True);tmp=out.with_suffix(out.suffix+'.tmp')
    with tmp.open('w',encoding='utf-8') as f:
        step=64
        for st in range(0,len(rows),step):
            chunk=rows[st:st+step];scores=sc.score_many([(r['query'],r['output']) for r in chunk])
            for r,e in zip(chunk,scores):
                rec={'system_id':r['system_id'],'split':r['split'],'truth_roles':r['truth_roles_training_only'],'action':r['action'],'query':r['query'],'evidence':e,'response_id':f"{r['system_id']}::{r['action']['target']}::v{r['action'].get('variant',0)}",'method':'v11_2+v13_5','system_metadata':{'mode':r.get('mode')},'metadata':{'selected_roles_training_only':r.get('selected_roles_training_only',[]),'manifest_sha256':r.get('manifest_sha256')}}
                f.write(json.dumps(rec,ensure_ascii=False)+'\n')
            f.flush();print(f"[score] {min(st+len(chunk),len(rows))}/{len(rows)}",flush=True)
    tmp.replace(out);print(f'[done] replay_bank={out} rows={len(rows)}',flush=True)
if __name__=='__main__':main()
