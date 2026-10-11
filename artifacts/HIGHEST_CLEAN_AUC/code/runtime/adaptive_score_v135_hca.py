from __future__ import annotations
import argparse,gc,json
from pathlib import Path
from adaptive_v1.judge import V134Scorer,V135Scorer
from v11.role_universe import TRAIN_ROLES,ROLE_DESCRIPTIONS


def load_jsonl(p):
    out=[]
    with Path(p).open(encoding='utf-8') as f:
        for line in f:
            if line.strip(): out.append(json.loads(line))
    return out


def main():
    ap=argparse.ArgumentParser(description='Score adaptive responses with frozen V13.4 + V13.5 dual Judge channels')
    ap.add_argument('--input',default='artifacts/adaptive_v1/public_observations.jsonl')
    ap.add_argument('--output',default='artifacts/adaptive_v1/replay_bank.jsonl')
    ap.add_argument('--v134-adapter',default='artifacts/HIGHEST_CLEAN_AUC/models/sensor_v13_4')
    ap.add_argument('--v135-adapter',default='artifacts/HIGHEST_CLEAN_AUC/models/sensor_v13_5')
    ap.add_argument('--item-batch',type=int,default=2)
    ap.add_argument('--role-batch',type=int,default=4)
    ap.add_argument('--v134-max-length',type=int,default=1024)
    ap.add_argument('--v135-max-length',type=int,default=768)
    ap.add_argument('--max-length',type=int,default=None,help='legacy alias for --v135-max-length')
    args=ap.parse_args()
    if args.max_length is not None: args.v135_max_length=int(args.max_length)
    rows=load_jsonl(args.input)
    if not rows: raise SystemExit('no adaptive observations to score')

    print(f'[dual-score] rows={len(rows)} phase=V13.4 absolute-composition',flush=True)
    s134=V134Scorer(args.v134_adapter,TRAIN_ROLES,ROLE_DESCRIPTIONS,item_batch=args.item_batch,role_batch=args.role_batch,max_length=args.v134_max_length)
    e134=[]; step=32
    for st in range(0,len(rows),step):
        chunk=rows[st:st+step]
        e134.extend(s134.score_many([(r['action']['target'],r['query'],r['output']) for r in chunk]))
        print(f'[score-v134] {min(st+len(chunk),len(rows))}/{len(rows)}',flush=True)
    del s134; gc.collect()
    try:
        import torch; torch.cuda.empty_cache()
    except Exception: pass

    print('[dual-score] phase=V13.5 causal-participation',flush=True)
    s135=V135Scorer(args.v135_adapter,TRAIN_ROLES,ROLE_DESCRIPTIONS,item_batch=args.item_batch,role_batch=args.role_batch,max_length=args.v135_max_length)
    e135=[]
    for st in range(0,len(rows),step):
        chunk=rows[st:st+step]
        e135.extend(s135.score_many([(r['query'],r['output']) for r in chunk]))
        print(f'[score-v135] {min(st+len(chunk),len(rows))}/{len(rows)}',flush=True)

    out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True); tmp=out.with_suffix(out.suffix+'.tmp')
    with tmp.open('w',encoding='utf-8') as f:
        for r,pres,causal in zip(rows,e134,e135):
            md={'selected_roles_training_only':r.get('selected_roles_training_only',[]),'manifest_sha256':r.get('manifest_sha256'),'evidence_v134':pres,'judge_channels':{'v134':'absolute_composition_raw_logit','v135':'query_prior_cancelled_causal_evidence'}}
            rec={'system_id':r['system_id'],'split':r['split'],'truth_roles':r['truth_roles_training_only'],'action':r['action'],'query':r['query'],'evidence':causal,'response_id':f"{r['system_id']}::{r['action']['target']}::v{r['action'].get('variant',0)}",'method':'v11+dual_v13_4_v13_5','system_metadata':{'mode':r.get('mode')},'metadata':md}
            f.write(json.dumps(rec,ensure_ascii=False)+'\n')
    tmp.replace(out)
    print(f'[done] dual replay_bank={out} rows={len(rows)} channels=2',flush=True)

if __name__=='__main__': main()
