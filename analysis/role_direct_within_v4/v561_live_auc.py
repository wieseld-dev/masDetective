#!/usr/bin/env python3
"""Live per-round MAS5 within-system AUC for the precommitted V5.6 nonadaptive protocol.

Zero additional API calls compared with V5.6. Reorders the same execute/score
operations so R1 evaluation is available before R2 starts. Evaluation labels
are loaded ONLY by live/report stage, never by generation or sensor scoring.
"""
from __future__ import annotations
import argparse, datetime, json, math, os, statistics
from pathlib import Path

from analysis.role_direct_within_v4 import v56_nonadaptive as v56


def read_scores(root: Path, round_idx: int, maxlen: int):
    path=root/f'sensor_public_len{maxlen}_round_{round_idx:02}.jsonl'
    if not path.is_file():
        raise RuntimeError(f'Missing round-score checkpoint {path}')
    items=v56.jl(path)
    keys={(str(x['session_id']),str(x['target_role']),int(x['round'])) for x in items}
    if len(items)!=100 or len(keys)!=100 or {x['round'] for x in items}!={round_idx}:
        raise RuntimeError(f'Expected exactly 100 distinct scored MAS/role rows for R{round_idx} len{maxlen}')
    return items


def score_round(root: Path, round_idx: int):
    import torch
    from peft import PeftConfig, PeftModel
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from analysis.role_direct_within_v4.pipeline import catalog, make_sensor_text
    if round_idx not in (1,2): raise RuntimeError('Only R1 and R2')
    if not torch.cuda.is_available() or '4090' not in torch.cuda.get_device_name(0):
        raise RuntimeError('Expected RTX 4090')
    if not (root/f'responses_public_dev_round_{round_idx:02}.jsonl').is_file():
        raise RuntimeError('Public MAS responses missing; refusing scoring')
    for length in (640,1536):
        if (root/f'sensor_public_len{length}_round_{round_idx:02}.jsonl').exists():
            raise RuntimeError('Refusing to overwrite prior Sensor checkpoint; inspect first')
    cfg=PeftConfig.from_pretrained(str(v56.SENSOR),local_files_only=True)
    tok=AutoTokenizer.from_pretrained(str(v56.SENSOR),local_files_only=True)
    if tok.pad_token_id is None:tok.pad_token=tok.eos_token
    tok.truncation_side='right'
    base=AutoModelForSequenceClassification.from_pretrained(cfg.base_model_name_or_path,
        num_labels=1,device_map={'':0},dtype=torch.bfloat16,
        attn_implementation='eager',local_files_only=True)
    base.config.pad_token_id=tok.pad_token_id
    model=PeftModel.from_pretrained(base,str(v56.SENSOR),is_trainable=False,local_files_only=True).eval()
    desc=catalog()['DESCRIPTIONS']
    cs=v56.jl(root/f'candidates_dev_round_{round_idx:02}.jsonl')
    rs=v56.jl(root/f'responses_public_dev_round_{round_idx:02}.jsonl')
    key=lambda x:(str(x['session_id']),str(x['target_role']))
    cc={key(x):x for x in cs};rr={key(x):x for x in rs}
    if len(cc)!=100 or len(rr)!=100 or set(cc)!=set(rr):
        raise RuntimeError('Expected 100 unique candidate/response pairs')
    out={640:[],1536:[]}
    for ix,k in enumerate(sorted(cc),1):
        x=rr[k]
        if cc[k]['query']!=x['query'] or x.get('execution_status')!='ok':
            raise RuntimeError(f'MAS query/status mismatch for {k}')
        inp=make_sensor_text(k[1],x['query'],x['final_output'],desc)
        full=len(tok(inp,add_special_tokens=True,truncation=False)['input_ids'])
        for length in (640,1536):
            enc=tok([inp],padding=True,truncation=True,max_length=length,return_tensors='pt').to('cuda')
            with torch.inference_mode(),torch.autocast(device_type='cuda',dtype=torch.bfloat16):
                val=float(model(**enc).logits.reshape(-1)[0].float().item())
            if not math.isfinite(val):raise RuntimeError(f'Nonfinite Sensor value for {k}')
            out[length].append({'session_id':k[0],'target_role':k[1],
                'round':round_idx,'logit':round(val,7),'full_tokens':full,
                'truncated':full>length})
        if ix%10==0:
            print(f'MAS5_V561_SENSOR_PROGRESS round={round_idx} scored={ix}/100',flush=True)
    for length in (640,1536):
        v56.safel(root/f'sensor_public_len{length}_round_{round_idx:02}.jsonl',out[length])
        print(f'MAS5_V561_SENSOR_DONE round={round_idx} window={length} truncations={sum(x["truncated"] for x in out[length])}',flush=True)


def evaluate(root:Path, current_round:int, maxlen:int):
    from sklearn.metrics import roc_auc_score
    roles=v56.j(root/'public_preflight.json')['roles']
    sids=v56.j(root/'public_preflight.json')['session_ids']
    if len(roles)!=20 or len(sids)!=5:raise RuntimeError('Expected 20 roles and 5 MAS')
    truth={str(x['session_id']):set(x['roles']) for x in v56.j(v56.SIMPLE)['systems'] if x.get('split')=='dev'}
    if set(truth)!=set(sids) or any(len(truth[sid])!=3 for sid in sids):
        raise RuntimeError('Expected 3 present roles for each of 5 MAS')
    scored={}
    for r in range(1,current_round+1):
        for x in read_scores(root,r,maxlen):
            k=(x['session_id'],x['target_role'],x['round'])
            if k in scored:raise RuntimeError('Duplicate scored role')
            scored[k]=x['logit']
    methods=['r1'] if current_round==1 else ['r1','r2','mean','max']
    result={}
    for method in methods:
        per_mas=[]
        for sid in sids:
            values=[]
            for role in roles:
                a=scored[(sid,role,1)]
                b=scored.get((sid,role,2))
                value=a if method=='r1' else (b if method=='r2' else (a+b)/2 if method=='mean' else max(a,b))
                values.append((role,value))
            y=[int(role in truth[sid]) for role,_ in values]
            auc=float(roc_auc_score(y,[score for _,score in values]))
            top3=[role for role,_ in sorted(values,key=lambda x:(-x[1],x[0]))[:3]]
            correct=len(set(top3)&truth[sid])
            per_mas.append({'session_id':sid,'within_mas_auc':round(auc,6),
                'exact_top3':correct==3,'top3_correct':correct,
                'role_f1_at_k3':round(correct/3,6),'top3':top3,
                'truth_eval_only':sorted(truth[sid])})
        result[method]={'mean_within_mas_auc':round(statistics.mean(s['within_mas_auc'] for s in per_mas),6),
            'exact_top3_count':sum(s['exact_top3'] for s in per_mas),
            'mean_role_f1_at_k3':round(statistics.mean(s['role_f1_at_k3'] for s in per_mas),6),
            'systems':per_mas}
    return result


def live(root: Path, round_idx:int):
    if round_idx not in (1,2):raise RuntimeError('Only R1/R2')
    outpath=root/f'live_auc_round_{round_idx:02}.json'
    if outpath.exists():raise RuntimeError('Live report already exists for this round')
    methods={}
    for length in (640,1536):
        methods[f'len{length}']=evaluate(root,round_idx,length)
    primary='r1' if round_idx==1 else 'mean'
    summary=methods['len640'][primary]
    first=methods['len640']['r1']
    obj={'round_completed':round_idx,
        'computed_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'primary_metric':'Within-MAS AUC averaged across 5 MAS (20 roles each, 3 positives)',
        'current_primary_method':f'len640_{primary}',
        'within_mas_auc':summary['mean_within_mas_auc'],
        'exact_top3':f'{summary["exact_top3_count"]}/5',
        'role_f1_at_k3':summary['mean_role_f1_at_k3'],
        'methods':methods,
        'evaluation_only_truth':True,
        'note':'This is DEV MAS5 reused from prior work, not unseen generalization; no adaptive R2.'}
    if round_idx==2:
        obj['change_auc_fused_r2_vs_r1']=round(summary['mean_within_mas_auc']-first['mean_within_mas_auc'],6)
        obj['change_exact_top3_fused_r2_vs_r1']=summary['exact_top3_count']-first['exact_top3_count']
    # The primary files are atomic and independent, and cannot be silently overwritten.
    v56.safej(outpath,obj)
    history=root/'live_auc_history.jsonl'
    with history.open('a',encoding='utf-8') as f:
        f.write(json.dumps({'round':round_idx,'auc':obj['within_mas_auc'],
            'exact_top3':obj['exact_top3'],'primary_method':obj['current_primary_method'],
            'auc_change':obj.get('change_auc_fused_r2_vs_r1')},ensure_ascii=False)+'\n')
    latest=root/'live_auc_latest.json'
    tmp=latest.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    os.replace(tmp,latest)
    print('MAS5_LIVE_AUC',f'ROUND={round_idx}',f'METHOD=len640_{primary}',
        f'AUC={obj["within_mas_auc"]:.6f}',f'EXACT_TOP3={obj["exact_top3"]}',
        f'F1={obj["role_f1_at_k3"]:.6f}',
        f'CHANGE_AUC={obj.get("change_auc_fused_r2_vs_r1","N/A")}',flush=True)
    for s in summary['systems']:
        print('MAS5_LIVE_PER_SYSTEM',f'ROUND={round_idx}',f'MAS={s["session_id"]}',
            f'AUC={s["within_mas_auc"]:.6f}',
            f'TOP3_CORRECT={s["top3_correct"]}/3',flush=True)
    print('MAS5_LIVE_FILE',latest,flush=True)


def final(root:Path):
    for length in (640,1536):
        path=root/f'sensor_public_len{length}.jsonl'
        if path.exists():raise RuntimeError(f'Final score file already exists: {path}')
        records=read_scores(root,1,length)+read_scores(root,2,length)
        v56.safel(path,records)
    v56.report(root)
    if not (root/'live_auc_round_02.json').exists():raise RuntimeError('Round 2 live snapshot missing')
    expected=v56.j(root/'live_auc_round_02.json')['within_mas_auc']
    got=v56.j(root/'final_MAS5_report.json')['precommitted_primary_result']['mean_within_mas_auc']
    if expected!=got:raise RuntimeError('Live AUC and final AUC discrepancy')
    print('MAS5_V561_FINAL_VERIFIED',expected,flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('stage',choices=['execute1','execute2','score1','score2','live1','live2','final'])
    p.add_argument('--root',required=True)
    args=p.parse_args()
    root=Path(args.root).resolve()
    if not root.is_relative_to(v56.PROJECT/'artifacts/role_direct_within_v4/nonadaptive_v56'):
        raise RuntimeError('Output root outside V56 area')
    actions={
        'execute1':lambda:v56.execute(root,1),
        'execute2':lambda:v56.execute(root,2),
        'score1':lambda:score_round(root,1),
        'score2':lambda:score_round(root,2),
        'live1':lambda:live(root,1),
        'live2':lambda:live(root,2),
        'final':lambda:final(root),
    }
    actions[args.stage]()

if __name__=='__main__':main()
