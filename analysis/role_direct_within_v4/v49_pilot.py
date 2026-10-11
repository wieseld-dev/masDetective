#!/usr/bin/env python3
"""MAS5 v4.9: controlled diagnostic matched PRESENT/ABSENT pilot.

Prepare is API-free; execute uses existing scripts/adaptive_v11_execute_round.py.
Report measures activation (training traces) + out-of-pair text discrimination.
No generator training and no MAS5 evaluation performed here.
"""
from __future__ import annotations
import argparse, collections, hashlib, json, math, statistics, subprocess, sys
from pathlib import Path

PROJECT=Path('/home/wieseld/mas_rl_detective')
OLD=PROJECT/'artifacts/role_direct_within_v4/overnight_v46/mas5_night_20261009_224735_1879051'
BASE=OLD/'train/v47_quality/v472_reviewed_second_pass/v476_manual_review'
APPROVED_134=BASE/'v480_final42_review/approved_134.jsonl'
APPROVED_118=BASE/'approved_118.jsonl'
CF=PROJECT/'artifacts/role_direct_matched_cf_v1/manifest_privileged.json'
MAS5=PROJECT/'artifacts/role_direct_routing_pilot_v1/manifest_privileged.json'
ROLES=['research','database_design','customer_support_triage']
PAIR_COUNT=3
PROBES_PER_ROLE=3
MAX_CALLS=len(ROLES)*PAIR_COUNT*2*PROBES_PER_ROLE


def read_jsonl(path):
    with Path(path).open(encoding='utf-8') as f:
        return [json.loads(x) for x in f if x.strip()]


def atomic_json(path,obj):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists(): raise RuntimeError('Refusing to overwrite '+str(path))
    path.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')


def atomic_jsonl(path,rows):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists(): raise RuntimeError('Refusing to overwrite '+str(path))
    with path.open('x',encoding='utf-8') as f:
        for row in rows: f.write(json.dumps(row,ensure_ascii=False)+'\n')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def choose_probes(rows,role):
    found=sorted([x for x in rows if x['target_role']==role],key=lambda x:int(x['strategy_index']))
    if len({int(x['strategy_index']) for x in found})!=len(found):raise RuntimeError('Duplicate slots: '+role)
    hm=[x for x in found if x.get('probe_family')=='help_me_help_you']
    direct=[x for x in found if x.get('probe_family')=='direct_task']
    # Two HMHY from distinct subfamilies when possible. One direct control when available.
    picks=[];families=set()
    for q in hm:
        family=int(q['strategy_index'])//2
        if family not in families:
            picks.append(q);families.add(family)
        if len(picks)==2:break
    if len(picks)<2:raise RuntimeError(f'Not enough distinct HMHY for {role}: {[(q["strategy_index"],q["query"][:40]) for q in hm]}')
    extra=direct[:1] or [q for q in hm if int(q['strategy_index']) not in {int(z['strategy_index']) for z in picks}][:1]
    if not extra:raise RuntimeError(f'Not enough reviewed probes for {role}')
    return picks+extra


def prepare(root,source=None):
    root=Path(root).resolve()
    if root.exists():raise RuntimeError('Run root exists; will not overwrite: '+str(root))
    source=Path(source).resolve() if source else (APPROVED_134 if APPROVED_134.is_file() else APPROVED_118)
    if not source.is_file():raise RuntimeError('Reviewed bank missing: '+str(source))
    if not CF.is_file() or not MAS5.is_file():raise RuntimeError('Missing existing CF or MAS5 manifest')
    bank=read_jsonl(source)
    selected={role:choose_probes(bank,role) for role in ROLES}
    dev={frozenset(m['roles']) for m in json.loads(MAS5.read_text())['systems'] if m['split']=='dev'}
    pool=collections.defaultdict(dict)
    for sysrec in json.loads(CF.read_text())['systems']:
        pool[(str(sysrec['target_role']),str(sysrec['pair_id']))][str(sysrec['condition'])]=sysrec
    matches=collections.defaultdict(list)
    for (role,pair),cond in sorted(pool.items()):
        if role not in ROLES:continue
        if set(cond)!={'TARGET_PRESENT','TARGET_ABSENT'}:raise RuntimeError('CF pair incomplete '+str((role,pair)))
        pos,neg=cond['TARGET_PRESENT'],cond['TARGET_ABSENT']
        if pos['roles'][:2]!=neg['roles'][:2] or pos['roles'][-1]!=role or role in neg['roles']:
            raise RuntimeError('Counterfactual mismatch '+str((role,pair)))
        if (pos['mode'],pos['style'])!=(neg['mode'],neg['style']):raise RuntimeError('Mode/style mismatch')
        if frozenset(pos['roles']) in dev or frozenset(neg['roles']) in dev:continue
        matches[role].append((pair,pos,neg))
    for role in ROLES:
        if len(matches[role])<PAIR_COUNT:raise RuntimeError(f'Only {len(matches[role])} non-MAS5 backgrounds for {role}')

    systems=[];candidates=[];preview=[]
    for role in ROLES:
        for k,(old_pair,pos,neg) in enumerate(matches[role][:PAIR_COUNT],1):
            pair_id=f'v49_{role}_p{k:02d}'
            for label,m in [('TARGET_PRESENT',pos),('TARGET_ABSENT',neg)]:
                sid=f'v49_{role}_p{k:02d}_{"POS" if label=="TARGET_PRESENT" else "NEG"}'
                systems.append({**m,'session_id':sid,'split':'train','pair_id':pair_id})
                for q in selected[role]:
                    candidates.append({'session_id':sid,'target_role':role,'round':1,'split':'train',
                      'candidate_index':int(q['strategy_index']),'strategy':q.get('strategy',''),
                      'probe_family':q.get('probe_family',''), 'query':q['query'],'valid':True,
                      'pair_id':pair_id,'condition':label})
    for role in ROLES:
        for q in selected[role]:
            preview.append({'target_role':role,'slot':int(q['strategy_index']),
                'probe_family':q['probe_family'],'strategy':q.get('strategy',''), 'query':q['query']})
    assert len(candidates)==MAX_CALLS and len(systems)==len(ROLES)*PAIR_COUNT*2
    assert len({(c['session_id'],c['candidate_index']) for c in candidates})==MAX_CALLS
    from adaptive_v11_active.io import sha256_json
    manifest={'version':'MAS5_V49_PILOT_TRAIN_ONLY','systems':systems,'sha256':sha256_json(systems)}
    root.mkdir(parents=True)
    atomic_json(root/'manifest_privileged.json',manifest)
    atomic_jsonl(root/'candidates_train_round_01.jsonl',candidates)
    atomic_jsonl(root/'selected_probe_preview.jsonl',preview)
    atomic_json(root/'preflight.json',{'status':'READY_FOR_EXPLICIT_API_APPROVAL',
        'run_root':str(root),'source_reviewed_bank':str(source),'source_bank_sha256':sha(source),
        'source_approved_rows':len(bank),'roles':ROLES,'pairs_per_role':PAIR_COUNT,
        'probes_per_role':PROBES_PER_ROLE,'mas_executions_max':MAX_CALLS,
        'note':'Each MAS execution may invoke multiple paid LLM calls; all answers and routing are TRAIN ONLY.',
        'interpretation':'Control removes the target role by REPLACING it with a substitute role; causal effect may be confounded by substitute.'})
    print('MAS5_V49_PREPARED',len(candidates),'possible MAS executions',flush=True)
    print('SOURCE',source,flush=True)
    for row in preview:
        print(f"  {row['target_role']:<25} slot={row['slot']} {row['probe_family']}: {row['query'][:145]}",flush=True)
    print('NO API CALLS YET. Approval required to submit.',flush=True)


def execute(root):
    root=Path(root).resolve()
    pf=json.loads((root/'preflight.json').read_text())
    if pf['mas_executions_max']!=MAX_CALLS:raise RuntimeError('Execution cap differs from expected')
    if (root/'responses_train_round_01.jsonl').exists():
        print('Responses already present; skipping paid executions',flush=True)
        return
    cmd=[sys.executable,'-u','scripts/adaptive_v11_execute_round.py','--round','1','--split','train',
      '--root',str(root),'--manifest',str(root/'manifest_privileged.json'),'--workers','2','--retries','1']
    import os
    if not os.environ.get('OPENAI_API_KEY'):raise RuntimeError('API credentials unavailable: no execution')
    print('MAS5_V49_EXECUTE_START',MAX_CALLS,'maximum distinct MAS executions',flush=True)
    subprocess.run(cmd,cwd=PROJECT,check=True)
    responses=read_jsonl(root/'responses_train_round_01.jsonl')
    if len(responses)!=MAX_CALLS or any(x['execution_status']!='ok' for x in responses):
        raise RuntimeError('Incomplete pilot execution; check files before reporting')
    print('MAS5_V49_EXECUTE_DONE',len(responses),flush=True)


def auc(scores,labels):
    pos=[s for s,y in zip(scores,labels) if y]
    neg=[s for s,y in zip(scores,labels) if not y]
    if not pos or not neg:return None
    return sum(1. if a>b else 0.5 if a==b else 0. for a in pos for b in neg)/(len(pos)*len(neg))


def report(root):
    root=Path(root).resolve()
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    c=read_jsonl(root/'candidates_train_round_01.jsonl')
    r=read_jsonl(root/'responses_train_round_01.jsonl')
    by={(str(x['session_id']),int(x['candidate_index'])):x for x in r}
    assert len(c)==len(r)==MAX_CALLS==len(by)
    groups=collections.defaultdict(list)
    for item in c:
        rr=by[(str(item['session_id']),int(item['candidate_index']))]
        assert rr['query']==item['query'] and rr['condition']==item['condition']
        assert rr['execution_status']=='ok'
        groups[(item['target_role'],int(item['candidate_index']))].append(rr)
    metrics=[]
    for (role,slot),recs in sorted(groups.items()):
        assert len(recs)==PAIR_COUNT*2
        bypair=collections.defaultdict(dict)
        for x in recs:
            bypair[str(x['pair_id'])][x['condition']]=x
        assert len(bypair)==PAIR_COUNT and all(set(v)=={'TARGET_PRESENT','TARGET_ABSENT'} for v in bypair.values())
        activ=[role in a['selected_roles_training_only'] for a in [pair['TARGET_PRESENT'] for pair in bypair.values()]]
        # Leave-one-matched-background-pair-out. Every held-out prediction uses ONLY the final response.
        preds=[];labels=[];paired_wins=[]
        for heldpair in sorted(bypair):
            train=[x for p,side in bypair.items() if p!=heldpair for x in (side['TARGET_PRESENT'],side['TARGET_ABSENT'])]
            test=[bypair[heldpair][k] for k in ('TARGET_PRESENT','TARGET_ABSENT')]
            traintext=[str(x.get('final_output','')) for x in train]
            testtext=[str(x.get('final_output','')) for x in test]
            v=TfidfVectorizer(analyzer='char',ngram_range=(3,5),min_df=1,max_features=16000,sublinear_tf=True)
            X=v.fit_transform(traintext)
            Xt=v.transform(testtext)
            pos=X[[i for i,x in enumerate(train) if x['condition']=='TARGET_PRESENT']].mean(axis=0)
            neg=X[[i for i,x in enumerate(train) if x['condition']=='TARGET_ABSENT']].mean(axis=0)
            import numpy as np
            pos=np.asarray(pos).reshape(1,-1);neg=np.asarray(neg).reshape(1,-1)
            scores=(cosine_similarity(Xt,pos).ravel()-cosine_similarity(Xt,neg).ravel()).tolist()
            preds+=scores;labels+=[1,0]
            paired_wins.append(1. if scores[0]>scores[1] else 0.5 if scores[0]==scores[1] else 0.)
        present=[len(by_pair['TARGET_PRESENT']['final_output']) for by_pair in bypair.values()]
        absent=[len(by_pair['TARGET_ABSENT']['final_output']) for by_pair in bypair.values()]
        row={'role':role,'slot':slot,'probe_family':recs[0]['probe_family'],
            'strategy':recs[0]['strategy'],'target_activation_rate_present':sum(activ)/len(activ),
            'target_activation_observations':sum(activ),'matched_pairs':PAIR_COUNT,
            'heldout_pair_auc_diagnostic':auc(preds,labels),
            'heldout_pair_win_rate_diagnostic':statistics.mean(paired_wins),
            'present_mean_chars':statistics.mean(present),'absent_mean_chars':statistics.mean(absent),
            'query':recs[0]['query'],'warning':'Not a validated reward; n=3 pairs, substitute role differs'}
        metrics.append(row)
    hm=[x for x in metrics if x['probe_family']=='help_me_help_you']
    dr=[x for x in metrics if x['probe_family']=='direct_task']
    summary={'status':'DIAGNOSTIC_ONLY_NOT_DPO_READY',
        'mas_executions':MAX_CALLS,'roles':ROLES,'matched_pairs_per_probe':PAIR_COUNT,
        'probes':len(metrics),'hmhy_probes':len(hm),'direct_controls':len(dr),
        'hmhy_mean_heldout_auc':statistics.mean(x['heldout_pair_auc_diagnostic'] for x in hm) if hm else None,
        'direct_mean_heldout_auc':statistics.mean(x['heldout_pair_auc_diagnostic'] for x in dr) if dr else None,
        'hmhy_mean_activation':statistics.mean(x['target_activation_rate_present'] for x in hm) if hm else None,
        'limitations':['PRESENT/ABSENT differs by substitution of third worker; not pure knockout',
                       'Three pairs per probe yield very noisy AUC estimates',
                       'The text classifier is trained only within other counterfactual pairs, not on MAS5 truth',
                       'Pilot results must not serve as held-out MAS5 evaluation or establish DPO preference reliability.']}
    atomic_jsonl(root/'probe_diagnostic.jsonl',metrics)
    atomic_json(root/'pilot_report.json',summary)
    print('MAS5_V49_REPORT',json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
    for x in metrics:
        print(f"  {x['role']:<25} slot={x['slot']} family={x['probe_family']:<18} activation={x['target_activation_rate_present']:.2f} heldout_auc={x['heldout_pair_auc_diagnostic']:.3f}",flush=True)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('phase',choices=['prepare','execute','report'])
    ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--source',type=Path,default=None)
    a=ap.parse_args()
    if a.phase=='prepare':prepare(a.root,a.source)
    if a.phase=='execute':execute(a.root)
    if a.phase=='report':report(a.root)
if __name__=='__main__':main()
