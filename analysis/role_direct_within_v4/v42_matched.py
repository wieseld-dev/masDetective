#!/usr/bin/env python3
"""Analyze genuinely matched Role PRESENT/ABSENT probes and gate DPO data.

Requires controlled SAME query on matched MAS compositions and target routing
available ONLY as training-time evidence. Never exposes these labels to runtime.
No API calls. No generator fine-tuning on repeatedly inspected DEV by default.
"""
from __future__ import annotations
import argparse, collections, json, math, re, statistics
from pathlib import Path
import numpy as np
import joblib
from analysis.role_direct_within_v4 import v42_sensor


def readl(path):
    with Path(path).open() as f:return [json.loads(x) for x in f if x.strip()]

def norm(s):return re.sub(r'\W+',' ',str(s).lower()).strip()


def join(candidates,responses):
    idx=collections.defaultdict(list)
    for r in responses:
        sid=str(r.get('system_id',r.get('session_id','')))
        idx[(sid,norm(r.get('query','')))].append(r)
    grouped=collections.defaultdict(dict)
    count=0
    for c in candidates:
        sid=str(c['session_id']);query=str(c['query']);key=(sid,norm(query))
        if len(idx[key])!=1:raise RuntimeError(f'Missing or duplicate execution for {sid} / {query[:50]}')
        r=idx[key][0]
        cond=str(c['condition']);role=str(c['target_role']);pair=str(c['pair_id'])
        if cond not in ('TARGET_PRESENT','TARGET_ABSENT'):raise RuntimeError(f'Unexpected condition {cond}')
        # The same role+pair+query must occur in both conditions, never join by role name alone.
        gkey=(role,pair,norm(query))
        if cond in grouped[gkey]:raise RuntimeError('Duplicate condition for '+repr(gkey))
        answer=str(r.get('final_output',''))
        if not answer.strip():raise RuntimeError('Missing final response')
        routed=role in set(map(str,r.get('selected_roles_training_only',[])))
        grouped[gkey][cond]={'query':query,'target_role':role,'system_id':sid,
                             'final_output':answer,'routed':routed}
        count+=1
    out=[]
    for (role,pair,qnorm),v in grouped.items():
        if set(v)!={'TARGET_PRESENT','TARGET_ABSENT'}:raise RuntimeError(f'Incomplete matched pair {role} {pair}')
        out.append({'target_role':role,'pair_id':pair,'query':v['TARGET_PRESENT']['query'],
                  'positive':v['TARGET_PRESENT'],'negative':v['TARGET_ABSENT']})
    return out,count


def score_model(path,paired):
    data=joblib.load(path)
    model=data['ranker'];ds=data['catalog'];allobs=[]
    for pair in paired:allobs.extend([pair['positive'],pair['negative']])
    scores=model.score(allobs,ds)
    for i,pair in enumerate(paired):
        pair['score_present']=float(scores[2*i]);pair['score_absent']=float(scores[2*i+1]);
        pair['score_delta']=float(scores[2*i]-scores[2*i+1]);
        pair['pair_win']=float(scores[2*i]>scores[2*i+1])+.5*float(scores[2*i]==scores[2*i+1]);
    return paired


def summarize(paired,source_label,allowed_to_train):
    byquery=collections.defaultdict(list)
    for p in paired:byquery[(p['target_role'],norm(p['query']))].append(p)
    audit=[];byrole=collections.defaultdict(list)
    for (role,qnorm),ps in byquery.items():
        if len({p['pair_id'] for p in ps})!=len(ps):raise RuntimeError('Pair ID repeated for same query')
        activation=statistics.mean(float(p['positive']['routed']) for p in ps)
        win=statistics.mean(p['pair_win'] for p in ps)
        avg=statistics.mean(p['score_delta'] for p in ps)
        z={'role':role,'query':ps[0]['query'],'background_pairs':len(ps),
           'target_activation_rate':activation,'paired_win_rate':win,'mean_signed_score_delta':avg,
           'utility':activation*win*max(0.,math.tanh(avg/2))}
        audit.append(z);byrole[role].append(z)
    # Minimum 3 independent backgrounds / candidate. A single lucky response is not enough.
    prefs=[]
    if allowed_to_train:
        for role,ss in byrole.items():
            valid=[x for x in ss if x['background_pairs']>=3]
            positives=[x for x in valid if x['target_activation_rate']>=.8 and x['paired_win_rate']>=.8 and x['mean_signed_score_delta']>.25]
            negatives=[x for x in valid if x['paired_win_rate']<=.6 or x['target_activation_rate']<.4]
            for good in positives:
                losers=sorted((z for z in negatives if norm(z['query'])!=norm(good['query'])),key=lambda z:z['utility'])
                if losers and good['utility']-losers[0]['utility']>.12:
                    prefs.append({'role':role,'chosen':good['query'],'rejected':losers[0]['query'],
                     'observed_activation_rate':good['target_activation_rate'],
                     'observed_pair_win_rate':good['paired_win_rate'],'observed_utility_gap':good['utility']-losers[0]['utility']})
    n_roles=len({x['role'] for x in prefs})
    # Robustness gate: not repeat 6-preference fiasco. Not a scientific guarantee.
    eligible=allowed_to_train and len(prefs)>=80 and n_roles>=12 and len(byquery)>=160
    report={'source':source_label,'unique_queries':len(byquery),'matched_comparisons':len(paired),
      'roles_with_queries':len(byrole),'n_queries_3_or_more_backgrounds':sum(x['background_pairs']>=3 for x in audit),
      'mean_target_activation':statistics.mean(x['target_activation_rate'] for x in audit) if audit else None,
      'mean_pairwin':statistics.mean(x['paired_win_rate'] for x in audit) if audit else None,
      'preference_pairs':len(prefs),'preference_roles':n_roles,'training_eligible':eligible,
      'dev_source_can_never_be_used_for_dpo_in_this_script':not allowed_to_train,
      'caution':'Frozen sensor proxy; actual information gain/Exact Top-3 still requires independent prospective test'}
    return report,audit,prefs


def main():
    p=argparse.ArgumentParser();p.add_argument('--candidates',required=True);p.add_argument('--responses',required=True)
    p.add_argument('--scorer',required=True);p.add_argument('--out',required=True)
    p.add_argument('--fresh-train-data',action='store_true',help='Explicit confirmation source has never been inspected or used as DEV')
    a=p.parse_args();out=Path(a.out).resolve()
    if not out.is_relative_to(v42_sensor.PROJECT/'artifacts/role_direct_within_v4'):
        raise RuntimeError('Matched output must remain under V4')
    out.mkdir(parents=True,exist_ok=False)
    sources=[str(a.candidates),str(a.responses)]
    if a.fresh_train_data and any('dev' in x.lower() or 'sealed' in x.lower() or 'simple5' in x.lower() for x in sources):
        raise RuntimeError('Refusing to train on DEV, Simple5, or sealed sources')
    pairs,n=join(readl(a.candidates),readl(a.responses))
    print('MATCHED JOIN',n,'records',len(pairs),'comparisons',flush=True)
    score_model(a.scorer,pairs)
    report,audit,prefs=summarize(pairs,';'.join(sources),a.fresh_train_data)
    v42_sensor.writej(out/'report.json',report)
    with (out/'query_evidence.jsonl').open('w') as f:
        for x in audit:f.write(json.dumps(x)+'\n')
    if report['training_eligible']:
        with (out/'generator_preferences.jsonl').open('w') as f:
            for x in prefs:f.write(json.dumps(x)+'\n')
    print('EVIDENCE REPORT',json.dumps(report,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
