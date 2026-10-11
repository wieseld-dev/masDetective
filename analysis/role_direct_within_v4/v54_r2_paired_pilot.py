#!/usr/bin/env python3
"""MAS5 V5.4: response-conditioned, query-matched R2 signal PILOT.

This is NOT a trained Generator/Selector nor true stateful conversational MAS.
The existing target resets all chat messages on each query. R2 questions are
self-contained, human-curated from public R1 observations only.

prepare: no API. freeze public questions before accessing privileged join map.
execute: explicitly authorized billable calls through established V11 runner.
report: descriptive matched query distance & routing diagnostics, NOT AUC/DPO.
"""
from __future__ import annotations
import argparse
import collections
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys

PROJECT = Path('/home/wieseld/mas_rl_detective')
ROLES = ('research','database_design','customer_support_triage')
SRC_SHA = '816c64ad36787d78924a1bf3126a84a1110f5b42054541b1e9aa54a767438fce'
CURATED = Path(__file__).with_name('v54_curated_public_queries.jsonl')
FIXED = {
  'customer_support_triage': 'I am investigating a slowdown after a software update. What observations should I provide first to help you identify the cause?',
  'database_design': 'I am planning an online-store database. What example records or constraints should I provide to help you check the design?',
  'research': 'I am planning a literature review on renewable energy. What project details should I provide first to help you guide the review?',
}
BLOCKED = {'condition','pair_id','session_id','system_id','roles','installed_roles',
           'truth','routing_trace','selected_roles_training_only','original_cf_pair','label'}


def jl(path):
    with Path(path).open(encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def j(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def savej(path,obj):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as f: json.dump(obj,f,indent=2,ensure_ascii=False); f.write('\n')


def savel(path,rows):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as f:
        for row in rows:f.write(json.dumps(row,ensure_ascii=False)+'\n')


def safetext(q):
    return re.sub(r'\s+',' ',str(q)).strip()


def check_public(public,curated):
    if len(public)!=18 or len(curated)!=18:raise RuntimeError('Expected 18 frozen public contexts and 18 curated questions')
    pids=[x['context_id'] for x in public]
    cids=[x['context_id'] for x in curated]
    if len(set(pids))!=18 or set(pids)!=set(cids):raise RuntimeError('Public IDs or curated IDs mismatch')
    pub={r['context_id']:r for r in public}
    for c in curated:
        ctx=c['context_id'];p=pub[ctx]
        if set(p)&BLOCKED:raise RuntimeError('Private fields in public context')
        if c['target_role']!=p['target_role'] or c['target_role'] not in ROLES:
            raise RuntimeError('Target role mismatch '+ctx)
        if not p['previous_response'].startswith(c['anchor_source_excerpt']):
            raise RuntimeError('R1 answer changed or wrong source excerpt '+ctx)
        q=safetext(c['query'])
        if '?' not in q or len(q.split())>50 or len(q.split())<8:
            raise RuntimeError('Curated query needs manual inspection '+ctx)
        if re.search(r'(?i)hidden\s+(?:role|agent)|system\s+prompt|target[_ ]role|routing\s+trace|present\s+vs\s+absent',q):
            raise RuntimeError('Forbidden query '+ctx)
        # Sanity check: share a meaningful topical term with R1 answer. Manual audit is still necessary.
        words=lambda s: {w for w in re.findall(r'[a-z][a-z0-9]{3,}',s.lower()) if w not in {
            'what','which','should','would','could','help','your','with','from','this','that','have','there',
            'into','provide','first','about','example','more','after','before','specific','some','them','user','need'}}
        if len(words(q)&words(p['previous_response']))<1:
            raise RuntimeError('Poor lexical anchor '+ctx)
    if len(set(safetext(c['query']).lower() for c in curated))!=18:
        raise RuntimeError('Duplicate adaptive questions')


def prepare(root,v51,pilot,pairs_per_role):
    if pairs_per_role not in (1,2,3): raise RuntimeError('pairs-per-role must be 1, 2, or 3')
    root=Path(root).resolve();v51=Path(v51).resolve();pilot=Path(pilot).resolve()
    if root.exists():raise RuntimeError('Refuse to overwrite '+str(root))
    public_path=v51/'public_scored_r1_states.jsonl'
    private_path=v51/'private_session_map_training_only.jsonl'
    manifest_path=pilot/'manifest_privileged.json'
    for path in (public_path,private_path,manifest_path,CURATED):
        if not path.is_file():raise RuntimeError('Missing input '+str(path))
    if sha(public_path)!=SRC_SHA:raise RuntimeError('Frozen public contexts SHA mismatch; refusing to use a different experiment')
    public=jl(public_path);curated=jl(CURATED)
    check_public(public,curated)
    pub={r['context_id']:r for r in public}
    frozen={c['context_id']:{'context_id':c['context_id'],'target_role':c['target_role'],
             'query':safetext(c['query']),'source':c['source']} for c in curated}
    # Freeze the probe strings independently, before reading ANY privileged mapping.
    freeze_hash=hashlib.sha256(json.dumps(frozen,sort_keys=True).encode()).hexdigest()
    from adaptive_v11_active.io import sha256_json
    private=jl(private_path);manifest=j(manifest_path)
    if len(private)!=18 or len({x['context_id'] for x in private})!=18:raise RuntimeError('Private mapping incomplete')
    p_by={r['context_id']:r for r in private}
    if set(p_by)!=set(pub):raise RuntimeError('Private map does not match public contexts')
    sys_by={r['session_id']:r for r in manifest['systems']}
    groups=collections.defaultdict(dict)
    for ctx,m in p_by.items():
        if m['session_id'] not in sys_by:raise RuntimeError('Missing pilot system '+str(m['session_id']))
        if m['target_role']!=pub[ctx]['target_role']:raise RuntimeError('Role alignment failure')
        if m['condition'] not in ('TARGET_PRESENT','TARGET_ABSENT'):raise RuntimeError('Invalid condition')
        k=(m['target_role'],m['pair_id'])
        if m['condition'] in groups[k]:raise RuntimeError('Duplicate condition '+str(k))
        groups[k][m['condition']]=(ctx,m)
    if len(groups)!=9 or any(set(x)!={'TARGET_PRESENT','TARGET_ABSENT'} for x in groups.values()):
        raise RuntimeError('Expected exactly 9 paired contexts')
    chosen=[]
    for role in ROLES:
        rr=sorted((k,vals) for k,vals in groups.items() if k[0]==role)
        if len(rr)!=3:raise RuntimeError('Expected 3 matched backgrounds for '+role)
        chosen+=rr[:pairs_per_role]
    candidates=[];systems=[];review=[]
    for (_,pair_id),pair in chosen:
        role=pair['TARGET_PRESENT'][1]['target_role']
        ctx_pos,_=pair['TARGET_PRESENT'];ctx_neg,_=pair['TARGET_ABSENT']
        qpos=frozen[ctx_pos]['query'];qneg=frozen[ctx_neg]['query']
        if qpos==qneg: raise RuntimeError('Adaptive queries did not diverge across public responses: '+pair_id)
        arms=[('adaptive_a',qpos,ctx_pos),('adaptive_b',qneg,ctx_neg),('fixed',FIXED[role],None)]
        # Same complete query is sent to both configurations; no question-text confound within arm.
        for condition,(ctx,m) in sorted(pair.items()):
            source=sys_by[m['session_id']]
            systems.append(dict(source))
            for idx,(arm,q,origin_ctx) in enumerate(arms):
                candidates.append({'session_id':m['session_id'],'round':2,'split':'train',
                    'target_role':role,'candidate_index':idx,'query':q,'valid':True,
                    'probe_family':'help_me_help_you','strategy':arm,'arm':arm,
                    'origin_context_id':origin_ctx, 'pair_id':pair_id,'condition':condition,
                    'pilot_version':'MAS5_V54_QUERY_MATCHED_R2'})
        for arm,q,origin_ctx in arms:
            review.append({'target_role':role,'pair_id':pair_id,'arm':arm,
                           'source_context_id':origin_ctx,'query':q,
                           'word_count':len(q.split()),'paired_executions':2})
    n=6*len(chosen)
    if len(candidates)!=n or len(systems)!=2*len(chosen):raise RuntimeError('Bad cardinality')
    if len({(r['session_id'],r['candidate_index']) for r in candidates})!=n:
        raise RuntimeError('Duplicate (session,candidate_index)')
    # Validation: every query has exactly two equal messages across POS and NEG.
    for k,grp in group_by(candidates,lambda x:(x['pair_id'],x['arm'])).items():
        if len(grp)!=2 or {x['condition'] for x in grp}!={'TARGET_PRESENT','TARGET_ABSENT'} or len({x['query'] for x in grp})!=1:
            raise RuntimeError('Query matching violated '+str(k))
    root.mkdir(parents=True)
    output_manifest={'version':'MAS5_V54_PILOT_TRAIN_ONLY','systems':systems,'sha256':sha256_json(systems)}
    savej(root/'manifest_privileged.json',output_manifest)
    savel(root/'candidates_train_round_02.jsonl',candidates)
    savel(root/'curated_public_frozen.jsonl',list(frozen.values()))
    with (root/'human_review.tsv').open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['target_role','pair_id','arm','source_context_id','query','word_count','paired_executions'],delimiter='\t')
        w.writeheader();w.writerows(review)
    preflight={'status':'READY_FOR_HUMAN_REVIEW_AND_SEPARATE_COST_APPROVAL',
        'pairs_per_role':pairs_per_role,'roles':list(ROLES),'matched_pairs':len(chosen),
        'mas_executions_max':n,'unique_queries':len(review),
        'billable_external_API':True,'per_MAS_execution_up_to_provider_calls':5,
        'MAX_ESTIMATED_LLM_PROVIDER_CALLS_EXCLUDING_RETRIES':5*n,
        'frozen_v51_public_sha256':SRC_SHA,'probe_sha256':freeze_hash,
        'no_generator_training':True,'no_sensor_training':True,'no_selector_training':True,
        'probe_origin':'manual semantic rewriting of V5.3 public Before/After examples, not claimed as trained-generator output',
        'query_matched':'each adaptive probe from either prior public response is cross-executed on both systems; plus fixed HMHY',
        'important_limitations':[
          'The existing MAS is stateless: independent R2 questions, not a real dialogue continuation',
          'This revisits V4.9 role/pairs for development; NOT held-out MAS5 and NOT final AUC',
          'TARGET_ABSENT replaces the third worker, so target effects can be confounded by substitute',
          'Matched response divergence is not role inference, EIG, or a proven DPO preference',
          'Billed model tokens/cost unknown; provider calls may exceed estimate during retries',
          'Original R1 observations are reused, not re-executed during R2 pilot',
          'One manually curated adaptive message per public context; eight-candidate Selector not evaluated',
        ]}
    savej(root/'preflight.json',preflight)
    os.chmod(root/'manifest_privileged.json',0o600)
    os.chmod(root/'candidates_train_round_02.jsonl',0o600)
    print('MAS5_V54_PREPARED',json.dumps(preflight,ensure_ascii=False),flush=True)
    for r in review:print('  ',r['target_role'],r['pair_id'],r['arm'],'Q:',r['query'],flush=True)
    print('NO API CALLS. Inspect human_review.tsv BEFORE explicit paid submission.',flush=True)


def group_by(xs,fn):
    d=collections.defaultdict(list)
    for x in xs:d[fn(x)].append(x)
    return d


def execute(root):
    root=Path(root).resolve();pre=j(root/'preflight.json')
    total=pre['mas_executions_max']
    if os.environ.get('V54_APPROVE_BILLABLE_CALLS')!=str(total):
        raise RuntimeError(f'Opt-in required: V54_APPROVE_BILLABLE_CALLS={total}')
    if not os.environ.get('OPENAI_API_KEY'): raise RuntimeError('OPENAI_API_KEY missing')
    c=jl(root/'candidates_train_round_02.jsonl')
    if len(c)!=total:raise RuntimeError('Execution cap changed')
    if (root/'responses_train_round_02.jsonl').exists():
        raise RuntimeError('Responses already exist; refusing duplicate paid execution')
    # The established executor caches successful requests incrementally for safe resumption.
    cmd=[sys.executable,'-u','scripts/adaptive_v11_execute_round.py',
         '--round','2','--split','train','--root',str(root),
         '--manifest',str(root/'manifest_privileged.json'),
         '--workers','2','--retries','1']
    print(f'MAS5_V54_BILLABLE_EXECUTION {total} MAS executions; maximum 5 model calls/execution without retries, provider billing unknown',flush=True)
    subprocess.run(cmd,cwd=PROJECT,check=True)
    r=jl(root/'responses_train_round_02.jsonl')
    if len(r)!=total or any(x.get('execution_status')!='ok' for x in r):
        raise RuntimeError('Incomplete target executions; inspect cache and logs')
    print('MAS5_V54_EXECUTION_COMPLETE',len(r),flush=True)


def report(root):
    root=Path(root).resolve();pre=j(root/'preflight.json')
    c=jl(root/'candidates_train_round_02.jsonl')
    r=jl(root/'responses_train_round_02.jsonl')
    if not (len(r)==len(c)==pre['mas_executions_max']):
        raise RuntimeError('Bad experiment length')
    by={(x['session_id'],x['candidate_index']):x for x in r}
    if len(by)!=len(r):raise RuntimeError('Duplicate answers')
    grouped=group_by(c,lambda x:(x['target_role'],x['pair_id']))
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    stats=[];pairs=[]
    for (role,pair_id),candidates in sorted(grouped.items()):
        if len(candidates)!=6:raise RuntimeError('Expected 6 executions per pair')
        # Fit unsupervised TF-IDF on all six responses for this background, no label fitting.
        responses=[by[(x['session_id'],x['candidate_index'])] for x in candidates]
        if any(x['execution_status']!='ok' for x in responses):raise RuntimeError('Incomplete responses')
        docs=[str(x['final_output']) for x in responses]
        tf=TfidfVectorizer(analyzer='char',ngram_range=(3,5),max_features=16000,sublinear_tf=True)
        emb=tf.fit_transform(docs)
        dist=1-cosine_similarity(emb)
        arm_stats=[]
        for arm in ('adaptive_a','adaptive_b','fixed'):
            indices={c['condition']:i for i,c in enumerate(candidates) if c['arm']==arm}
            if set(indices)!={'TARGET_PRESENT','TARGET_ABSENT'}:raise RuntimeError('Missing condition')
            ip,in_=indices['TARGET_PRESENT'],indices['TARGET_ABSENT']
            p,n=responses[ip],responses[in_]
            role_activated=role in p.get('selected_roles_training_only',[])
            if role in n.get('selected_roles_training_only',[]):raise RuntimeError('Target activated in absent system; invalid label')
            d=round(max(0.,float(dist[ip,in_])),6)
            data={'target_role':role,'pair_id':pair_id,'arm':arm,
              'response_text_distance_diagnostic':d,
              'present_selected_target_training_only':bool(role_activated),
              'present_response_characters':len(docs[ip]),'absent_response_characters':len(docs[in_]),
              'absolute_length_delta':abs(len(docs[ip])-len(docs[in_])),
              'present_response_sha256':hashlib.sha256(docs[ip].encode()).hexdigest(),
              'absent_response_sha256':hashlib.sha256(docs[in_].encode()).hexdigest(),
              'warning':'Descriptive paired response distance is not AUC or information gain'}
            arm_stats.append(data);stats.append(data)
        adaptive=[x for x in arm_stats if x['arm']!='fixed']
        fixed=[x for x in arm_stats if x['arm']=='fixed'][0]
        pairs.append({'target_role':role,'pair_id':pair_id,
          'adaptive_mean_text_distance':statistics.mean(x['response_text_distance_diagnostic'] for x in adaptive),
          'fixed_text_distance':fixed['response_text_distance_diagnostic'],
          'adaptive_minus_fixed_text_distance_diagnostic':round(statistics.mean(x['response_text_distance_diagnostic'] for x in adaptive)-fixed['response_text_distance_diagnostic'],6),
          'adaptive_present_activation_rate_training_only':statistics.mean(x['present_selected_target_training_only'] for x in adaptive),
          'fixed_present_activation_training_only':fixed['present_selected_target_training_only']})
    summary={'status':'DESCRIPTIVE_PAIRED_R2_SIGNAL_ONLY',
      'mas_executions_completed':len(r),'role_count':len({x['target_role'] for x in c}),
      'matched_pairs':len(pairs),'adaptive_query_arms':2*len(pairs),'fixed_query_arms':len(pairs),
      'mean_adaptive_minus_fixed_text_distance_diagnostic':round(statistics.mean(x['adaptive_minus_fixed_text_distance_diagnostic'] for x in pairs),6),
      'adaptive_present_activation_rate_training_only':round(statistics.mean(x['present_selected_target_training_only'] for x in stats if x['arm']!='fixed'),6),
      'fixed_present_activation_rate_training_only':round(statistics.mean(x['present_selected_target_training_only'] for x in stats if x['arm']=='fixed'),6),
      'not_valid_auc':True,'not_eig_or_dpo_reward':True,'not_statistically_significant':True,
      'not_mas5_end_to_end':True,'not_stateful_conversation':True,
      'warning':'Distances may reflect replacement-role confounds or content/style differences. This pilot cannot establish reliable discrimination.'}
    savej(root/'report_diagnostic.json',summary)
    savel(root/'arms_diagnostic.jsonl',stats)
    savel(root/'pairs_diagnostic.jsonl',pairs)
    print('MAS5_V54_REPORT',json.dumps(summary,indent=2,ensure_ascii=False),flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('phase',choices=['prepare','execute','report'])
    p.add_argument('--root',required=True)
    p.add_argument('--v51-root')
    p.add_argument('--v49-pilot')
    p.add_argument('--pairs-per-role',type=int,default=1)
    a=p.parse_args()
    if a.phase=='prepare':
        if not a.v51_root or not a.v49_pilot: p.error('prepare requires --v51-root and --v49-pilot')
        prepare(a.root,a.v51_root,a.v49_pilot,a.pairs_per_role)
    elif a.phase=='execute':execute(a.root)
    else: report(a.root)

if __name__=='__main__':main()
