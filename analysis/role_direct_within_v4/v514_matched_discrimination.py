#!/usr/bin/env python3
"""V5.14: audit-first 30-call, matched PRESENT/ABSENT role probe diagnostic.

Offline preparation only; paid execute requires separate environment approval.
Each of 15 queries is sent unchanged to two MAS variants which share background,
mode and style but differ by REPLACING the role. One background pair per role.
This is causal substitution, not pure role ablation. It is NOT an independent test
of cross-composition generalization and is not a Generator training run.
"""
from __future__ import annotations
import argparse, collections, csv, datetime, hashlib, json, math, os, statistics, subprocess, sys
from pathlib import Path

ROOT=Path(os.environ.get('V514_PROJECT_ROOT', str(Path.home()/'mas_rl_detective'))).resolve()
BASE=ROOT/'artifacts/role_direct_within_v4/matched_v514'
CF=ROOT/'artifacts/role_direct_matched_cf_v1/manifest_privileged.json'
MAS5=ROOT/'artifacts/role_direct_routing_pilot_v1/manifest_privileged.json'
ROLES=('code','planning','research')
NQUESTIONS=5
CALLS=30
QUESTIONS={
 'code':[
  ('direct_artifact','I am troubleshooting PostgreSQL latency after a migration. Please write a standalone Python 3 script that reads a JSON EXPLAIN plan and prints the three most expensive nodes. Include runnable unit tests for missing keys and nested plans.'),
  ('counterexample','I have two PostgreSQL queries with identical estimated cost but very different actual execution times after a migration. Can you produce a small reproducible SQL example and a script to compare EXPLAIN ANALYZE output without making changes to production?'),
  ('debug_failure','A Python parser for PostgreSQL EXPLAIN (FORMAT JSON) crashes when a plan contains a nested CTE and a missing Actual Rows field. Show a minimal failing fixture, a corrected parsing function, and a regression test.'),
  ('constraints','I need to automate safe index recommendations for slow PostgreSQL queries. What database permissions, query-plan samples, and safety constraints should I provide before you can draft a read-only Python utility with precise command-line arguments?'),
  ('verification','Please implement a Python function that compares two PostgreSQL query plans and reports whether the estimated scan type changed, with unit tests for sequential scans, index scans, and absent plan nodes.'),
 ],
 'planning':[
  ('direct_artifact','We need to move a production PostgreSQL database with a two-hour maintenance window. Build an ordered migration plan with dependencies, owners, checkpoints, rollback triggers, and a concrete critical path.'),
  ('counterexample','Our database migration plan assumes replicas will catch up before cutover, but replication lag unexpectedly doubles. Show a revised dependency-aware schedule and identify which milestones must move or be canceled.'),
  ('debug_failure','A database cutover failed because application configuration changed before replica verification finished. Provide a corrected sequence of tasks, a dependency table, and explicit decision gates that prevent this ordering error.'),
  ('constraints','I am planning a zero-downtime PostgreSQL migration across two regions. Which specific timing estimates, dependencies, failure tolerances, and ownership constraints should I provide so you can produce a feasible plan and rollback decision tree?'),
  ('verification','Compare two plans for a PostgreSQL schema migration: one migrates tables in parallel and one in dependency order. Give measurable acceptance criteria, a milestone schedule, and a way to identify the plan that violates prerequisites.'),
 ],
 'research':[
  ('direct_artifact','After migrating PostgreSQL, query latency increased threefold. Compare the competing hypotheses of changed statistics, missing indexes, and connection-pool contention, and propose a falsifiable investigation plan with expected observations.'),
  ('counterexample','We suspect a missing index caused PostgreSQL latency after migration, but EXPLAIN shows the same indexes and scan types as before. What alternative causal hypotheses and discriminating measurements would you test first?'),
  ('debug_failure','A team concluded that database migration caused latency because both happened on the same day. Critique this inference, identify confounders, and specify a controlled measurement design to test causality.'),
  ('constraints','I am studying why query response times rose after a PostgreSQL migration. What minimal evidence should I provide to distinguish optimizer regressions from infrastructure contention, and what result would falsify each explanation?'),
  ('verification','Design a small controlled experiment to determine whether stale PostgreSQL table statistics explain a latency regression. State the hypothesis, controls, measurements, uncertainty, and what outcomes would reject the hypothesis.'),
 ],
}

def getj(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def getl(path):return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]
def putj(path,data):
    with Path(path).open('x',encoding='utf-8') as f:json.dump(data,f,indent=2,ensure_ascii=False);f.write('\n')
def putl(path,items):
    with Path(path).open('x',encoding='utf-8') as f:
        for item in items:f.write(json.dumps(item,ensure_ascii=False)+'\n')
def tsv(path,records,columns):
    with Path(path).open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,delimiter='\t',fieldnames=columns,extrasaction='ignore');w.writeheader();w.writerows(records)
def stamp(s):return hashlib.sha256(s.encode('utf-8')).hexdigest()
def safe_run(root):
    p=Path(root).resolve();b=BASE.resolve()
    if p.parent!=b or not p.name.startswith('pilot_'):raise RuntimeError('Run must be inside v514 output area')
    return p

def validate_questions():
    if set(QUESTIONS)!=set(ROLES) or any(len(QUESTIONS[r])!=NQUESTIONS for r in ROLES):raise RuntimeError('Bad probe bank cardinality')
    seen=set()
    for role in ROLES:
        for family,q in QUESTIONS[role]:
            if not 22<=len(q.split())<=115 or not (q.endswith(('.', '?'))):
                raise RuntimeError(f'Question formatting failed {role}: {q}')
            if q in seen:raise RuntimeError('Duplicated query')
            seen.add(q)
    return True

def valid_pairs(cf,mas5):
    dev={frozenset(x['roles']) for x in mas5['systems'] if x.get('split')=='dev'}
    grouped=collections.defaultdict(dict)
    for s in cf['systems']:
        key=(str(s.get('target_role')),str(s.get('pair_id')))
        condition=str(s.get('condition'))
        if condition in grouped[key]:raise RuntimeError(f'Duplicate condition {key}')
        grouped[key][condition]=s
    valid=collections.defaultdict(list)
    for (role,pair),sides in sorted(grouped.items()):
        if role not in ROLES:continue
        if set(sides)!={'TARGET_PRESENT','TARGET_ABSENT'}:continue
        p,n=sides['TARGET_PRESENT'],sides['TARGET_ABSENT']
        if len(p['roles'])!=3 or len(n['roles'])!=3 or p['roles'][:2]!=n['roles'][:2]:continue
        if p['roles'][2]!=role or role in n['roles'] or set(p['roles'])==set(n['roles']):continue
        if (p['mode'],p['style'])!=(n['mode'],n['style']):continue
        if frozenset(p['roles']) in dev or frozenset(n['roles']) in dev:continue
        valid[role].append((pair,p,n))
    return valid

def prepare(run, cf_path=CF, mas5_path=MAS5):
    validate_questions();run=safe_run(run)
    if run.exists():raise RuntimeError(f'Refusing overwrite: {run}')
    if not Path(cf_path).is_file() or not Path(mas5_path).is_file():raise RuntimeError('Missing original CF/MAS5 manifests')
    cf,mas=getj(cf_path),getj(mas5_path)
    pairs=valid_pairs(cf,mas)
    selected={}
    for role in ROLES:
        options=pairs[role]
        if not options:raise RuntimeError(f'No valid matched CF background for {role}; no paid work prepared')
        # Deliberately use a later preexisting background rather than v49/v50 first three if possible.
        # Never call this a sealed holdout: historical training may still share compositions.
        selected[role]=options[3] if len(options)>3 else options[-1]
    from adaptive_v11_active.io import sha256_json
    systems=[];candidates=[];preview=[]
    for role in ROLES:
        pid,p,n=selected[role]
        newpair='v514_'+role
        for condition,orig in [('TARGET_PRESENT',p),('TARGET_ABSENT',n)]:
            sid=f'{newpair}_{"PRESENT" if condition=="TARGET_PRESENT" else "ABSENT"}'
            systems.append({**orig,'session_id':sid,'pair_id':newpair,'split':'train','condition':condition})
            for i,(family,q) in enumerate(QUESTIONS[role]):
                candidates.append({'session_id':sid,'target_role':role,'round':1,'split':'train',
                    'candidate_index':i,'query':q,'valid':True,'family':family,
                    'pair_id':newpair,'condition':condition,'original_cf_pair':pid})
        preview.extend({'role':role,'candidate_index':i,'family':fam,'query':q,'pair_id':newpair,
                        'original_cf_pair':pid} for i,(fam,q) in enumerate(QUESTIONS[role]))
    assert len(candidates)==CALLS and len(systems)==6 and len(preview)==15
    if len({(x['session_id'],x['candidate_index']) for x in candidates})!=CALLS:raise RuntimeError('Nonunique candidates')
    run.mkdir(parents=True)
    putj(run/'manifest_privileged.json',{'version':'MAS5_V514_DIAGNOSTIC','systems':systems,'sha256':sha256_json(systems)})
    putl(run/'candidates_train_round_01.jsonl',candidates)
    putl(run/'public_probe_preview.jsonl',preview)
    tsv(run/'public_probe_preview.tsv',preview,['role','candidate_index','family','query','original_cf_pair'])
    putj(run/'preflight.json',{'status':'PREPARED_OFFLINE_NO_API','calls_cap':CALLS,
      'roles':list(ROLES),'questions_per_role':NQUESTIONS,'matched_backgrounds_per_role':1,
      'pairs':{r:{'original_cf_pair':selected[r][0],'mode':selected[r][1]['mode'],
                   'style':selected[r][1]['style'],'present_roles':selected[r][1]['roles'],
                   'absent_roles':selected[r][2]['roles'],'available_valid_cf_backgrounds':len(pairs[r])} for r in ROLES},
      'all_queries_fixed_manual_not_generator_model':True,'question_identical_between_present_absent':True,
      'not_new_composition_sealed_test':True,
      'potential_model_training_composition_overlap':'unknown; frozen V5113 model trained on prior observations',
      'design_limitation':'ABSENT substitutes one other worker. Substitution, not a pure role knockout. 1 background/role prevents composition-heldout generalization estimate.',
      'cost':'30 MAS executions; each MAS may call multiple billable API models; unknown dollars',
      'requires_separate_execution_confirmation':'V514_APPROVE_MAS_EXECUTIONS=30'})
    print('V514_PREPARED',run,'30 possible billable MAS calls; 15 identical matched questions; no API calls',flush=True)
    for p in preview:print('V514_QUERY',p['role'],p['candidate_index'],p['family'],p['query'],flush=True)
    return run

def execute(run):
    run=safe_run(run);pf=getj(run/'preflight.json')
    if pf['calls_cap']!=CALLS:raise RuntimeError('Wrong preflight budget')
    if os.environ.get('V514_APPROVE_MAS_EXECUTIONS')!='30':
        raise RuntimeError('Explicit approval required: V514_APPROVE_MAS_EXECUTIONS=30')
    if not os.environ.get('OPENAI_API_KEY'):raise RuntimeError('Missing OPENAI_API_KEY')
    response=run/'responses_train_round_01.jsonl'
    if response.exists():raise RuntimeError('Response file already exists; refusing duplicate billable run')
    candidates=getl(run/'candidates_train_round_01.jsonl')
    if len(candidates)!=30 or any(not x['valid'] for x in candidates):raise RuntimeError('Unexpected executable candidates')
    k=collections.defaultdict(dict)
    for c in candidates:k[(c['target_role'],c['candidate_index'])][c['condition']]=c['query']
    if len(k)!=15 or any(set(z)!={'TARGET_PRESENT','TARGET_ABSENT'} or len(set(z.values()))!=1 for z in k.values()):
        raise RuntimeError('Questions are not exactly paired')
    cmd=[sys.executable,'-u','scripts/adaptive_v11_execute_round.py','--round','1','--split','train',
         '--root',str(run),'--manifest',str(run/'manifest_privileged.json'),'--workers','2','--retries','0']
    print('V514_EXEC_START max 30 unique MAS executions, target backend may issue multiple charged calls each',flush=True)
    subprocess.run(cmd,cwd=ROOT,check=True)
    # If interruption occurs, target executor caches successful calls to avoid duplicate API requests on resume.
    rows=getl(response)
    if len(rows)!=30 or any(x.get('execution_status')!='ok' or not x.get('final_output') for x in rows):
        raise RuntimeError('Incomplete run; do NOT delete cache; investigate before retrying')
    if any(x.get('cache_hit') for x in rows):print('V514_CACHE_HITS',sum(x['cache_hit'] for x in rows),flush=True)
    print('V514_EXEC_COMPLETE 30/30 MAS responses',flush=True)
    return len(rows)

def auc(pos,neg):
    if not pos or not neg:return None
    return sum(float(a>b)+.5*float(a==b) for a in pos for b in neg)/(len(pos)*len(neg))

def role_predictions(run,bykey):
    """Optional frozen, previously trained V5113 models; no fitting on pilot answers."""
    try:
        import joblib
        from analysis.role_direct_within_v4 import v5111_rolewise_verifier as base
    except ImportError as exc:
        return {},f'Unavailable optional frozen model dependencies: {exc}'
    models_dir=ROOT/'artifacts/role_direct_within_v4/sensor_v5113'
    sources=sorted(models_dir.glob('rolewise_*'),reverse=True)
    src=next((x for x in sources if (x/'model_answer_only.joblib').is_file() and (x/'model_length_only.joblib').is_file()),None)
    if src is None:return {},'Frozen V5113 models unavailable; only model-free response difference metrics reported'
    predictions={}
    for arm in ('answer_only','length_only'):
        record=joblib.load(src/f'model_{arm}.joblib')
        if record.get('arm')!=arm or not set(ROLES).issubset(record['models']):raise RuntimeError('V5113 frozen models do not match expected roles')
        for role in ROLES:
            data=[];order=[]
            for i in range(NQUESTIONS):
                for cond in ('TARGET_PRESENT','TARGET_ABSENT'):
                    row=bykey[(role,i,cond)]; data.append({'role':role,'query':row['query'],'answer':row['final_output']})
                    order.append((role,i,cond))
            scores=base.predict(record['models'][role],data,arm)
            for k,score in zip(order,scores):predictions[(arm,*k)]=float(score)
    return predictions,str(src)

def report(run):
    run=safe_run(run)
    if (run/'report.json').exists():raise RuntimeError('Report already exists; refusing overwrite')
    cands=getl(run/'candidates_train_round_01.jsonl');answers=getl(run/'responses_train_round_01.jsonl')
    if len(answers)!=CALLS or len(cands)!=CALLS:raise RuntimeError('Need all 30 executed responses')
    bykey={};expect={(x['session_id'],x['candidate_index']):x for x in cands}
    if len(expect)!=CALLS:raise RuntimeError('Duplicate candidates')
    for r in answers:
        k=(r['session_id'],r['candidate_index'])
        if k not in expect or k in bykey:raise RuntimeError('Unmatched or repeated MAS response')
        c=expect[k]
        if (r['query'],r['target_role'],r['condition'])!=(c['query'],c['target_role'],c['condition']):raise RuntimeError('Candidate/response mismatch')
        if r['execution_status']!='ok' or not str(r['final_output']).strip():raise RuntimeError('Non-success MAS response')
        bykey[(c['target_role'],c['candidate_index'],c['condition'])]=r
    if len(bykey)!=CALLS:raise RuntimeError('Missing roles/questions/conditions')
    preds,modelpath=role_predictions(run,bykey)
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    pairrows=[];totals=[]
    for role in ROLES:
        rolepairs=[]
        for i,(family,q) in enumerate(QUESTIONS[role]):
            p=bykey[(role,i,'TARGET_PRESENT')];n=bykey[(role,i,'TARGET_ABSENT')]
            if p['query']!=n['query'] or p['query']!=q:raise RuntimeError('Matched pair query differs')
            texts=[p['final_output'],n['final_output']]
            vectors=TfidfVectorizer(analyzer='char',ngram_range=(3,5),min_df=1).fit_transform(texts)
            sim=float(cosine_similarity(vectors[0],vectors[1])[0,0])
            row={'role':role,'slot':i,'family':family,'query':q,'present_chars':len(texts[0]),
                 'absent_chars':len(texts[1]),'delta_chars':len(texts[0])-len(texts[1]),
                 'cosine_similarity_char_3_5':round(sim,5),'exact_same_answer':int(texts[0]==texts[1]),
                 'oracle_target_activated_when_present_eval_only':int(role in p.get('selected_roles_training_only',[])) if 'selected_roles_training_only' in p else '',
                 'present_excerpt':texts[0][:330].replace('\n',' '),'absent_excerpt':texts[1][:330].replace('\n',' ')}
            for arm in ('answer_only','length_only'):
                if preds:
                    ps=preds[(arm,role,i,'TARGET_PRESENT')];ns=preds[(arm,role,i,'TARGET_ABSENT')]
                    row[f'{arm}_present_score']=round(ps,7);row[f'{arm}_absent_score']=round(ns,7)
                    row[f'{arm}_paired_win']=float(ps>ns)+.5*float(ps==ns)
                    row[f'{arm}_score_delta']=round(ps-ns,7)
            pairrows.append(row);rolepairs.append(row)
        summary={'role':role,'pairs':len(rolepairs),'mean_similarity':round(statistics.mean(x['cosine_similarity_char_3_5'] for x in rolepairs),5),
                 'exact_identical_answers':sum(x['exact_same_answer'] for x in rolepairs),
                 'mean_abs_length_delta':round(statistics.mean(abs(x['delta_chars']) for x in rolepairs),2),
                 'oracle_activation_rate_eval_only':round(statistics.mean(x['oracle_target_activated_when_present_eval_only'] for x in rolepairs),5) if all(isinstance(x['oracle_target_activated_when_present_eval_only'],int) for x in rolepairs) else None}
        if preds:
            for arm in ('answer_only','length_only'):
                summary[f'{arm}_paired_win_rate']=round(statistics.mean(x[f'{arm}_paired_win'] for x in rolepairs),5)
                summary[f'{arm}_within_pairset_auc']=round(auc([x[f'{arm}_present_score'] for x in rolepairs],
                                             [x[f'{arm}_absent_score'] for x in rolepairs]),5)
        totals.append(summary)
    pair_cols=['role','slot','family','query','present_chars','absent_chars','delta_chars','cosine_similarity_char_3_5',
               'exact_same_answer','oracle_target_activated_when_present_eval_only','answer_only_present_score','answer_only_absent_score','answer_only_score_delta',
               'answer_only_paired_win','length_only_score_delta','length_only_paired_win','present_excerpt','absent_excerpt']
    tsv(run/'matched_pairs.tsv',pairrows,pair_cols)
    summary_cols=['role','pairs','mean_similarity','exact_identical_answers','mean_abs_length_delta',
                  'oracle_activation_rate_eval_only','answer_only_paired_win_rate','answer_only_within_pairset_auc',
                  'length_only_paired_win_rate','length_only_within_pairset_auc']
    tsv(run/'comparison.tsv',totals,summary_cols)
    result={'status':'COMPLETE_DIAGNOSTIC_NOT_HELDOUT_GENERALIZATION','mas_executions':CALLS,
       'total_matched_probes':15,'matched_backgrounds_per_role':1,'matched_role_comparison':totals,
       'frozen_models_source':modelpath,'trained_or_fitted_on_pilot':False,
       'limitations':['Only one matched background per role: 5 probes are correlated by shared pair of MAS configurations; AUC is NOT across new MAS compositions',
                      'Replacement worker in ABSENT changes a second factor; cannot attribute the entire difference to missing target role',
                      'Handwritten fixed probes, not trained Generator or adaptive Selector',
                      'Privileged routing/activation is used only in retrospective evaluation after response collection; never provided to frozen scorer',
                      'Frozen models might already have seen role compositions in earlier historical training; no guarantees of sealed evaluation',
                      'Response cosine is descriptive similarity; low similarity alone is NOT evidence of presence detection',
                      'Not comparable directly to 20-role Within-MAS AUC or Exact Top-3.']}
    putj(run/'report.json',result)
    lines=['# V5.14 Matched role discrimination — diagnostic only','',
           'Same exact probe for TARGET_PRESENT vs TARGET_ABSENT; one background per role.','',
           '| Role | 5 paired probes | Text similarity | Identical answers | Answer-only paired win | Length-only paired win |',
           '|---|---:|---:|---:|---:|---:|']
    for x in totals:
        def fmt(k):return str(x[k]) if k in x else 'N/A'
        lines.append(f'| {x["role"]} | {x["pairs"]} | {x["mean_similarity"]} | {x["exact_identical_answers"]} | {fmt("answer_only_paired_win_rate")} | {fmt("length_only_paired_win_rate")} |')
    lines+=['','**Interpretation:** Paired win uses a frozen classifier, not a classifier fitted on this pilot. Identical query is a negative control for query-only information, but response length can remain confounded. Five paired samples in one background are *not* five independent MAS configurations.','',
       'The ABSENT system has a substitute third worker. A difference may be due to the replacement worker, not only absence of target.','',
       'See matched_pairs.tsv for questions and representative answer excerpts.']
    with (run/'report.md').open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
    for x in totals:print('V514_RESULT',json.dumps(x),flush=True)
    print('V514_FINISHED',run,flush=True)
    return result

def selftest():
    validate_questions()
    assert auc([0.6,0.9],[0.2,0.8])==.75
    assert auc([1,1],[1,1])==.5
    # Synthetic two-sided role pair ensures substitution is validated.
    cf={'systems':[{'target_role':'code','pair_id':'a','condition':c,'roles':['data','operations',r],
             'mode':'parallel','style':'concise'} for c,r in [('TARGET_PRESENT','code'),('TARGET_ABSENT','planning')]]}
    mas={'systems':[{'split':'dev','roles':['code','writing','research']}]}
    assert len(valid_pairs(cf,mas)['code'])==1
    cf['systems'][1]['roles'][0]='writing';assert not valid_pairs(cf,mas)['code']
    print('V514_SELFTEST_OK',flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['selftest','prepare','execute','report']);p.add_argument('--root',default=None)
    p.add_argument('--cf',default=str(CF));p.add_argument('--mas5',default=str(MAS5))
    a=p.parse_args()
    if a.phase=='selftest':selftest();return
    if a.root is None:raise SystemExit('--root required')
    if a.phase=='prepare':prepare(a.root,a.cf,a.mas5)
    if a.phase=='execute':execute(a.root)
    if a.phase=='report':report(a.root)
if __name__=='__main__':main()
