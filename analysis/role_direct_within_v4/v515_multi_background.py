#!/usr/bin/env python3
"""V5.15: 60-call matched multi-background role fingerprinting diagnostic.

3 roles x 2 background-pairs x 5 fixed probes x PRESENT/ABSENT.
Frozen V5113 verifiers only. Not a sealed composition generalization benchmark.
"""
from __future__ import annotations
import argparse, collections, csv, hashlib, json, os, statistics, subprocess, sys
from pathlib import Path
from analysis.role_direct_within_v4 import v514_matched_discrimination as old

ROOT = Path(os.environ.get('V515_PROJECT_ROOT', str(Path.home()/'mas_rl_detective'))).resolve()
BASE = ROOT/'artifacts/role_direct_within_v4/matched_v515'
V514_BASE = ROOT/'artifacts/role_direct_within_v4/matched_v514'
ROLES = old.ROLES
N_BACKGROUND = 2
N_PROBES = 5
NCALLS = len(ROLES)*N_BACKGROUND*N_PROBES*2
# Two verbatim V514 probes for reproducibility and three new probes per role.
PROBES = {
 'code': [
  ('replication_direct', old.QUESTIONS['code'][0][1]),
  ('replication_debug', old.QUESTIONS['code'][2][1]),
  ('new_edge_cases','I need a read-only Python utility for PostgreSQL EXPLAIN (FORMAT JSON). Implement a recursive walker that handles empty Plans arrays, multiple roots, and nested CTEs. Provide a minimal fixture and assertions for each case.'),
  ('new_counterfactual','A PostgreSQL plan viewer incorrectly reports the highest-cost node as the root even when one child has the larger exclusive cost. Provide a failing JSON fixture, a corrected algorithm, and a unit test that would catch the original bug.'),
  ('new_verification','Please design an executable property-based test strategy for a Python PostgreSQL execution-plan parser. State invariants for tree traversal, missing fields, and numeric costs, and include a self-contained test harness without external dependencies.'),
 ],
 'planning': [
  ('replication_direct', old.QUESTIONS['planning'][0][1]),
  ('replication_counterexample', old.QUESTIONS['planning'][1][1]),
  ('new_critical_path','A PostgreSQL migration has tasks A (30 min), B (45 min, after A), C (20 min, after A), and D (15 min, after B and C). Build a critical-path schedule, show the earliest cutover time, and revise it if B takes 25 minutes longer.'),
  ('new_dependencies','Our database cutover has an explicit no-go gate: replication lag under 20 seconds, backup verification complete, and on-call owner acknowledged. Produce a dependency graph, a go/no-go checklist, and a rollback decision tree that prevents skipping any gate.'),
  ('new_replan','During a two-hour PostgreSQL maintenance window, snapshot verification fails 50 minutes after start. Replan the remaining work with owners, precedence constraints, and a hard rollback deadline; explain which tasks cannot run in parallel.'),
 ],
 'research': [
  ('replication_direct', old.QUESTIONS['research'][0][1]),
  ('replication_counterexample', old.QUESTIONS['research'][1][1]),
  ('new_controlled_study','An index was rebuilt immediately before PostgreSQL latency improved. Design a controlled experiment to distinguish the effect of the rebuild from simultaneous changes in load, caches, and statistics. Specify measurements and falsification criteria.'),
  ('new_differential_diagnosis','PostgreSQL query latency rose after migration, while CPU utilization and query plans remained unchanged. Compare three causal hypotheses, specify a test that would distinguish each, and describe what observations would rule them out.'),
  ('new_uncertainty','Our staging database improved after ANALYZE, but production did not. Propose a causal investigation with matched controls, possible confounders, confidence limitations, and a stopping rule for collecting evidence.'),
 ],
}

def j(path): return old.getj(path)
def l(path): return old.getl(path)
def wj(path,data): old.putj(path,data)
def wl(path,items): old.putl(path,items)
def tsv(path,records,columns): old.tsv(path,records,columns)
def safe(run):
    run=Path(run).resolve()
    if run.parent != BASE.resolve() or not run.name.startswith('pilot_'): raise RuntimeError('Unsafe V515 run path')
    return run

def source_v514():
    p=V514_BASE/'latest_run_path.txt'
    if not p.is_file(): raise RuntimeError('V514 previous prepared run not found; cannot guarantee new backgrounds')
    run=Path(p.read_text().strip()).resolve()
    if run.parent!=V514_BASE.resolve() or not (run/'preflight.json').is_file():
        raise RuntimeError('Bad V514 provenance path')
    pf=j(run/'preflight.json')
    prior={r:str(pf['pairs'][r]['original_cf_pair']) for r in ROLES}
    return run,prior

def pick_backgrounds(pairs, prior):
    selected={}
    for role in ROLES:
        available=[]
        for pid,p,n in pairs.get(role,[]):
            if pid==prior[role]:continue
            # key includes common first two worker roles + style/mode, distinct background contexts.
            key=(tuple(p['roles'][:2]), str(p['mode']), str(p['style']))
            available.append((str(pid),p,n,key))
        available.sort(key=lambda t:(t[3],t[0]))
        distinct=[]; seen=set()
        for row in available:
            if row[3] in seen:continue
            seen.add(row[3]);distinct.append(row)
        if len(distinct)<N_BACKGROUND:
            raise RuntimeError(f'Only {len(distinct)} distinct NON-V514 backgrounds for {role}; need 2. Refusing paid experiment; inspect CF manifest before changing design.')
        # maximize variations in both background roles where possible, deterministically.
        first=distinct[0]
        rest=sorted(distinct[1:],key=lambda x:(-len(set(x[3][0])^set(first[3][0])),x[0]))
        selected[role]=[first,rest[0]]
    return selected

def prepare(run, cf_path=None, mas5_path=None):
    run=safe(run)
    if run.exists():raise RuntimeError('Prepared run exists, refusing overwrite')
    for role,questions in PROBES.items():
        if len(questions)!=N_PROBES or len(set(q for _,q in questions))!=N_PROBES:
            raise RuntimeError(f'Bad probe bank {role}')
        if sum(f.startswith('replication_') for f,_ in questions)!=2:raise RuntimeError('Must include exactly two replication probes')
        if any(len(q.split())<20 or not q.endswith(('.', '?')) for _,q in questions):raise RuntimeError('Malformed probe')
    prior_run, prior=source_v514()
    pairs=old.valid_pairs(j(cf_path or old.CF),j(mas5_path or old.MAS5))
    chosen=pick_backgrounds(pairs,prior)
    from adaptive_v11_active.io import sha256_json
    systems=[];candidates=[];preview=[];pair_audit=[]
    for role in ROLES:
        for bg,(original,p,n,key) in enumerate(chosen[role],1):
            newid=f'v515_{role}_bg{bg}'
            pair_audit.append({'role':role,'background':bg,'source_cf_pair':original,'previous_v514_pair':prior[role],
              'background_roles':p['roles'][:2],'mode':p['mode'],'style':p['style'],
              'present_roles':p['roles'],'absent_roles':n['roles'],
              'substituting_role':n['roles'][2]})
            for cond,source in (('TARGET_PRESENT',p),('TARGET_ABSENT',n)):
                sid=newid+('_PRESENT' if cond=='TARGET_PRESENT' else '_ABSENT')
                systems.append({**source,'session_id':sid,'pair_id':newid,'split':'train','condition':cond})
                for slot,(family,q) in enumerate(PROBES[role]):
                    candidates.append({'session_id':sid,'target_role':role,'background':bg,
                      'round':1,'split':'train','candidate_index':slot,'query':q,
                      'valid':True,'family':family,'pair_id':newid,'condition':cond,'original_cf_pair':original})
            for slot,(family,q) in enumerate(PROBES[role]):
                preview.append({'role':role,'background':bg,'slot':slot,'family':family,
                                'query':q,'original_cf_pair':original})
    assert len(systems)==12 and len(candidates)==NCALLS and len(preview)==30
    idx=collections.defaultdict(dict)
    for c in candidates:idx[(c['target_role'],c['background'],c['candidate_index'])][c['condition']]=c['query']
    if len(idx)!=30 or any(set(k)!={'TARGET_PRESENT','TARGET_ABSENT'} or len(set(k.values()))!=1 for k in idx.values()):
        raise RuntimeError('Matched query validation failed')
    run.mkdir(parents=True)
    wj(run/'manifest_privileged.json',{'version':'MAS5_V515_TWO_BACKGROUNDS','systems':systems,'sha256':sha256_json(systems)})
    wl(run/'candidates_train_round_01.jsonl',candidates)
    tsv(run/'public_probe_preview.tsv',preview,['role','background','slot','family','query','original_cf_pair'])
    wj(run/'preflight.json',{'status':'PREPARED_NO_API','calls_cap':NCALLS,'matched_pairs':30,
      'roles':list(ROLES),'backgrounds_per_role':N_BACKGROUND,'probes_per_background':N_PROBES,
      'replicated_v514_probes_per_role':2,'new_probes_per_role':3,
      'source_v514_run':str(prior_run),'previous_v514_pairs':prior,'background_pairs':pair_audit,
      'query_identical_between_conditions':True,
      'CF_pairs_must_exclude_V137_DEV':True,
      'not_sealed_new_composition_test':True,
      'control_limitation':'ABSENT uses third-worker substitution, not pure role removal',
      'budget':'60 MAS executions maximum; MAS executions can each trigger multiple paid calls',
      'requires_explicit_approval':'V515_APPROVE_MAS_EXECUTIONS=60'})
    print('V515_PREPARED',run,'60 planned MAS calls, 2 new backgrounds per role, 5 fixed matched probes, NO API CALLS',flush=True)
    for row in pair_audit: print('V515_BACKGROUND',json.dumps(row),flush=True)
    return run

def execute(run):
    run=safe(run)
    if os.getenv('V515_APPROVE_MAS_EXECUTIONS')!='60': raise RuntimeError('Explicit 60-execution approval required')
    if not os.getenv('OPENAI_API_KEY'):raise RuntimeError('Missing OPENAI_API_KEY')
    pf=j(run/'preflight.json')
    if pf['calls_cap']!=NCALLS or len(pf['background_pairs'])!=6: raise RuntimeError('Wrong preflight budget')
    cands=l(run/'candidates_train_round_01.jsonl')
    if len(cands)!=NCALLS or len({(c['session_id'],c['candidate_index']) for c in cands})!=NCALLS:
        raise RuntimeError('Wrong/duplicate candidates')
    if (run/'responses_train_round_01.jsonl').exists():raise RuntimeError('Already executed; refusing duplicate charged calls')
    cmd=[sys.executable,'-u','scripts/adaptive_v11_execute_round.py','--round','1','--split','train',
         '--root',str(run),'--manifest',str(run/'manifest_privileged.json'),'--workers','2','--retries','0']
    print('V515_EXEC_START scheduled=60, potential multiple paid API calls per MAS execution',flush=True)
    subprocess.run(cmd,cwd=ROOT,check=True)
    answers=l(run/'responses_train_round_01.jsonl')
    if len(answers)!=NCALLS or any(x.get('execution_status')!='ok' or not str(x.get('final_output','')).strip() for x in answers):
        raise RuntimeError('Incomplete responses. Preserve target_public_cache.jsonl; investigate before any retry.')
    print('V515_EXEC_SUCCESS',len(answers),'MAS responses',flush=True)

def auc(pos,neg):return old.auc(pos,neg)

def frozen_scores(bykey):
    try:
        import joblib
        from analysis.role_direct_within_v4 import v5111_rolewise_verifier as verifier
    except ImportError as exc:return {},str(exc)
    model_base=ROOT/'artifacts/role_direct_within_v4/sensor_v5113'
    dirs=sorted(model_base.glob('rolewise_*'),reverse=True)
    source=next((p for p in dirs if all((p/f'model_{a}.joblib').is_file() for a in ('answer_only','length_only'))),None)
    if source is None:return {},'No frozen V5113 model files found'
    out={}
    for arm in ('answer_only','length_only'):
        record=joblib.load(source/f'model_{arm}.joblib')
        if record.get('arm')!=arm or not set(ROLES).issubset(record['models']):raise RuntimeError('Bad frozen model')
        for role in ROLES:
            obs=[];keylist=[]
            for bg in (1,2):
                for slot in range(N_PROBES):
                    for cond in ('TARGET_PRESENT','TARGET_ABSENT'):
                        k=(role,bg,slot,cond);a=bykey[k]
                        obs.append({'role':role,'query':a['query'],'answer':a['final_output']});keylist.append((arm,*k))
            scores=verifier.predict(record['models'][role],obs,arm)
            for k,score in zip(keylist,scores):out[k]=float(score)
    return out,str(source)

def report(run):
    run=safe(run)
    if (run/'report.json').exists():raise RuntimeError('Refusing overwrite existing report')
    candidates=l(run/'candidates_train_round_01.jsonl')
    answers=l(run/'responses_train_round_01.jsonl')
    if len(candidates)!=NCALLS or len(answers)!=NCALLS:raise RuntimeError('Need full 60 responses')
    expected={(c['session_id'],c['candidate_index']):c for c in candidates}
    if len(expected)!=NCALLS:raise RuntimeError('Duplicate candidate identity')
    bykey={}
    for answer in answers:
        pair=(answer['session_id'],answer['candidate_index'])
        if pair not in expected:raise RuntimeError('Unexpected response')
        cand=expected[pair]
        if any(answer.get(k)!=cand.get(k) for k in ('query','target_role','background','condition')):
            raise RuntimeError(f'Response mismatch {pair}')
        if answer.get('execution_status')!='ok' or not answer.get('final_output'):raise RuntimeError('Non-success response')
        k=(cand['target_role'],cand['background'],cand['candidate_index'],cand['condition'])
        if k in bykey:raise RuntimeError('Duplicate response')
        bykey[k]=answer
    if len(bykey)!=NCALLS:raise RuntimeError('Missing response')
    scores,model_path=frozen_scores(bykey)
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    results=[]
    for role in ROLES:
        for bg in (1,2):
            for slot,(family,query) in enumerate(PROBES[role]):
                pos,neg=(bykey[(role,bg,slot,cond)] for cond in ('TARGET_PRESENT','TARGET_ABSENT'))
                if pos['query']!=neg['query'] or pos['query']!=query:raise RuntimeError('Unequal matched probes')
                ap,an=pos['final_output'],neg['final_output']
                vectors=TfidfVectorizer(analyzer='char',ngram_range=(3,5),min_df=1).fit_transform([ap,an])
                row={'role':role,'background':bg,'slot':slot,'family':family,'query':query,
                     'present_chars':len(ap),'absent_chars':len(an),'delta_chars':len(ap)-len(an),
                     'cosine_similarity_char_3_5':round(float(cosine_similarity(vectors[0],vectors[1])[0,0]),5),
                     'oracle_activated_present_eval_only':int(role in pos.get('selected_roles_training_only',[])) if 'selected_roles_training_only' in pos else '',
                     'present_excerpt':ap[:220].replace('\n',' '),'absent_excerpt':an[:220].replace('\n',' ')}
                if scores:
                    for arm in ('answer_only','length_only'):
                        pscore=scores[(arm,role,bg,slot,'TARGET_PRESENT')]
                        nscore=scores[(arm,role,bg,slot,'TARGET_ABSENT')]
                        row[arm+'_present_score']=round(pscore,8)
                        row[arm+'_absent_score']=round(nscore,8)
                        row[arm+'_delta']=round(pscore-nscore,8)
                        row[arm+'_win']=float(pscore>nscore)+.5*float(pscore==nscore)
                results.append(row)
    bygroup=collections.defaultdict(list)
    for r in results:bygroup[(r['role'],r['background'])].append(r)
    bgrows=[]
    for role in ROLES:
        for bg in (1,2):
            rr=bygroup[role,bg]
            row={'role':role,'background':bg,'n_pairs':len(rr),
                 'mean_text_similarity':round(statistics.mean(x['cosine_similarity_char_3_5'] for x in rr),5),
                 'mean_abs_length_delta':round(statistics.mean(abs(x['delta_chars']) for x in rr),2)}
            if scores:
                for arm in ('answer_only','length_only'):
                    row[arm+'_paired_win_rate']=round(statistics.mean(x[arm+'_win'] for x in rr),5)
                    row[arm+'_within_background_auc']=round(auc([x[arm+'_present_score'] for x in rr],[x[arm+'_absent_score'] for x in rr]),5)
            bgrows.append(row)
    role_summary=[]
    for role in ROLES:
        rr=[r for r in results if r['role']==role]
        row={'role':role,'n_backgrounds':2,'n_pairs':10,
             'mean_text_similarity':round(statistics.mean(x['cosine_similarity_char_3_5'] for x in rr),5)}
        if scores:
            for arm in ('answer_only','length_only'):
                row[arm+'_paired_win_rate']=round(statistics.mean(x[arm+'_win'] for x in rr),5)
                row[arm+'_pooled_auc']=round(auc([x[arm+'_present_score'] for x in rr],[x[arm+'_absent_score'] for x in rr]),5)
                row[arm+'_both_backgrounds_positive']=bool(all(statistics.mean(x[arm+'_win'] for x in bygroup[role,bg])>.5 for bg in (1,2)))
        role_summary.append(row)
    probes=[]
    for role in ROLES:
        for slot,(fam,q) in enumerate(PROBES[role]):
            x=[r for r in results if r['role']==role and r['slot']==slot]
            item={'role':role,'slot':slot,'family':fam,'query':q,'two_backgrounds':len(x)}
            if scores:
                item.update({'answer_only_paired_wins':sum(r['answer_only_win'] for r in x),
                             'length_only_paired_wins':sum(r['length_only_win'] for r in x)})
            probes.append(item)
    pair_cols=['role','background','slot','family','query','present_chars','absent_chars','delta_chars',
      'cosine_similarity_char_3_5','oracle_activated_present_eval_only','answer_only_present_score',
      'answer_only_absent_score','answer_only_delta','answer_only_win','length_only_delta','length_only_win',
      'present_excerpt','absent_excerpt']
    bg_cols=['role','background','n_pairs','mean_text_similarity','mean_abs_length_delta',
      'answer_only_paired_win_rate','answer_only_within_background_auc',
      'length_only_paired_win_rate','length_only_within_background_auc']
    role_cols=['role','n_backgrounds','n_pairs','mean_text_similarity','answer_only_paired_win_rate',
      'answer_only_pooled_auc','answer_only_both_backgrounds_positive','length_only_paired_win_rate',
      'length_only_pooled_auc','length_only_both_backgrounds_positive']
    tsv(run/'matched_pairs.tsv',results,pair_cols)
    tsv(run/'by_background.tsv',bgrows,bg_cols)
    tsv(run/'comparison.tsv',role_summary,role_cols)
    tsv(run/'probe_consistency.tsv',probes,list(probes[0]))
    previous=V514_BASE/'latest_run_path.txt'
    prior_csv=[]
    if previous.is_file():
        file=Path(previous.read_text().strip())/'comparison.tsv'
        if file.is_file():
            with file.open(newline='',encoding='utf-8') as f:prior_csv=list(csv.DictReader(f,delimiter='\t'))
    report={'status':'DIAGNOSTIC_NOT_SEALED_HOLDOUT','n_mas_executions':NCALLS,
        'n_roles':3,'backgrounds_per_role':2,'matched_probes_per_background':5,
        'new_questions_per_role':3,'replicated_v514_questions_per_role':2,
        'frozen_verifier_source':model_path,'new_model_fitted_on_v515':False,
        'by_role':role_summary,'by_background':bgrows,'prior_v514_results_for_context_only':prior_csv,
        'limitations':['ABSENT replaces target with another worker: substitution confound remains.',
          'Two backgrounds per role, with only 5 correlated probes each, cannot establish statistical generalization.',
          'Previously inspected data and previously trained model; composition-disjoint from V137 DEV in CF filter does not guarantee complete historical composition isolation.',
          'Same probes reused across backgrounds; do not treat 10 paired observations as 10 independent MAS configurations.',
          'Positive activation traces used in retrospective auditing ONLY, not role-verifier inputs.',
          'No 20-role Within-MAS or exact Top-3 evaluation in this paired experiment.',
          'Probe set is manually authored; results do not demonstrate Generator learning.']}
    wj(run/'report.json',report)
    lines=['# V5.15 — Paired Role Fingerprinting, two new backgrounds','',
       'Scores from frozen V5.11.3, without fitting. **Not a sealed independent generalization benchmark**.','',
       '| Role | Background | Answer paired win | Answer AUC | Length paired win | Length AUC |',
       '|---|---:|---:|---:|---:|---:|']
    for r in bgrows:
        lines.append(f"| {r['role']} | {r['background']} | {r.get('answer_only_paired_win_rate','N/A')} | {r.get('answer_only_within_background_auc','N/A')} | {r.get('length_only_paired_win_rate','N/A')} | {r.get('length_only_within_background_auc','N/A')} |")
    lines+=['','Each background compares exactly identical queries between PRESENT and ABSENT.',
       'The target is **replaced** by another worker in ABSENT, not removed.',
       'Check `matched_pairs.tsv` for all 30 query-pair score deltas.']
    (run/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    for x in bgrows:print('V515_BG_RESULT',json.dumps(x),flush=True)
    for x in role_summary:print('V515_ROLE_RESULT',json.dumps(x),flush=True)
    print('V515_COMPLETE',run,flush=True)
    return report

def selftest():
    if NCALLS!=60:raise AssertionError('Bad call count')
    if set(PROBES)!=set(ROLES):raise AssertionError('Bad roles')
    assert all(len(v)==5 for v in PROBES.values())
    assert all(sum(k.startswith('replication_') for k,_ in v)==2 for v in PROBES.values())
    assert auc([.9,.3],[.2,.8])==.75
    # Need two distinct backgrounds excluding V514 source and no duplicate common background.
    p=lambda a,b:{'roles':[a,b,'code'],'mode':'hierarchical','style':'default'}
    n=lambda a,b:{'roles':[a,b,'documentation'],'mode':'hierarchical','style':'default'}
    pairs={'code':[('old',p('data','operations'),n('data','operations')),
                   ('a',p('writing','design'),n('writing','design')),
                   ('dup',p('writing','design'),n('writing','design')),
                   ('b',p('math','research'),n('math','research'))]}
    # Explicit per-role backgrounds fixture.
    allpairs={r: [(pid,{**pos,'roles':pos['roles'][:2]+[r]},neg) for pid,pos,neg in pairs['code']] for r in ROLES}
    picked=pick_backgrounds(allpairs,{r:'old' for r in ROLES})
    assert all({x[0] for x in picked[r]}=={'a','b'} for r in ROLES)
    print('V515_SELFTEST_OK',flush=True)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('phase',choices=['selftest','prepare','execute','report'])
    ap.add_argument('--root',default=None)
    ap.add_argument('--cf',default=None)
    ap.add_argument('--mas5',default=None)
    args=ap.parse_args()
    if args.phase=='selftest': selftest();return
    if not args.root:raise RuntimeError('--root required')
    if args.phase=='prepare':prepare(args.root,args.cf,args.mas5)
    elif args.phase=='execute':execute(args.root)
    elif args.phase=='report':report(args.root)
if __name__=='__main__':main()
