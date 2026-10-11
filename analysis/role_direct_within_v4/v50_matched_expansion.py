#!/usr/bin/env python3
"""MAS5 V5.0 -- train-only matched HMHY study. No automatic DPO/test execution.

No private routing or target labels are used by the attacker. Privileged labels are
used solely for retrospective TRAIN scoring. Counterfactual substitution is NOT a
clean role knockout. Pilot data remain diagnostic-only, not training input.
"""
from __future__ import annotations
import argparse, collections, csv, hashlib, itertools, json, math, os, statistics, subprocess, sys
from pathlib import Path

P = Path('/home/wieseld/mas_rl_detective')
OLD = P/'artifacts/role_direct_within_v4/overnight_v46/mas5_night_20261009_224735_1879051'
BASE = OLD/'train/v47_quality/v472_reviewed_second_pass/v476_manual_review'
BANK = BASE/'v480_final42_review/approved_134.jsonl'
DECISIONS = P/'analysis/role_direct_within_v4/v50_review_decisions.tsv'
CF = P/'artifacts/role_direct_matched_cf_v1/manifest_privileged.json'
MAS5 = P/'artifacts/role_direct_routing_pilot_v1/manifest_privileged.json'
PILOT_PARENT = P/'artifacts/role_direct_within_v4/pilot_v49'
PILOT_ROLES = {'research','database_design','customer_support_triage'}
ROLES = ('code','writing','math','planning','research','data','design','operations','qa_testing','translation',
         'financial_analysis','database_design','accessibility_review','fact_checking','requirements_analysis',
         'incident_management','experiment_design','risk_assessment','documentation','customer_support_triage')
TRAIN_ROLES = tuple(x for x in ROLES if x not in PILOT_ROLES)
MAX_CALLS = 17*3*4*2 # 408; leave all three V4.9 roles outside training.

def jl(p):
    with Path(p).open(encoding='utf-8') as f:return [json.loads(line) for line in f if line.strip()]
def j(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def put(p,data):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as f:f.write(json.dumps(data,indent=2,ensure_ascii=False)+'\n')
def putlines(p,rows):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as f:
        for row in rows:f.write(json.dumps(row,ensure_ascii=False)+'\n')
def key(r):return (str(r['target_role']),int(r['strategy_index']))
def cf_pairs():
    pairs=collections.defaultdict(dict)
    for s in j(CF)['systems']:
        k=(str(s['target_role']),str(s['pair_id']))
        if s['condition'] in pairs[k]:raise RuntimeError(f'Duplicate CF {k}')
        pairs[k][s['condition']]=s
    mas5_sets={frozenset(s['roles']) for s in j(MAS5)['systems'] if s['split']=='dev'}
    result=collections.defaultdict(list)
    for (role,pid),ss in sorted(pairs.items()):
        if set(ss)!={'TARGET_PRESENT','TARGET_ABSENT'}:raise RuntimeError(f'Unpaired {role} {pid}')
        pos,neg=ss['TARGET_PRESENT'],ss['TARGET_ABSENT']
        if not (pos['roles'][:2]==neg['roles'][:2] and pos['roles'][-1]==role and role not in neg['roles']):
            raise RuntimeError(f'Invalid CF background {role} {pid}')
        if (pos['mode'],pos['style'])!=(neg['mode'],neg['style']):raise RuntimeError(f'Unmatched mode/style {pid}')
        if frozenset(pos['roles']) in mas5_sets or frozenset(neg['roles']) in mas5_sets:continue
        result[role].append((pid,pos,neg))
    return result

def choose_four(bank,role):
    hm=sorted([x for x in bank if x['target_role']==role and x.get('probe_family')=='help_me_help_you'],
              key=lambda x:int(x['strategy_index']))
    direct=sorted([x for x in bank if x['target_role']==role and x.get('probe_family')=='direct_task'],
                  key=lambda x:int(x['strategy_index']))
    if len(hm)<4:raise RuntimeError(f'Fewer than four reviewed HMHY for {role}: {len(hm)}')
    # Include a reviewed Direct control where available; otherwise four HMHY.
    # In either case maximize HMHY pairs under IDENTICAL family conditioning for future DPO.
    hnum=3 if direct else 4
    options=[]
    for chosen in itertools.combinations(hm,hnum):
        f=collections.Counter(int(x['strategy_index'])//2 for x in chosen)
        complete=sum(v==2 for v in f.values())
        distinct=len(f)
        options.append(((complete,distinct,-sum(int(x['strategy_index']) for x in chosen)),chosen))
    hchosen=list(max(options,key=lambda x:x[0])[1])
    if not any(sum(int(x['strategy_index'])//2==fam for x in hchosen)==2 for fam in range(3)):
        raise RuntimeError(f'Cannot assemble same-family HMHY pair: {role}')
    return sorted(hchosen + (direct[:1] if direct else []),key=lambda x:int(x['strategy_index']))

def reviewed_bank_source(root):
    """Use reviewed_134; if not materialized, reconstruct ONLY from exact manual decisions.

    The reconstruction is kept in the new isolated run. No original file is changed.
    """
    if BANK.is_file():
        return jl(BANK), BANK
    a=BASE/'approved_118.jsonl'
    b=BASE/'v479_manual_completion/MAS5_candidate_bank_160.jsonl'
    for x in (a,b,DECISIONS):
        if not x.is_file():raise RuntimeError(f'Missing reviewed-134 source {x}; install V5 package or V4.8 review')
    original={key(x):x for x in jl(a)}
    full={key(x):x for x in jl(b)}
    if len(original)!=118 or len(full)!=160 or not all(full.get(k)==v for k,v in original.items()):
        raise RuntimeError('Previously approved question was altered')
    with DECISIONS.open(newline='',encoding='utf-8') as f:ds=list(csv.DictReader(f,delimiter='\t'))
    if len(ds)!=42:raise RuntimeError('Need 42 manual decisions')
    chosen=dict(original);seen=set()
    for d in ds:
        k=(d['role'],int(d['slot']))
        if k in original or k not in full or k in seen:raise RuntimeError(f'Bad decision key {k}')
        seen.add(k)
        if d['decision'] not in ('KEEP','REJECT'):raise RuntimeError('Unrecognized manual decision')
        if full[k]['query']!=d['query'] or full[k].get('strategy','')!=d['strategy']:
            raise RuntimeError(f'Manual decision query mismatch {k}')
        if d['decision']=='KEEP':chosen[k]=full[k]
    if len(seen)!=42 or len(chosen)!=134:raise RuntimeError('Review decisions did not generate 134')
    target=root/'reviewed_134_reconstructed.jsonl'
    putlines(target,[chosen[k] for k in sorted(chosen)])
    print('V50_SAFE_RECONSTRUCT_REVIEWED_134',target,flush=True)
    return list(chosen.values()),target

def prep(root):
    root=Path(root).resolve()
    if not root.is_relative_to(P/'artifacts/role_direct_within_v4'):
        raise RuntimeError('Output root must be isolated within artifacts/role_direct_within_v4')
    if root.exists():raise RuntimeError(f'Output already exists: {root}')
    for p in (CF,MAS5):
        if not p.is_file():raise RuntimeError(f'Missing source {p}')
    # isolated RUN can be created before validating sources; if error it contains no executable manifest
    root.mkdir(parents=True)
    bank,bank_source=reviewed_bank_source(root)
    if len(bank)!=134 or len({key(x) for x in bank})!=134:raise RuntimeError('Reviewed 134 bank is not valid')
    if set(x['target_role'] for x in bank)!=set(ROLES):raise RuntimeError('Missing target roles')
    from adaptive_v11_active.io import sha256_json
    pairpool=cf_pairs();systems=[];cand=[];selected=[];skipped_pilot=[]
    for role in TRAIN_ROLES:
        questions=choose_four(bank,role)
        backgrounds=pairpool[role]
        if len(backgrounds)<3:raise RuntimeError(f'Need >=3 non-MAS5 CF pairs for {role}, got {len(backgrounds)}')
        use=backgrounds[:3]
        for q in questions:
            selected.append({'role':role,'slot':int(q['strategy_index']),'strategy':q.get('strategy'),
                'family':int(q['strategy_index'])//2,'query':q['query']})
        for ix,(original_pair,pos,neg) in enumerate(use,1):
            pair_id=f'v50_{role}_p{ix:02d}'
            for condition,origin in [('TARGET_PRESENT',pos),('TARGET_ABSENT',neg)]:
                sid=f'v50_{role}_p{ix:02d}_{"POS" if condition=="TARGET_PRESENT" else "NEG"}'
                systems.append({**origin,'session_id':sid,'pair_id':pair_id,'split':'train'})
                for q in questions:
                    cand.append({'session_id':sid,'target_role':role,'round':1,'split':'train',
                       'candidate_index':int(q['strategy_index']),'strategy':q.get('strategy',''),
                       'probe_family':q['probe_family'],'query':q['query'],'valid':True,
                       'pair_id':pair_id,'condition':condition,'original_cf_pair':original_pair})
    if len(cand)!=MAX_CALLS or len(systems)!=102 or len(selected)!=68:
        raise RuntimeError(f'Unexpected prepared cardinality: {len(cand)}, {len(systems)}, {len(selected)}')
    ids=[(r['session_id'],r['candidate_index']) for r in cand]
    if len(set(ids))!=len(ids):raise RuntimeError('Duplicate session/candidate')
    put(root/'manifest_privileged.json',{'version':'V50_TRAIN_CF_ONLY','systems':systems,'sha256':sha256_json(systems)})
    putlines(root/'candidates_train_round_01.jsonl',cand)
    putlines(root/'selected_probes.jsonl',selected)
    put(root/'preflight.json',{'status':'PREPARED_WAITING_EXPLICIT_COST_APPROVAL',
         'source_bank':str(bank_source),'source_sha256':digest(bank_source),'candidate_count':len(cand),
         'mas_executions_max':MAX_CALLS,'target_roles':17,'held_out_pilot_roles':sorted(PILOT_ROLES),'probe_count':68,'control_policy':'3 HMHY + 1 reviewed Direct when available, otherwise 4 HMHY',
         'pairs_per_role':3,'excluded_pilot_roles':sorted(PILOT_ROLES),
         'attack_time_uses_privileged_info':False,'dev_mas5_compositions_used':False,
         'limitation':'Negative condition uses substitute worker, not pure target knockout; no trustworthy reward yet.',
         'cost':'Each MAS execution can involve multiple billable API requests; unknown USD cost.',
         'warning':'No automatic DPO training or MAS5 run; validate observational signal first.'})
    print('V50_PREPARED TRAIN_ONLY',len(cand),'API-MAS executions MAX',MAX_CALLS,flush=True)
    print('TRAIN roles',len(TRAIN_ROLES),'selected probes',len(selected),'HMHY',sum(x['slot']<6 for x in selected),'Direct',sum(x['slot']>=6 for x in selected),'heldout pilot roles',sorted(PILOT_ROLES),flush=True)
    print('RUN',root,flush=True)
    return root

def execute(root):
    root=Path(root).resolve();pre=j(root/'preflight.json')
    if pre['mas_executions_max']!=MAX_CALLS:raise RuntimeError('Cap mismatch')
    if digest(pre['source_bank'])!=pre['source_sha256']:raise RuntimeError('Approved bank changed since preparation')
    if (root/'responses_train_round_01.jsonl').exists():
        raise RuntimeError('Responses already exist: refusing duplicate paid execution')
    if os.environ.get('V50_APPROVE_MAS_EXECUTIONS')!=str(MAX_CALLS):
        raise RuntimeError(f'Explicit approval needed: V50_APPROVE_MAS_EXECUTIONS={MAX_CALLS}')
    if not os.environ.get('OPENAI_API_KEY'):raise RuntimeError('OPENAI_API_KEY missing')
    args=[sys.executable,'-u','scripts/adaptive_v11_execute_round.py','--round','1','--split','train',
        '--root',str(root),'--manifest',str(root/'manifest_privileged.json'),'--workers','2','--retries','1']
    print('V50_EXECUTE_START',MAX_CALLS,'max MAS runs; separate API provider calls may be more',flush=True)
    subprocess.run(args,cwd=P,check=True)
    responses=jl(root/'responses_train_round_01.jsonl')
    if len(responses)!=MAX_CALLS or any(x.get('execution_status')!='ok' for x in responses):
        raise RuntimeError(f'Incomplete responses: {len(responses)}/{MAX_CALLS}; inspect cache before retry')
    print('V50_EXECUTE_DONE',len(responses),flush=True)

# The scientific measurement: within-probe, held-background-out response-only TF-IDF signal
# and a length-only adversarial control. No learned model evaluated on MAS5 here.
def score_one(pairs,do_permutations=False):
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    n=len(pairs)
    if n<2:raise RuntimeError('Cannot do held-out CV on fewer than two pairs')
    def experiment(swaps):
        predictions=[];labels=[];len_predictions=[];paired=[];len_paired=[]
        for held in range(n):
            train=[]
            for i,p in enumerate(pairs):
                if i==held:continue
                train.extend([(p['TARGET_ABSENT'] if swaps[i] else p['TARGET_PRESENT'],1),
                              (p['TARGET_PRESENT'] if swaps[i] else p['TARGET_ABSENT'],0)])
            sample=[pairs[held]['TARGET_ABSENT'] if swaps[held] else pairs[held]['TARGET_PRESENT'],
                    pairs[held]['TARGET_PRESENT'] if swaps[held] else pairs[held]['TARGET_ABSENT']]
            tx=[str(v[0].get('final_output','')) for v in train]
            vx=[str(v.get('final_output','')) for v in sample]
            v=TfidfVectorizer(analyzer='char',ngram_range=(3,5),max_features=16000,sublinear_tf=True)
            tr=v.fit_transform(tx);te=v.transform(vx)
            pos=np.asarray(tr[[i for i,(_,y) in enumerate(train) if y==1]].mean(axis=0)).reshape(1,-1)
            neg=np.asarray(tr[[i for i,(_,y) in enumerate(train) if y==0]].mean(axis=0)).reshape(1,-1)
            scores=(cosine_similarity(te,pos).ravel()-cosine_similarity(te,neg).ravel()).tolist()
            av_pos=statistics.mean(len(t) for t,(_,y) in zip(tx,train) if y==1)
            av_neg=statistics.mean(len(t) for t,(_,y) in zip(tx,train) if y==0)
            # Length-only discriminant (train-only learned direction and center)
            center=(av_pos+av_neg)/2
            length_scores=[(len(t)-center)*(av_pos-av_neg) for t in vx]
            predictions+=scores;len_predictions+=length_scores;labels.extend([1,0])
            paired.append(float(scores[0]>scores[1])+.5*float(scores[0]==scores[1]))
            len_paired.append(float(length_scores[0]>length_scores[1])+.5*float(length_scores[0]==length_scores[1]))
        return {'auc':calc_auc(predictions,labels),'pair_win':statistics.mean(paired),
                'length_auc':calc_auc(len_predictions,labels),'length_pair_win':statistics.mean(len_paired)}
    obs=experiment((False,)*n)
    if do_permutations:
        null=[experiment(combo)['pair_win'] for combo in itertools.product([False,True],repeat=n)]
        obs['exact_label_flip_p_at_least_as_large']=sum(v>=obs['pair_win']-1e-12 for v in null)/len(null)
        obs['null_permutations']=len(null)
    return obs

def calc_auc(scores,labels):
    pos=[s for s,y in zip(scores,labels) if y]
    neg=[s for s,y in zip(scores,labels) if not y]
    if not pos or not neg:raise RuntimeError('Missing condition for AUC')
    return sum(1 if a>b else .5 if a==b else 0 for a in pos for b in neg)/(len(pos)*len(neg))

def collect_groups(candidates,responses):
    rs={(str(r['session_id']),int(r['candidate_index'])):r for r in responses}
    if len(rs)!=len(responses) or len(candidates)!=len(responses):raise RuntimeError('Duplicate/missing responses')
    grouped=collections.defaultdict(lambda:collections.defaultdict(dict))
    for c in candidates:
        k=(str(c['session_id']),int(c['candidate_index']))
        if k not in rs:raise RuntimeError(f'Missing response: {k}')
        a=rs[k]
        if str(a['query'])!=str(c['query']) or a.get('condition')!=c['condition']:
            raise RuntimeError('Candidate/response mismatch')
        if a.get('execution_status')!='ok':raise RuntimeError('Failed MAS response')
        itemkey=(c['target_role'],int(c['candidate_index']))
        pk=c['pair_id'];side=c['condition']
        if side in grouped[itemkey][pk]:raise RuntimeError('Duplicate matched response')
        grouped[itemkey][pk][side]=a
    return grouped

def compute_metrics(candidates,responses,permutations=False):
    g=collect_groups(candidates,responses);res=[]
    for (role,slot),pairdict in sorted(g.items()):
        pairs=[]
        for pairid,conditions in sorted(pairdict.items()):
            if set(conditions)!={'TARGET_PRESENT','TARGET_ABSENT'}:raise RuntimeError(f'Missing side {pairid}')
            pairs.append(conditions)
        scores=score_one(pairs,do_permutations=permutations)
        present=[v['TARGET_PRESENT'] for v in pairs]
        activation=[int(role in v.get('selected_roles_training_only',[])) for v in present]
        original=next(c for c in candidates if c['target_role']==role and int(c['candidate_index'])==slot)
        row={'role':role,'slot':slot,'family':slot//2,'strategy':original['strategy'],
             'query':original['query'],'probe_family':original.get('probe_family'),
             'matched_backgrounds':len(pairs),'target_activation_rate':statistics.mean(activation),
             'target_activation_n':sum(activation),**scores}
        res.append(row)
    return res

def audit(root,pilot):
    root=Path(root).resolve();pilot=Path(pilot).resolve()
    c=jl(pilot/'candidates_train_round_01.jsonl');r=jl(pilot/'responses_train_round_01.jsonl')
    if len(c)!=54 or len(r)!=54:raise RuntimeError('Pilot not 54 completed executions')
    metrics=compute_metrics(c,r,permutations=True)
    hm=[x for x in metrics if x['probe_family']=='help_me_help_you'];dr=[x for x in metrics if x['probe_family']=='direct_task']
    report={'status':'DIAGNOSTIC_ONLY','pilot':str(pilot),'probes':len(metrics),'hmhy':len(hm),'direct':len(dr),
       'response_text_auc_mean_hmhy':statistics.mean(x['auc'] for x in hm),
       'length_only_auc_mean_hmhy':statistics.mean(x['length_auc'] for x in hm),
       'hmhy_pair_win_mean':statistics.mean(x['pair_win'] for x in hm),
       'hmhy_length_pair_win_mean':statistics.mean(x['length_pair_win'] for x in hm),
       'mean_exact_permutation_p_hmhy':statistics.mean(x['exact_label_flip_p_at_least_as_large'] for x in hm),
       'summary':'Only three matched pairs/probe; exact per-probe flip test resolution at best 1/8. Not MAS5 role ranking.',
       'conclusion':'Do not train Generator DPO from this pilot alone; inspect length baseline and substitution confound.'}
    root.mkdir(parents=True,exist_ok=False)
    putlines(root/'per_probe_audit.jsonl',metrics)
    put(root/'pilot_robustness_audit.json',report)
    print('V50_PILOT_AUDIT',json.dumps(report,indent=2),flush=True)
    for x in metrics:
        print(f"  {x['role']:<23} slot={x['slot']} auc={x['auc']:.3f} length_auc={x['length_auc']:.3f} pairwin={x['pair_win']:.2f} exact_p={x['exact_label_flip_p_at_least_as_large']:.3f}",flush=True)

def evidence(root):
    root=Path(root).resolve();c=jl(root/'candidates_train_round_01.jsonl');r=jl(root/'responses_train_round_01.jsonl')
    if len(c)!=MAX_CALLS or len(r)!=MAX_CALLS:raise RuntimeError(f'Incomplete evidence: {len(r)}/{MAX_CALLS}')
    ms=compute_metrics(c,r)
    byrole=collections.defaultdict(list)
    for x in ms:byrole[x['role']].append(x)
    pairs=[];preferences=[];auditpairs=[]
    for role,xx in byrole.items():
        byfamily=collections.defaultdict(list)
        for x in xx:byfamily[x['family']].append(x)
        for fam,qs in byfamily.items():
            if len(qs)!=2:continue
            a,b=sorted(qs,key=lambda x:(x['pair_win'],x['auc'],x['target_activation_rate']),reverse=True)
            gap=a['pair_win']-b['pair_win']
            qualifies=(a['matched_backgrounds']>=3 and b['matched_backgrounds']>=3 and
                a['target_activation_rate']>=.75 and a['pair_win']>=.75 and
                gap>=1/3 and a['auc']-a['length_auc']>.05 and
                a['pair_win']-a['length_pair_win']>=1/3)
            record={'role':role,'family':('intake','ambiguity','validation')[fam],
                 'chosen':a['query'],'rejected':b['query'],'chosen_slot':a['slot'],
                 'rejected_slot':b['slot'],'paired_win_gap':gap,
                 'chosen_auc':a['auc'],'chosen_length_auc':a['length_auc'],
                 'chosen_activation':a['target_activation_rate'],'pairs_per_probe':a['matched_backgrounds'],
                 'reliability_gate_pass':qualifies,
                 'limitation':'train-only proxy, differences from substituted workers, not verified causal role knockout'}
            auditpairs.append(record)
            if qualifies:preferences.append(record)
    covered={x['role'] for x in preferences};families={x['family'] for x in preferences}
    summary={'status':'EVIDENCE_READY_FOR_TRAINING_REVIEW' if len(preferences)>=12 and len(covered)>=8 and len(families)>=2 else 'INSUFFICIENT_RELIABLE_PREFERENCES',
         'execution_count':len(r),'matched_observations':len(r)//2,
         'unique_probes':len(ms),'candidate_preferences':len(auditpairs),
         'accepted_preferences':len(preferences),'accepted_roles':len(covered),'accepted_families':len(families),
         'hmhy_mean_auc':statistics.mean(x['auc'] for x in ms if x['probe_family']=='help_me_help_you'),
         'hmhy_mean_length_auc':statistics.mean(x['length_auc'] for x in ms if x['probe_family']=='help_me_help_you'),
         'direct_mean_auc':statistics.mean(x['auc'] for x in ms if x['probe_family']=='direct_task') if any(x['probe_family']=='direct_task' for x in ms) else None,
         'direct_count':sum(x['probe_family']=='direct_task' for x in ms),
         'training_only':True,'dpo_ran':False,'sensor_trained':False,'selector_trained':False,
         'caution':'Passing gates does not establish reliable DPO reward without additional independent system-family controls.'}
    putlines(root/'probe_evidence.jsonl',ms)
    putlines(root/'candidate_preference_audit.jsonl',auditpairs)
    putlines(root/'potential_dpo_preferences.jsonl',preferences)
    put(root/'evidence_report.json',summary)
    print('V50_EVIDENCE_REPORT',json.dumps(summary,indent=2),flush=True)

def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='phase',required=True)
    x=sub.add_parser('audit');x.add_argument('--root',type=Path,required=True);x.add_argument('--pilot',type=Path,required=True)
    x=sub.add_parser('prepare');x.add_argument('--root',type=Path,required=True)
    for name in ('execute','evidence'):
        x=sub.add_parser(name);x.add_argument('--root',type=Path,required=True)
    a=p.parse_args()
    if a.phase=='audit':audit(a.root,a.pilot)
    elif a.phase=='prepare':prep(a.root)
    elif a.phase=='execute':execute(a.root)
    elif a.phase=='evidence':evidence(a.root)
if __name__=='__main__':main()
