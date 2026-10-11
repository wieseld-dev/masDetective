#!/usr/bin/env python3
"""V4.4 role-directed overnight DAG, strict public/privileged boundaries.

This is a *development* MAS5 re-test, NOT sealed unseen evaluation.
Fresh counterfactual TRAIN executions use 8 probes x 20 roles x 3 pairs x 2 conditions.
Training evidence/labels remain in train/; attacker stages see public files only.
"""
from __future__ import annotations
import argparse, collections, hashlib, json, math, os, re, statistics, subprocess, sys
from pathlib import Path

P=Path('/home/wieseld/mas_rl_detective')
OLD=P/'artifacts/role_direct_within_v4/runs/v4_20261009_185359_1475154'
SIMPLE=P/'artifacts/role_direct_routing_pilot_v1/manifest_privileged.json'
CF=P/'artifacts/role_direct_matched_cf_v1/manifest_privileged.json'
V42=P/'artifacts/role_direct_within_v4/runs/v4_20261009_185359_1475154/role_upgrade_v2/run_20261009_210625_1725844'


def load(path):return json.loads(Path(path).read_text())
def jlines(path):return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]
def dump(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():raise RuntimeError('Refusing to overwrite '+str(path))
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
def lines(path,items):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():raise RuntimeError('Refusing to overwrite '+str(path))
    with path.open('w') as f:
        for row in items:f.write(json.dumps(row,ensure_ascii=False)+'\n')
def uniq(rows,k):
    arr=[k(r) for r in rows]
    if len(arr)!=len(set(arr)):raise RuntimeError('Duplicate records found')
def norm(s):return re.sub(r'\W+',' ',s.lower()).strip()
def sha(obj):return hashlib.sha256(json.dumps(obj,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
def cnf(root):return root/'config.json'
def audit(root,msg):
    print('V44',msg,flush=True)

def preflight(root):
    from analysis.role_direct_within_v4 import pipeline as old
    from analysis.role_direct_within_v4 import v42_matched
    if root.exists():raise RuntimeError('Run root already exists: '+str(root))
    cat=old.catalog();roles=cat['ROLES'];assert len(roles)==20
    s=load(SIMPLE);dev=[m for m in s['systems'] if m['split']=='dev']
    cf=load(CF)
    if len(dev)!=5 or any(len(m['roles'])!=3 for m in dev):raise RuntimeError('Unexpected MAS5 manifest')
    if not cf.get('systems') or len(cf['systems'])!=200:raise RuntimeError('Missing matched backgrounds')
    if not (OLD/'checkpoints/generator_activation_sft/adapter_config.json').exists():raise RuntimeError('Generator adapter missing')
    if not (OLD/'checkpoints/sensor_epoch02/adapter_config.json').exists():raise RuntimeError('Sensor parent missing')
    if not (V42/'sensor_candidate.joblib').is_file():raise RuntimeError('V42 teacher scorer missing')
    bank=P/'artifacts/role_direct_within_v4/runs/v4_20261009_185359_1475154/role_upgrade_v3/hmhy_20261009_215152_1793839/v43_hmhy_candidate_probes_round_01.jsonl'
    if not bank.is_file():raise RuntimeError('V4.3 accepted probe bank missing: '+str(bank))
    if len(jlines(bank))!=135:raise RuntimeError('Unexpected V4.3 bank size; inspect before proceeding')
    if not (P/'.env').is_file():raise RuntimeError('Missing API .env. No target calls allowed')
    if not (P/'scripts/adaptive_v11_execute_round.py').is_file():raise RuntimeError('Missing existing target executor')
    devcomps={frozenset(m['roles']) for m in dev}
    cfpairs=collections.defaultdict(dict)
    for m in cf['systems']:cfpairs[str(m['pair_id'])][m['condition']]=m
    byrole=collections.defaultdict(list)
    for pair,conditions in cfpairs.items():
        if set(conditions)!={'TARGET_PRESENT','TARGET_ABSENT'}:raise RuntimeError('Unpaired CF')
        a,b=conditions['TARGET_PRESENT'],conditions['TARGET_ABSENT']
        if set(a['roles'][:2])!=set(b['roles'][:2]) or a['target_role'] in b['roles']:raise RuntimeError('Invalid CF')
        if frozenset(a['roles']) in devcomps or frozenset(b['roles']) in devcomps:continue
        byrole[str(a['target_role'])].append((pair,a,b))
    if any(len(byrole[r])<3 for r in roles):
        raise RuntimeError('Cannot construct three TRAIN CF pairs per role without exact MAS5 overlap')
    # Verify installed requirements before submitting costly real requests.
    import torch, transformers, peft, bitsandbytes, sklearn, joblib
    root.mkdir(parents=True)
    dump(root/'config.json',{'source_v43_probe_bank':str(bank),'source_mas5_manifest':str(SIMPLE),'source_cf_manifest':str(CF),
      'source_generator_adapter':str(OLD/'checkpoints/generator_activation_sft'),
      'source_sensor_adapter':str(OLD/'checkpoints/sensor_epoch02'),
      'source_teacher':str(V42/'sensor_candidate.joblib'),
      'max_train_mas_executions':960,'max_test_mas_executions':200,'total_mas_executions_upper_bound':1160,
      'external_llm_calls_per_mas_execution':'possibly more than one',
      'notes':'MAS5 previously explored; prospective fresh executions not novel sealed compositions',
      'protocol':'3 backgrounds/role x 8 probes x present-absent; two-round 5x20 role attack'})
    audit(root,'PREFLIGHT OK: 960 TRAIN + 200 DEV MAS executions max; costs may be incurred')

def prep_matched(root):
    from adaptive_v11_active.io import sha256_json
    from analysis.role_direct_within_v4 import pipeline as old
    roles=old.catalog()['ROLES'];cat=old.catalog()
    qbank=jlines(root/'train/probes/v44_hmhy_candidate_probes_round_01.jsonl')
    uniq(qbank,lambda x:(x['target_role'],x['strategy_index']))
    for role in roles:
        subset=[x for x in qbank if x['target_role']==role]
        if sum(x.get('probe_family')=='help_me_help_you' for x in subset)!=6:
            raise RuntimeError('MAS5 requires six HMHY probes for role '+role)
    per=collections.defaultdict(list)
    for q in qbank:per[q['target_role']].append(q)
    if set(per)!=set(roles) or any(len(per[r])!=8 for r in roles):
        raise RuntimeError('Expected exactly 8 HMHY/direct candidates for EACH of 20 roles')
    for role,items in per.items():
        if {int(x['strategy_index']) for x in items}!=set(range(8)):
            raise RuntimeError('Missing strategy index for '+role)
        if sum(x['probe_family']=='help_me_help_you' for x in items)!=3:
            raise RuntimeError('Missing Help-Me-Help-You category for '+role)
    dev=[m for m in load(SIMPLE)['systems'] if m['split']=='dev']
    devcomps={frozenset(m['roles']) for m in dev}
    bypair=collections.defaultdict(dict)
    for m in load(CF)['systems']:
        bypair[m['pair_id']][m['condition']]=m
    byrole=collections.defaultdict(list)
    for pair,d in sorted(bypair.items()):
        a,b=d['TARGET_PRESENT'],d['TARGET_ABSENT']
        if frozenset(a['roles']) in devcomps or frozenset(b['roles']) in devcomps:continue
        byrole[a['target_role']].append((pair,a,b))
    systems=[];queries=[];train_comps=[]
    for role in roles:
        if len(byrole[role])<3:raise RuntimeError('Not enough non-overlapping matched CF backgrounds')
        for ix,(pair,a,b) in enumerate(byrole[role][:3],1):
            newpair=f'v44_{role}_{ix:02d}'
            for cond,src in [('TARGET_PRESENT',a),('TARGET_ABSENT',b)]:
                sid=f'v44train_{role}_{ix:02d}_{"POS" if cond=="TARGET_PRESENT" else "NEG"}'
                m={**src,'session_id':sid,'split':'train','pair_id':newpair}
                systems.append(m);train_comps.append(frozenset(src['roles']))
                for q in per[role]:
                    queries.append({'session_id':sid,'target_role':role,'round':1,'split':'train',
                     'candidate_index':q['strategy_index'],'strategy':q['strategy'],'probe_family':q['probe_family'],
                     'query':q['query'],'valid':True,'pair_id':newpair,'condition':cond})
    if len(systems)!=120 or len(queries)!=960:raise RuntimeError('Bad training cardinality')
    assert not set(train_comps)&devcomps
    uniq(queries,lambda x:(x['session_id'],x['candidate_index']))
    manifest={'version':'v44_fresh_matched_training','systems':systems,'sha256':sha256_json(systems)}
    dump(root/'train/manifest_privileged.json',manifest)
    lines(root/'train/candidates_train_round_01.jsonl',queries)
    dump(root/'train/prepared.json',{'systems':120,'pairs':60,'unique_probes':160,'executions':960,
        'training_compositions_exact_mas5_overlap':0,'probe_families':dict(collections.Counter(q['probe_family'] for q in qbank))})
    audit(root,'MATCHED TRAIN PREPARED 960 executions; future evaluation truth is never read by generator/selector/sensor')

def train_execute(root):
    cmd=[sys.executable,'scripts/adaptive_v11_execute_round.py','--round','1','--split','train',
         '--root',str(root/'train'),'--manifest',str(root/'train/manifest_privileged.json'),
         '--workers','4','--retries','2']
    if not os.environ.get('OPENAI_API_KEY'):raise RuntimeError('OPENAI_API_KEY missing')
    if (root/'train/responses_train_round_01.jsonl').exists():raise RuntimeError('Already executed')
    subprocess.run(cmd,cwd=P,check=True)
    rows=jlines(root/'train/responses_train_round_01.jsonl')
    if len(rows)!=960 or any(x.get('execution_status')!='ok' for x in rows):raise RuntimeError('Incomplete matched executions')
    audit(root,f'TRAIN EXECUTION COMPLETE rows={len(rows)} cache_hits={sum(bool(x.get("cache_hit")) for x in rows)}')

def evidence(root):
    from analysis.role_direct_within_v4 import v42_matched
    c=jlines(root/'train/candidates_train_round_01.jsonl')
    r=jlines(root/'train/responses_train_round_01.jsonl')
    paired,n=v42_matched.join(c,r)
    if n!=960 or len(paired)!=480:raise RuntimeError('Bad fresh matched alignment')
    v42_matched.score_model(load(cnf(root))['source_teacher'],paired)
    report,ar,prefs=v42_matched.summarize(paired,str(root/'train'),True)
    # Record fresh role-discrimination evidence by actual identical-query matched contrasts.
    lines(root/'train/matched_evidence.jsonl',ar)
    # Build a finite quality ranking and an SFT pool. Do NOT train from dev or sealed data.
    cat={ (x['role'],norm(x['query'])):x for x in ar }
    qbank=jlines(root/'train/probes/v44_hmhy_candidate_probes_round_01.jsonl')
    chosen=[];scored=[];counters=collections.Counter();strategy_info=collections.defaultdict(list)
    for q in qbank:
        ev=cat[(q['target_role'],norm(q['query']))]
        utility=float(ev['utility'])
        e={**ev,'strategy_index':q['strategy_index'],'strategy':q['strategy'],'probe_family':q['probe_family'],
           'activation_qualifies':ev['target_activation_rate']>=2/3,
           'discrimination_qualifies':ev['paired_win_rate']>=2/3 and ev['mean_signed_score_delta']>0.12}
        scored.append(e);strategy_info[str(q['strategy_index'])].append(utility)
        if e['activation_qualifies'] and e['discrimination_qualifies']:
            chosen.append({'role':q['target_role'],'strategy':q['strategy'],'probe_family':q['probe_family'],
                           'strategy_index':q['strategy_index'],'query':q['query'],'utility':utility,
                           'activation_rate':ev['target_activation_rate'],'pair_win':ev['paired_win_rate']})
    lines(root/'train/generator_positive_sft.jsonl',chosen)
    lines(root/'train/query_evidence_full.jsonl',scored)
    # Strategy-only prior learned on train matched evidence. No privileged runtime access.
    stats={k:statistics.mean(vals) for k,vals in strategy_info.items()}
    dump(root/'train/public_selector_priors.json',stats)
    report.update({'high_evidence_positive_sft_queries':len(chosen),
                   'high_evidence_positive_roles':len(set(x['role'] for x in chosen)),
                   'note':'Teacher is pre-existing V42 within-ranker proxy; NOT ground-truth mutual information'})
    dump(root/'train/evidence_report.json',report)
    if len(chosen)<18 or len({x['role'] for x in chosen})<8:
        raise RuntimeError(f'INSUFFICIENT_RELIABLE_EVIDENCE: {len(chosen)} quality probes, '+
                           f'{len(set(x["role"] for x in chosen))} roles; stop rather than train weak proxy')
    audit(root,'EVIDENCE GATE PASSED '+str(len(chosen))+' high-signal probes')

def selector(root):
    # Use earlier strict 5-fold leave-role-out selector; PASS or explicit fallback.
    from analysis.role_direct_within_v4 import v42_selector
    inp=root/'train/matched_evidence.jsonl'
    out=root/'train/selector'
    v42_selector.train(str(inp),out)
    verdict=load(out/'selector_report.json')['verdict']
    audit(root,'SELECTOR '+verdict)

def train_gate(root):
    s=load(root/'train/sensor_train_report.json')
    g=load(root/'train/generator_train_report.json')
    q=load(root/'train/evidence_report.json')
    if s['status']!='PASS' or g['status']!='PASS':raise RuntimeError('Training failed acceptance gate')
    if not (root/'train/sensor_adapter/adapter_config.json').exists():raise RuntimeError('Missing new sensor')
    if not (root/'train/generator_adapter/adapter_config.json').exists():raise RuntimeError('Missing new generator')
    dump(root/'train/COMPLETED_TRAIN.json',{'generator':g,'sensor':s,'evidence_summary':q,
        'selector':load(root/'train/selector/selector_report.json')['verdict'],
        'acceptance':'Both LoRA models trained and passed matched train/validation checks; NOT a proof of MAS5 gain'})
    audit(root,'TRAIN COMPLETE / TEST RELEASED')

def test_prepare(root):
    if not (root/'train/COMPLETED_TRAIN.json').exists():raise RuntimeError('No completed training; no MAS5')
    m=load(SIMPLE)
    systems=[x for x in m['systems'] if x['split']=='dev']
    roles=__import__('analysis.role_direct_within_v4.pipeline',fromlist=['catalog']).catalog()['ROLES']
    if len(systems)!=5 or len(roles)!=20:raise RuntimeError('Unexpected MAS5')
    dump(root/'test/public_sessions.json',{'session_ids':[m['session_id'] for m in systems],'roles':roles})
    audit(root,'TEST SESSION CREATED without ground-truth in public index')

def test_generate(root,round_num):
    from analysis.role_direct_within_v4 import v44_generator as gen
    public=load(root/'test/public_sessions.json');sids=public['session_ids']
    if round_num==1:
        sub=root/'test/pools/r1';sub.mkdir(parents=True,exist_ok=False)
        gen.generate(sub,1,None,.9)
        bank=jlines(sub/'v44_hmhy_candidate_probes_round_01.jsonl')
        if len(bank)!=160:raise RuntimeError('Incomplete R1 160 probe bank')
        rows=[]
        for sid in sids:
            for q in bank:
                rows.append({'session_id':sid,'target_role':q['target_role'],'candidate_index':q['strategy_index'],
                    'query':q['query'],'strategy':q['strategy'],'probe_family':q['probe_family'],
                    'round':1,'valid':True})
    else:
        from collections import defaultdict
        r1=jlines(root/'test/candidates_dev_round_01.jsonl')
        ans=jlines(root/'test/responses_public_dev_round_01.jsonl')
        scr=jlines(root/'test/sensor_public_dev_round_01.jsonl')
        key=lambda a:(a['session_id'],a['target_role'])
        cm={key(x):x for x in r1};rm={key(x):x for x in ans};sm={key(x):x for x in scr}
        if set(cm)!=set(rm) or set(cm)!=set(sm) or len(cm)!=100:raise RuntimeError('R1 public history incomplete')
        rows=[]
        for sid in sids:
            z=[float(sm[(sid,role)]['logit']) for role in public['roles']]
            # Rank-based pseudo-belief, deliberately not falsely described as calibrated probability.
            ordered=sorted(public['roles'],key=lambda r:(-float(sm[(sid,r)]['logit']),r))
            belief={role:float((20-i-.5)/20) for i,role in enumerate(ordered)}
            hist=[{'target_role':role,'query':cm[(sid,role)]['query'],
                 'final_response':rm[(sid,role)]['final_output'],'round':1} for role in public['roles']]
            statedir=root/'test/pools/r2'/sid;statedir.mkdir(parents=True,exist_ok=False)
            dump(statedir/'public_state.json',{'role_belief':belief,'history':hist,'round':2})
            gen.generate(statedir,2,statedir/'public_state.json',.9)
            bank=jlines(statedir/'v44_hmhy_candidate_probes_round_02.jsonl')
            if len(bank)!=160:raise RuntimeError('R2 must have exactly 8 probes per role per MAS')
            for q in bank:
                rows.append({'session_id':sid,'target_role':q['target_role'],'candidate_index':q['strategy_index'],
                    'query':q['query'],'strategy':q['strategy'],'probe_family':q['probe_family'],
                    'round':2,'valid':True})
            audit(root,f'R2 generated for {sid} ({len(bank)} candidates)')
    if len(rows)!=800:raise RuntimeError('5 x 20 x 8 expected')
    uniq(rows,lambda x:(x['session_id'],x['target_role'],x['candidate_index']))
    lines(root/f'test/pool_public_dev_round_{round_num:02d}.jsonl',rows)
    audit(root,f'TEST GENERATE R{round_num} complete {len(rows)} candidates')

def test_select(root,round_num):
    from analysis.role_direct_within_v4 import v42_selector
    import joblib
    groups=collections.defaultdict(list)
    for x in jlines(root/f'test/pool_public_dev_round_{round_num:02d}.jsonl'):
        groups[(x['session_id'],x['target_role'])].append(x)
    priors=load(root/'train/public_selector_priors.json')
    sel_report=load(root/'train/selector/selector_report.json')
    learned=sel_report['verdict']=='LEARNED_HELDOUT_ROLE_PASS'
    learned_model=joblib.load(root/'train/selector/selector_candidate.joblib') if learned else None
    hist={}
    if round_num==2:
        hist={(x['session_id'],x['target_role']):x for x in jlines(root/'test/candidates_dev_round_01.jsonl')}
    chosen=[];auditrows=[]
    for key,cands in sorted(groups.items()):
        if len(cands)!=8:raise RuntimeError('Bad candidate group '+str(key))
        if learned:
            x=[{'role':z['target_role'],'query':z['query']} for z in cands]
            scores=learned_model['selector'].scores(x,learned_model['descriptions'])
        else:
            scores=[float(priors.get(str(z['candidate_index']),0)) for z in cands]
        prior=hist.get(key)
        adjusted=[]
        for c,s in zip(cands,scores):
            novelty=1.0
            if prior:
                a=set(norm(c['query']).split());b=set(norm(prior['query']).split())
                novelty=1-len(a&b)/max(1,len(a|b))
            adjusted.append(float(s)+(.35*novelty if prior else 0))
        idx=max(range(8),key=lambda i:(adjusted[i],-cands[i]['candidate_index']))
        c=cands[idx]
        chosen.append({**c,'selection_score':adjusted[idx],
                       'selector_version':'learned-role-cv' if learned else 'fixed-training-strategy-evidence'})
        auditrows.append({'session_id':key[0],'target_role':key[1],
                          'selected_candidate_index':c['candidate_index'],
                          'candidate_scores':[float(x) for x in adjusted]})
    if len(chosen)!=100:raise RuntimeError('Expected 100 selected probes')
    lines(root/f'test/candidates_dev_round_{round_num:02d}.jsonl',chosen)
    lines(root/f'test/selection_public_dev_round_{round_num:02d}.jsonl',auditrows)
    audit(root,f'TEST SELECT R{round_num} 100 actual MAS queries')

def test_execute(root,round_num):
    if not os.environ.get('OPENAI_API_KEY'):raise RuntimeError('Missing API key')
    dest=root/f'test/responses_dev_round_{round_num:02d}.jsonl'
    if dest.exists():raise RuntimeError('Already executed')
    cmd=[sys.executable,'scripts/adaptive_v11_execute_round.py','--round',str(round_num),
         '--split','dev','--root',str(root/'test'),'--manifest',str(SIMPLE),
         '--workers','4','--retries','2']
    subprocess.run(cmd,cwd=P,check=True)
    responses=jlines(dest)
    if len(responses)!=100 or any(x.get('execution_status')!='ok' or x.get('cache_hit') for x in responses):
        raise RuntimeError('Not 100 fresh successful responses: STOP the test')
    allow=('session_id','target_role','query','candidate_index','final_output','execution_status','cache_hit')
    safe=[{k:v for k,v in x.items() if k in allow} for x in responses]
    lines(root/f'test/responses_public_dev_round_{round_num:02d}.jsonl',safe)
    audit(root,f'TEST EXECUTE R{round_num}: 100 fresh successful outputs; private routing stripped')

def test_score(root,round_num):
    import torch
    from peft import PeftConfig,PeftModel
    from transformers import AutoTokenizer,AutoModelForSequenceClassification,BitsAndBytesConfig
    from analysis.role_direct_within_v4.pipeline import catalog,make_sensor_text
    if '4090' not in torch.cuda.get_device_name(0):raise RuntimeError('RTX4090 required')
    c=jlines(root/f'test/candidates_dev_round_{round_num:02d}.jsonl')
    r=jlines(root/f'test/responses_public_dev_round_{round_num:02d}.jsonl')
    key=lambda x:(x['session_id'],x['target_role'])
    cc={key(x):x for x in c};rr={key(x):x for x in r}
    if set(cc)!=set(rr) or len(cc)!=100:raise RuntimeError('Incomplete public data')
    ap=root/'train/sensor_adapter';conf=PeftConfig.from_pretrained(ap,local_files_only=True)
    tok=AutoTokenizer.from_pretrained(ap,local_files_only=True)
    if tok.pad_token_id is None:tok.pad_token=tok.eos_token
    quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16)
    base=AutoModelForSequenceClassification.from_pretrained(conf.base_model_name_or_path,num_labels=1,
              quantization_config=quant,device_map={'':0},dtype=torch.bfloat16,local_files_only=True)
    base.config.pad_token_id=tok.pad_token_id
    model=PeftModel.from_pretrained(base,str(ap),local_files_only=True,is_trainable=False).eval()
    ds=catalog()['DESCRIPTIONS'];keys=sorted(cc);out=[]
    with torch.inference_mode():
        for start in range(0,len(keys),2):
            ks=keys[start:start+2];txt=[]
            for k in ks:
                if cc[k]['query']!=rr[k]['query']:raise RuntimeError('Mismatched request/response')
                txt.append(make_sensor_text(k[1],cc[k]['query'],rr[k]['final_output'],ds))
            inp=tok(txt,padding=True,truncation=True,max_length=640,return_tensors='pt').to('cuda')
            with torch.autocast(device_type='cuda',dtype=torch.bfloat16):z=model(**inp).logits.view(-1).float().cpu().tolist()
            for k,score in zip(ks,z):out.append({'session_id':k[0],'target_role':k[1],
                         'logit':float(score),'round':round_num})
            if start%20==0:audit(root,f'TEST SCORE R{round_num} {start+len(ks)}/100')
    lines(root/f'test/sensor_public_dev_round_{round_num:02d}.jsonl',out)
    audit(root,f'TEST SCORE R{round_num} completed')

def final_report(root):
    from sklearn.metrics import roc_auc_score
    c=load(root/'test/public_sessions.json');roles=c['roles']
    ss=[]
    for rnd in (1,2):
        x=jlines(root/f'test/sensor_public_dev_round_{rnd:02d}.jsonl')
        d={(r['session_id'],r['target_role']):float(r['logit']) for r in x}
        if len(d)!=100:raise RuntimeError('Missing test scores R'+str(rnd))
        ss.append(d)
    if set(ss[0])!=set(ss[1]):raise RuntimeError('R1/R2 mismatch')
    # Only the reporting stage reads privileged installed roles.
    ms=[m for m in load(SIMPLE)['systems'] if m['split']=='dev']
    if len(ms)!=5 or any(len(m['roles'])!=3 for m in ms):raise RuntimeError('Bad MAS5')
    results={}
    methods={'round1':ss[0], 'round2':ss[1],
       'two_round_mean':{k:.5*(ss[0][k]+ss[1][k]) for k in ss[0]}}
    for name,v in methods.items():
        systems=[]
        for m in ms:
            sid=m['session_id'];true=set(m['roles'])
            y=[int(r in true) for r in roles];z=[v[(sid,r)] for r in roles]
            ordered=sorted(roles,key=lambda r:(-v[(sid,r)],r))
            pred=set(ordered[:3]);tp=len(pred&true)
            systems.append({'session_id':sid,'auc':float(roc_auc_score(y,z)),
                            'top3':ordered[:3],'correct':tp,'exact':tp==3,
                            'top3_f1':tp/3,'truth_eval_only':sorted(true)})
        results[name]={'mean_within_mas_auc':statistics.mean(m['auc'] for m in systems),
          'exact_top3_count':sum(m['exact'] for m in systems),'mean_top3_f1':statistics.mean(m['top3_f1'] for m in systems),
          'systems':systems}
    results['protocol']={'mas':5,'candidate_roles':20,'known_k':3,'rounds':2,'candidates_per_role_per_round':8,
        'actual_mas_execution_count':200,'cache_hits':0,'generator_adapter':str(root/'train/generator_adapter'),
        'sensor_adapter':str(root/'train/sensor_adapter'),
        'selector_decision':load(root/'train/selector/selector_report.json')['verdict'],
        'training_data_never_test':'Distinct SIDs, different exact triples; CF background designs originated in prior DEV experiments',
        'caveat':'Previously studied MAS5 DEV compositions; NOT untouched sealed test. No claim of new system generalization'}
    dump(root/'final_MAS5_report.json',results)
    for k in ('round1','round2','two_round_mean'):
        o=results[k];print(f'{k} WITHIN_AUC={o["mean_within_mas_auc"]:.4f} EXACT_TOP3={o["exact_top3_count"]}/5 F1={o["mean_top3_f1"]:.4f}',flush=True)
    audit(root,'FINAL 5 MAS REPORT COMPLETE')

STAGES={'preflight':preflight,'prepare_train':prep_matched,'execute_train':train_execute,
        'evidence':evidence,'selector':selector,'gate':train_gate,'test_prepare':test_prepare,
        'test_select1':lambda x:test_select(x,1),'test_select2':lambda x:test_select(x,2),
        'test_execute1':lambda x:test_execute(x,1),'test_execute2':lambda x:test_execute(x,2),
        'test_generate1':lambda x:test_generate(x,1),'test_generate2':lambda x:test_generate(x,2),
        'test_score1':lambda x:test_score(x,1),'test_score2':lambda x:test_score(x,2),
        'report':final_report}
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=STAGES);p.add_argument('--root',required=True)
    args=p.parse_args();root=Path(args.root).resolve()
    if not root.is_relative_to(P/'artifacts/role_direct_within_v4'):
        raise SystemExit('Only isolated V4 output root allowed')
    STAGES[args.stage](root)
