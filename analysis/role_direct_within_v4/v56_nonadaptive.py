#!/usr/bin/env python3
"""MAS5 v5.6, two independent role probes from a previously generated vetted bank.

Zero R2 response-conditioning. Generator checkpoint is not sampled online in this
experiment: the 160-question bank was produced previously by a Generator. At each
round the policy selects a DIFFERENT query from that same role's bank, without
reading any answer, truth or trace. Current Sensor is frozen. Only the final
report reads installed role labels; the attacker-facing files contain public data.
"""
from __future__ import annotations
import argparse, collections, csv, hashlib, json, math, os, re, statistics, subprocess, sys
from pathlib import Path

PROJECT=Path(os.environ.get('V56_PROJECT_ROOT','/home/wieseld/mas_rl_detective')).resolve()
SIMPLE=PROJECT/'artifacts/role_direct_routing_pilot_v1/manifest_privileged.json'
SENSOR=PROJECT/'artifacts/role_direct_within_v4/runs/v4_20261009_185359_1475154/checkpoints/sensor_epoch02'
APPROVED_134=PROJECT/'artifacts/role_direct_within_v4/overnight_v46/mas5_night_20261009_224735_1879051/train/v47_quality/v472_reviewed_second_pass/v476_manual_review/v480_final42_review/approved_134.jsonl'
BANK_CANDIDATES=[
 PROJECT/'artifacts/role_direct_within_v4/overnight_v46/mas5_night_20261009_224735_1879051/train/v47_quality/v472_reviewed_second_pass/v476_manual_review/v479_manual_completion/MAS5_candidate_bank_160.jsonl',
]
DENIED=re.compile(r'(?i)\b(?:system prompt|hidden agent|internal routing|routing trace|present[- /]vs[- /]absent|installed roles|target[_ ]role)\b')

def jl(path):
    with Path(path).open(encoding='utf-8') as f:return [json.loads(s) for s in f if s.strip()]
def j(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def safej(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2);f.write('\n')
def safel(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as f:
        for r in x:f.write(json.dumps(r,ensure_ascii=False)+'\n')
def words(q):return re.findall(r"[a-z0-9]+(?:'[a-z]+)?",str(q).lower())
def sim(a,b):
    a,b=set(words(a)),set(words(b));return len(a&b)/max(1,len(a|b))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def choose_bank():
    if os.environ.get('V56_BANK'):
        p=Path(os.environ['V56_BANK']).resolve()
        if not p.is_file():raise RuntimeError(f'V56_BANK missing: {p}')
        return p
    for p in BANK_CANDIDATES:
        if p.is_file():return p
    raise RuntimeError('Approved 160-question generator bank not found. Set V56_BANK to your reviewed JSONL file. No jobs were submitted.')
def manifests(public_manifest):
    from analysis.role_direct_within_v4.pipeline import catalog
    cat=catalog();roles=list(cat['ROLES']);ms=[m for m in j(public_manifest)['systems'] if m.get('split')=='dev']
    if len(roles)!=20 or len(set(roles))!=20:raise RuntimeError('Expected 20 candidate roles')
    if len(ms)!=5 or len({m['session_id'] for m in ms})!=5:raise RuntimeError('Expected exactly five DEV MAS')
    return roles,[str(m['session_id']) for m in ms]
def select_pair(role,records):
    hm=[x for x in records if x.get('probe_family')=='help_me_help_you']
    if len(hm)<2:raise RuntimeError('Need at least two HMHY queries for '+role)
    # Public, static, prespecified diversity/length ranking; never sensor/truth.
    def base(x):
        q=x['query'];n=len(words(q))
        return (abs(n-38)+.3*max(0,n-50),int(x['strategy_index']))
    first=min(hm,key=base)
    alt=[x for x in hm if x is not first and str(x['query']).strip().lower()!=str(first['query']).strip().lower()]
    if not alt:raise RuntimeError('No distinct R2 query '+role)
    # Different HMHY families preferred; avoid nearly identical queries.
    def second_rank(x):
        famsame=(int(x['strategy_index'])//2)==(int(first['strategy_index'])//2)
        overlap=sim(x['query'],first['query'])
        return (overlap>.66,famsame,overlap,-len(set(words(x['query']))-set(words(first['query']))),*base(x))
    second=min(alt,key=second_rank)
    if sim(first['query'],second['query'])>.77:raise RuntimeError('Cannot find sufficiently different probes for '+role)
    return (first,second)
def prepare(root):
    root=Path(root).resolve();allowed=PROJECT/'artifacts/role_direct_within_v4/nonadaptive_v56'
    if not root.is_relative_to(allowed):raise RuntimeError('Output root must stay within nonadaptive_v56')
    if root.exists():raise RuntimeError('Refusing overwrite: '+str(root))
    if not SIMPLE.is_file():raise RuntimeError('Missing established MAS5 manifest')
    if not (SENSOR/'adapter_config.json').is_file():raise RuntimeError('Missing V4 frozen Sensor')
    if not (PROJECT/'scripts/adaptive_v11_execute_round.py').is_file():raise RuntimeError('Missing established MAS executor')
    roles,sids=manifests(SIMPLE);bankfile=choose_bank();bank=jl(bankfile)
    by=collections.defaultdict(list);keys=set()
    for x in bank:
        role=str(x.get('target_role',''));idx=int(x.get('strategy_index',-1));q=str(x.get('query','')).strip()
        if role not in roles or idx not in range(8) or not q or DENIED.search(q):raise RuntimeError('Bad/unsafe bank row')
        key=(role,idx)
        if key in keys:raise RuntimeError('Duplicate bank slot '+str(key))
        keys.add(key);by[role].append(x)
    if len(bank)!=160 or any(len(by[r])!=8 for r in roles):
        raise RuntimeError('Expect exactly 8 reviewed candidate questions for each of 20 roles')
    # Prefer the separately human-reviewed approved_134 subset for EXECUTED queries.
    # All eight bank entries remain visible as proposed candidates; questionable
    # manual completion entries must never be chosen if the reviewed set exists.
    approved_keys=None
    if APPROVED_134.is_file():
        approved_keys={(str(x['target_role']),int(x['strategy_index']),str(x['query']).strip()) for x in jl(APPROVED_134)}
        for role in roles:
            if sum(x.get('probe_family')=='help_me_help_you' and (role,int(x['strategy_index']),str(x['query']).strip()) in approved_keys for x in by[role])<2:
                raise RuntimeError('Fewer than two reviewed HMHY probes in approved_134 for '+role)
    eligible={r:([x for x in by[r] if (r,int(x['strategy_index']),str(x['query']).strip()) in approved_keys] if approved_keys is not None else by[r]) for r in roles}
    selected={r:select_pair(r,eligible[r]) for r in roles}
    # Freeze all selected public probes before writing any evaluation labels or truth.
    root.mkdir(parents=True)
    public={'roles':roles,'session_ids':sids,'source_bank':str(bankfile),'bank_sha256':sha(bankfile),
            'selector':'fixed public lexical diversity (not learned Selector)',
            'executed_probes_from_approved_134':approved_keys is not None,
            'r2_has_no_response_conditioning':True,'rounds':2,'probes_per_role_per_round':1}
    safej(root/'public_preflight.json',public)
    sel=[];pools={1:[],2:[]};cands={1:[],2:[]}
    for role in roles:
        qs=selected[role]
        for round_idx,chosen in enumerate(qs,1):
            sel.append({'target_role':role,'round':round_idx,'original_strategy_index':int(chosen['strategy_index']),
                        'probe_family':chosen['probe_family'],'query':chosen['query'],
                        'word_count':len(words(chosen['query'])),'selection':'frozen_public_diversity'})
            for sid in sids:
                # Keep the full eight-candidate bank for inspection; selection is one/round.
                for p in sorted(by[role],key=lambda z:int(z['strategy_index'])):
                    pools[round_idx].append({'session_id':sid,'target_role':role,
                        'candidate_index':int(p['strategy_index']),'round':round_idx,
                        'probe_family':p['probe_family'],'query':p['query'],'valid':True})
                cands[round_idx].append({'session_id':sid,'target_role':role,'round':round_idx,'split':'dev',
                    'candidate_index':int(chosen['strategy_index']),'query':chosen['query'],'valid':True,
                    'probe_family':chosen['probe_family'],'strategy':chosen.get('strategy',''),
                    'selected_by':'fixed_public_diversity_no_history'})
    for r in (1,2):
        if len(pools[r])!=800 or len(cands[r])!=100:raise RuntimeError('Bad MAS5 cardinality')
        safel(root/f'pool_public_dev_round_{r:02}.jsonl',pools[r])
        safel(root/f'candidates_dev_round_{r:02}.jsonl',cands[r])
    if any(cands[1][i]['query']==cands[2][i]['query'] for i in range(100)):
        raise RuntimeError('R2 must be different from R1 for same session+role')
    safel(root/'review_40_questions.jsonl',sel)
    with (root/'review_40_questions.tsv').open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(sel[0]),delimiter='\t');w.writeheader();w.writerows(sel)
    safej(root/'billable_preflight.json',{'MAS5':5,'candidate_roles':20,'roles_installed_per_MAS':3,
        'rounds':2,'candidate_pool_per_role_per_round':8,'MAS_queries_per_round':100,
        'total_MAS_executions':200,'estimated_provider_calls_if_5_per_MAS':1000,
        'provider_calls_may_exceed_estimate_with_retries':True,'billable_cost_unknown':True,
        'no_adaptivity':True,'question_bank':'previously generated and reviewed (not newly generated tonight)',
        'no_live_generator_inference':True,'no_sensor_or_selector_training':True,
        'frozen_sensor_checkpoint':str(SENSOR),'primary_fusion':'arithmetic mean of two frozen logits',
        'evaluation':'DEV MAS5 revisited; not new sealed generalization','source_bank_sha256':sha(bankfile)})
    print('MAS5_V56_PREPARED 5 MAS 20 ROLES 8 candidates/role/round, 200 selected billable MAS executions',flush=True)
    print('BANK',bankfile,'APPROVED_134_USED',approved_keys is not None,flush=True)
    print('QUESTIONS',root/'review_40_questions.tsv',flush=True)
    for x in sel:print('PREVIEW',x['target_role'],'R'+str(x['round']),x['query'][:175],flush=True)

def execute(root,rd):
    root=Path(root).resolve()
    if os.environ.get('V56_APPROVE_MAS_EXECUTIONS')!='200':raise RuntimeError('Explicit approval V56_APPROVE_MAS_EXECUTIONS=200 required')
    if not os.environ.get('OPENAI_API_KEY'):raise RuntimeError('Missing API key; check .env')
    pf=j(root/'billable_preflight.json');assert pf['total_MAS_executions']==200
    dest=root/f'responses_dev_round_{rd:02}.jsonl'
    if dest.exists():
        # No unintentional duplicate requests, even if partially written.
        raise RuntimeError('Existing responses: refusing paid rerun. Inspect before attempting resumption: '+str(dest))
    cmd=[sys.executable,'-u','scripts/adaptive_v11_execute_round.py','--round',str(rd),
         '--split','dev','--root',str(root),'--manifest',str(SIMPLE),'--workers','3','--retries','1']
    print('MAS5_V56_EXECUTE_START round',rd,'expected 100 MAS queries',flush=True)
    subprocess.run(cmd,cwd=PROJECT,check=True)
    rows=jl(dest); cand=jl(root/f'candidates_dev_round_{rd:02}.jsonl')
    key=lambda x:(str(x['session_id']),str(x['target_role']))
    cc={key(x):x for x in cand};rr={key(x):x for x in rows}
    if len(rows)!=100 or len(rr)!=100 or set(cc)!=set(rr):raise RuntimeError('Incomplete/inconsistent responses')
    safe=[];fields=('session_id','target_role','round','query','candidate_index','final_output','execution_status')
    for k,x in rr.items():
        if x.get('execution_status')!='ok' or x['query']!=cc[k]['query']:
            raise RuntimeError('Mismatch or failed response '+str(k))
        if not x.get('final_output'):raise RuntimeError('Empty final response '+str(k))
        safe.append({fld:x[fld] for fld in fields if fld in x})
    # Do not expose privileged execution fields to scorer.
    safel(root/f'responses_public_dev_round_{rd:02}.jsonl',safe)
    print('MAS5_V56_EXECUTE_COMPLETE',rd,'responses',len(safe),flush=True)

def score(root,maxlen):
    import torch
    from peft import PeftConfig,PeftModel
    from transformers import AutoTokenizer,AutoModelForSequenceClassification
    from analysis.role_direct_within_v4.pipeline import catalog,make_sensor_text
    if not torch.cuda.is_available() or '4090' not in torch.cuda.get_device_name(0):raise RuntimeError('RTX 4090 required')
    if maxlen not in (640,1536):raise RuntimeError('Only preregistered windows 640/1536')
    cfg=PeftConfig.from_pretrained(str(SENSOR),local_files_only=True)
    tok=AutoTokenizer.from_pretrained(str(SENSOR),local_files_only=True)
    if tok.pad_token_id is None:tok.pad_token=tok.eos_token
    tok.truncation_side='right'
    base=AutoModelForSequenceClassification.from_pretrained(cfg.base_model_name_or_path,
           num_labels=1,device_map={'':0},dtype=torch.bfloat16,attn_implementation='eager',local_files_only=True)
    base.config.pad_token_id=tok.pad_token_id
    model=PeftModel.from_pretrained(base,str(SENSOR),is_trainable=False,local_files_only=True).eval()
    desc=catalog()['DESCRIPTIONS']
    allrows=[]
    for rd in (1,2):
        c=jl(root/f'candidates_dev_round_{rd:02}.jsonl')
        r=jl(root/f'responses_public_dev_round_{rd:02}.jsonl')
        key=lambda x:(str(x['session_id']),str(x['target_role']))
        cc={key(x):x for x in c};rr={key(x):x for x in r}
        if len(cc)!=100 or set(cc)!=set(rr):raise RuntimeError('Input mismatch for R'+str(rd))
        for k in sorted(cc):
            x=rr[k];q=cc[k]['query']
            if q!=x['query']:raise RuntimeError('Query mismatch')
            inp=make_sensor_text(k[1],q,x['final_output'],desc)
            full=len(tok(inp,add_special_tokens=True,truncation=False)['input_ids'])
            enc=tok([inp],padding=True,truncation=True,max_length=maxlen,return_tensors='pt').to('cuda')
            with torch.inference_mode(),torch.autocast(device_type='cuda',dtype=torch.bfloat16):
                val=float(model(**enc).logits.reshape(-1)[0].float().item())
            if not math.isfinite(val):raise RuntimeError('Nonfinite Sensor score')
            allrows.append({'session_id':k[0],'target_role':k[1],'round':rd,
                            'logit':round(val,7),'full_tokens':full,'truncated':full>maxlen})
            if len(allrows)%20==0:print('MAS5_V56_SCORE',maxlen,len(allrows),'of 200',flush=True)
    safel(root/f'sensor_public_len{maxlen}.jsonl',allrows)
    print('MAS5_V56_SENSOR_COMPLETE',maxlen,'truncated',sum(x['truncated'] for x in allrows),flush=True)

def report(root):
    from sklearn.metrics import roc_auc_score
    roles=j(root/'public_preflight.json')['roles'];sids=j(root/'public_preflight.json')['session_ids']
    labels={str(x['session_id']):set(x['roles']) for x in j(SIMPLE)['systems'] if x.get('split')=='dev'}
    if len(labels)!=5 or set(labels)!=set(sids):raise RuntimeError('Manifest labels missing')
    out={'primary_metric':'within_mas_auc','precommitted_primary':'mean_of_two_rounds_len640',
         'methods':{},'note':'DEV MAS5 rerun, not unseen sealed. No training or adaptive probing. Mean fusion is not calibrated.'}
    for maxlen in (640,1536):
        d=jl(root/f'sensor_public_len{maxlen}.jsonl')
        by={(x['session_id'],x['target_role'],int(x['round'])):x for x in d}
        if len(d)!=200 or len(by)!=200:raise RuntimeError('Unexpected score rows')
        for method in ('r1','r2','mean','max'):
            systems=[]
            for sid in sids:
                zz=[];yy=[]
                for role in roles:
                    a=by[(sid,role,1)]['logit'];b=by[(sid,role,2)]['logit']
                    v={'r1':a,'r2':b,'mean':(a+b)/2,'max':max(a,b)}[method]
                    zz.append(v);yy.append(int(role in labels[sid]))
                if sum(yy)!=3:raise RuntimeError('Bad installed-role truth cardinality')
                auc=float(roc_auc_score(yy,zz))
                ranked=sorted(zip(roles,zz),key=lambda z:(-z[1],z[0]))
                top3=[x[0] for x in ranked[:3]];correct=len(set(top3)&labels[sid])
                systems.append({'session_id':sid,'within_mas_auc':round(auc,6),
                                'top3':top3,'truth_eval_only':sorted(labels[sid]),
                                'top3_correct':correct,'exact_top3':correct==3,'role_f1_at_k3':correct/3})
            out['methods'][f'len{maxlen}_{method}']={'mean_within_mas_auc':round(statistics.mean(s['within_mas_auc'] for s in systems),6),
                'exact_top3_count':sum(s['exact_top3'] for s in systems),'mean_role_f1_at_k3':round(statistics.mean(s['role_f1_at_k3'] for s in systems),6),
                'systems':systems}
        out[f'len{maxlen}_truncated_count']=sum(bool(x['truncated']) for x in d)
    out['precommitted_primary_result']=out['methods']['len640_mean']
    out['diagnostic_secondary_1536_mean']=out['methods']['len1536_mean']
    out['mas_executions']=200;out['selected_questions_per_role']=2;out['selector_training_performed']=False
    out['generator_weight_training_performed']=False;out['claim_of_new_generalization']=False
    safej(root/'final_MAS5_report.json',out)
    for k,v in out['methods'].items():
        print('MAS5_V56_RESULT',k,'WithinAUC',v['mean_within_mas_auc'],
              'ExactTop3',str(v['exact_top3_count'])+'/5','F1',v['mean_role_f1_at_k3'],flush=True)
    print('MAS5_V56_FINAL',root/'final_MAS5_report.json',flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','execute1','execute2','score640','score1536','report']);p.add_argument('--root',required=True)
    x=p.parse_args();root=Path(x.root).resolve()
    if not root.is_relative_to(PROJECT/'artifacts/role_direct_within_v4/nonadaptive_v56'):
        raise RuntimeError('Output root outside V56 run area')
    {'prepare':lambda:prepare(root),'execute1':lambda:execute(root,1),'execute2':lambda:execute(root,2),
     'score640':lambda:score(root,640),'score1536':lambda:score(root,1536),'report':lambda:report(root)}[x.stage]()
if __name__=='__main__':main()
