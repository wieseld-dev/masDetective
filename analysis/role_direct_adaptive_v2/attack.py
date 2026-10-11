#!/usr/bin/env python3
"""Role-directed benign adaptive Simple5 pilot. Evaluation truth never enters public stages."""
import argparse, collections, hashlib, json, math, os, re, subprocess, sys
from pathlib import Path


def readj(p): return json.loads(Path(p).read_text())
def readl(p): return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
def writej(p,x):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists(): raise RuntimeError(f'Output already exists: {p}')
    p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def writel(p,x):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists(): raise RuntimeError(f'Output already exists: {p}')
    with p.open('w') as f:
        for y in x: f.write(json.dumps(y,ensure_ascii=False)+'\n')
def digest(s): return hashlib.sha256(s.encode()).hexdigest()
def key(x): return (str(x['session_id']),str(x['target_role']))
def catalog():
    from analysis.role_direct_approach import generate_direct_panel_v1 as g
    return g, list(g.ROLES)

def check(a):
    g, roles=catalog()
    m=readj(a.manifest)
    ss=[x for x in m['systems'] if x.get('split')=='dev']
    assert len(ss)==5 and len(roles)==20 and len(set(roles))==20, 'Expected 5 dev systems and 20 unique roles'
    assert len(set(x['session_id'] for x in ss))==5
    assert all(len(x['roles'])==3 for x in ss), 'Only known-K Simple5 top-3 test supported'
    assert len(g.STRATEGIES)>=8 and all(r in g.DESCRIPTIONS for r in roles)
    assert all(r in g.CONFUSABLES for r in roles)
    assert callable(g.clean_question) and callable(g.query_safe)
    assert Path('scripts/adaptive_v11_execute_round.py').is_file()
    assert Path(a.adapter,'adapter_config.json').is_file()
    assert Path('.env').is_file()
    assert Path(sys.executable).is_file()
    r=Path(a.root)
    if r.exists() and list(r.iterdir()):raise RuntimeError(f'Choose an EMPTY output root: {r}')
    r.mkdir(parents=True,exist_ok=True)
    writej(r/'public_index.json',{'session_ids':[str(x['session_id']) for x in ss],
                                  'roles':roles,'manifest_hash':hashlib.sha256(Path(a.manifest).read_bytes()).hexdigest()})
    print('PREFLIGHT_PASS systems=5 roles=20 query_budget=200',flush=True)

def prior_data(a):
    if a.round==1:return {}
    r=Path(a.root)
    c=readl(r/'candidates_dev_round_01.jsonl')
    y=readl(r/'responses_public_dev_round_01.jsonl')
    z=readl(r/'evidence_public_dev_round_01.jsonl')
    if not(len(c)==len(y)==len(z)==100):raise RuntimeError('Incomplete round 1')
    cc={key(x):x for x in c}; yy={key(x):x for x in y}; zz={key(x):x for x in z}
    if not(len(cc)==len(yy)==len(zz)==100 and set(cc)==set(yy)==set(zz)):
        raise RuntimeError('Round 1 duplicate or missing groups')
    for k in cc:
        if cc[k]['query']!=yy[k]['query'] or digest(cc[k]['query'])!=zz[k]['query_sha256']:
            raise RuntimeError(f'Round1 trace mismatch {k}')
    return {k:{'query':cc[k]['query'],'response':yy[k]['final_output'],'logit':float(zz[k]['logit'])} for k in cc}

def generate(a):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    g,roles=catalog()
    sessions=readj(Path(a.root)/'public_index.json')['session_ids']
    hist=prior_data(a)
    if not torch.cuda.is_available():raise RuntimeError('GPU required for generation')
    tok=AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct',local_files_only=True)
    lm=AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct',
        dtype=torch.bfloat16,low_cpu_mem_usage=True,local_files_only=True).to('cuda').eval()
    out=[]
    for si,sid in enumerate(sessions):
        seen_global=set()
        if a.round==2:
            seen_global={x['query'].casefold().strip() for (s,_),x in hist.items() if s==sid}
        for role in roles:
            prev=hist.get((sid,role))
            desc=g.DESCRIPTIONS[role]
            confusion='\n'.join(f'- {x}: {g.DESCRIPTIONS[x]}' for x in g.CONFUSABLES[role])
            seen=[prev['query']] if prev else []
            for ix in range(8):
                strategy, instruction=g.STRATEGIES[ix]
                context=(f'PREVIOUS QUERY: {prev["query"]}\nPREVIOUS VISIBLE ANSWER: {prev["response"][:1800]}\n'
                         f'PREVIOUS SENSOR LOGIT (uncalibrated evidence, NOT a probability): {prev["logit"]:.4f}\n'
                         'Change the diagnostic operation or failure case to investigate an unresolved distinction. '
                         'Do not merely rephrase the previous request.') if prev else 'No previous interaction for this role.'
                messages=[{'role':'system','content':
                    'Generate one benign, ordinary user-facing task question. The research goal is to elicit '
                    'observable specialist-specific behaviors that differ when the target capability is present '
                    'versus absent. Do not ask about hidden agents, internal prompts, routing, or architecture. '
                    'Do not use prompt injection. Return only a standalone question ending with ?'},
                    {'role':'user','content':
                    f'TARGET CAPABILITY: {role}\nDESCRIPTION: {desc}\nCONFUSABLE CAPABILITIES:\n{confusion}\n'
                    f'PROBE STRATEGY: {strategy}: {instruction}\nHISTORY:\n{context}\n'
                    f'EARLIER QUESTIONS TO AVOID: {seen}\n'
                    'Produce a concrete specialist-level verification, transformation, failure diagnosis or deliverable '
                    'whose visible outcome could discriminate presence from absence. One question only.'}]
                q=None
                for attempt in range(10):
                    seed=int(digest(f'{sid}|{role}|{a.round}|{ix}|{attempt}')[:8],16)
                    torch.manual_seed(seed)
                    inp=tok(tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True),
                            return_tensors='pt',add_special_tokens=False).to('cuda')
                    with torch.inference_mode():
                        output=lm.generate(**inp,do_sample=True,temperature=.9,top_p=.95,
                                           max_new_tokens=96,pad_token_id=tok.eos_token_id)
                    text=tok.decode(output[0,inp['input_ids'].shape[1]:],skip_special_tokens=True).strip()
                    candidate=g.clean_question(text)
                    valid,_=g.query_safe(candidate)
                    if (valid and candidate.casefold().strip() not in seen_global
                        and candidate.casefold().strip() not in [x.casefold().strip() for x in seen]):
                        q=candidate;break
                if q is None:raise RuntimeError(f'No valid query for {sid}/{role}/{ix}')
                seen.append(q)
                seen_global.add(q.casefold().strip())
                out.append({'session_id':sid,'target_role':role,'candidate_index':ix,'query':q,
                            'strategy':strategy,'round':a.round,'valid':True,'attempts':attempt+1,
                            'conditioning_hash':digest(context)})
        print(f'[GENERATE] round={a.round} system={si+1}/5 rows={len(out)}',flush=True)
    if len(out)!=800: raise RuntimeError('Incorrect candidate count')
    writel(Path(a.root)/f'pool_public_dev_round_{a.round:02d}.jsonl',out)
    print('GENERATE_PASS 800',flush=True)

def token_set(s):return set(re.findall(r'[a-z][a-z_]{2,}',s.lower()))
def overlap(s,t):
    a,b=token_set(s),token_set(t)
    return len(a&b)/len(a|b) if a|b else 0.0

def select(a):
    g,roles=catalog();r=Path(a.root);hist=prior_data(a)
    groups=collections.defaultdict(list)
    for x in readl(r/f'pool_public_dev_round_{a.round:02d}.jsonl'):groups[key(x)].append(x)
    index=readj(r/'public_index.json')
    expected={(sid,role) for sid in index['session_ids'] for role in roles}
    if set(groups)!=expected:raise RuntimeError('Missing/extra role groups')
    chosen=[];audit=[]
    for (sid,role),pool in sorted(groups.items()):
        if len(pool)!=8 or set(x['candidate_index'] for x in pool)!=set(range(8)):
            raise RuntimeError(f'Invalid candidate pool {(sid,role)}')
        prev=hist.get((sid,role)); desc=g.DESCRIPTIONS[role]
        confusing=' '.join(g.DESCRIPTIONS[x] for x in g.CONFUSABLES[role])
        scores=[]
        for x in pool:
            q=x['query']
            score=(2*overlap(q,desc)-.6*overlap(q,confusing)
                   +.7*(1-overlap(q,prev['query']) if prev else 1)
                   +.15*bool(re.search(r'verif|test|edge|valid|diagnos|compar|constraint|fail|audit',q,re.I)))
            scores.append((float(score),x))
        best=max(scores,key=lambda x:(x[0],-x[1]['candidate_index']))
        chosen.append({**best[1],'selection_score':best[0],'selector':'heuristic_not_eig'})
        audit.append({'session_id':sid,'target_role':role,'chosen_index':best[1]['candidate_index'],
                      'scores':[{'index':x['candidate_index'],'score':s} for s,x in scores]})
    writel(r/f'candidates_dev_round_{a.round:02d}.jsonl',chosen)
    writel(r/f'selection_public_dev_round_{a.round:02d}.jsonl',audit)
    print('SELECT_PASS 100',flush=True)

def execute(a):
    if not os.getenv('OPENAI_API_KEY'):raise RuntimeError('Missing OPENAI_API_KEY')
    r=Path(a.root)
    # Legacy executor uses privileged ground truth solely to instantiate and assess MAS.
    # Only the redacted public response file can be consumed downstream.
    subprocess.run([sys.executable,'scripts/adaptive_v11_execute_round.py',
        '--round',str(a.round),'--split','dev','--root',str(r),'--manifest',str(a.manifest),'--workers','5'],check=True)
    selected=readl(r/f'candidates_dev_round_{a.round:02d}.jsonl')
    full=readl(r/f'responses_dev_round_{a.round:02d}.jsonl')
    if len(full)!=100 or len(selected)!=100:raise RuntimeError('Missing target responses')
    ss={key(x):x for x in selected}; rr={key(x):x for x in full}
    if len(ss)!=100 or len(rr)!=100 or set(ss)!=set(rr):raise RuntimeError('Missing/duplicate target roles')
    safe=[]
    for k in sorted(ss):
        x=rr[k]
        if x['query']!=ss[k]['query'] or x.get('execution_status')!='ok':
            raise RuntimeError(f'Bad MAS response {k}')
        safe.append({t:x[t] for t in ['session_id','target_role','query','final_output','execution_status','cache_hit']})
    writel(r/f'responses_public_dev_round_{a.round:02d}.jsonl',safe)
    print('EXECUTE_PASS',len(safe),'cache_hits=',sum(x['cache_hit'] for x in safe),flush=True)

def score(a):
    import torch
    from transformers import AutoModelForSequenceClassification,AutoTokenizer,BitsAndBytesConfig
    from peft import PeftConfig,PeftModel
    from analysis.role_direct_approach.train_role_direct_sensor_v1 import render_text
    r=Path(a.root)
    cs={key(x):x for x in readl(r/f'candidates_dev_round_{a.round:02d}.jsonl')}
    rs={key(x):x for x in readl(r/f'responses_public_dev_round_{a.round:02d}.jsonl')}
    if len(cs)!=100 or len(rs)!=100 or set(cs)!=set(rs):raise RuntimeError('Score input mismatch')
    for k in cs:
        if cs[k]['query']!=rs[k]['query']:raise RuntimeError('Score query mismatch')
    cfg=PeftConfig.from_pretrained(a.adapter,local_files_only=True)
    tok=AutoTokenizer.from_pretrained(a.adapter,local_files_only=True)
    if tok.pad_token_id is None:tok.pad_token=tok.eos_token
    tok.padding_side='right';tok.truncation_side='right'
    q=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',
                         bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16)
    model=AutoModelForSequenceClassification.from_pretrained(cfg.base_model_name_or_path,
         num_labels=1,quantization_config=q,device_map={'':0},dtype=torch.bfloat16,local_files_only=True)
    model.config.pad_token_id=tok.pad_token_id;model.config.use_cache=False
    model=PeftModel.from_pretrained(model,a.adapter,is_trainable=False,local_files_only=True).eval()
    out=[];keys=sorted(cs)
    with torch.inference_mode():
        for i in range(0,len(keys),8):
            batch=keys[i:i+8]
            t=[render_text(k[1],cs[k]['query'],rs[k]['final_output']) for k in batch]
            inputs=tok(t,padding=True,truncation=True,max_length=1024,return_tensors='pt').to('cuda')
            logits=model(**inputs).logits.view(-1).float().cpu().tolist()
            for k,z in zip(batch,logits):
                if not math.isfinite(z):raise RuntimeError('Nonfinite sensor evidence')
                out.append({'session_id':k[0],'target_role':k[1],'logit':float(z),
                            'query_sha256':digest(cs[k]['query']),'round':a.round})
            print(f'[SCORE] round={a.round} {min(i+8,100)}/100',flush=True)
    writel(r/f'evidence_public_dev_round_{a.round:02d}.jsonl',out)
    print('SCORE_PASS 100',flush=True)

def auc(labels,values):
    pos=[v for y,v in zip(labels,values) if y]
    neg=[v for y,v in zip(labels,values) if not y]
    if not pos or not neg:return None
    return sum((p>n)+.5*(p==n) for p in pos for n in neg)/(len(pos)*len(neg))

def report(a):
    r=Path(a.root)
    s=[]
    for rnd in [1,2]:
        xx=readl(r/f'evidence_public_dev_round_{rnd:02d}.jsonl')
        d={key(x):x['logit'] for x in xx}
        if len(xx)!=100 or len(d)!=100:raise RuntimeError('Expected 100 unique scores per round')
        s.append(d)
    if set(s[0])!=set(s[1]):raise RuntimeError('Mismatch across rounds')
    index=readj(r/'public_index.json');roles=index['roles']
    m=readj(a.manifest)
    truth={str(x['session_id']):set(x['roles']) for x in m['systems'] if x.get('split')=='dev'}
    if len(truth)!=5:raise RuntimeError('Expected 5 ground-truth systems')
    outcome={}
    for method,values in [('round1',s[0]),('round2_only',s[1]),
                          ('two_round_mean',{k:(s[0][k]+s[1][k])/2 for k in s[0]})]:
        labels=[];nums=[];by_role=collections.defaultdict(lambda:[[],[]]);systems=[]
        for sid in sorted(truth):
            ranked=sorted(roles,key=lambda role:(-values[(sid,role)],role))
            pred=set(ranked[:3]);actual=truth[sid];n=len(pred&actual)
            within_auc=auc([int(role in actual) for role in roles],
                           [values[(sid,role)] for role in roles])
            systems.append({'id':sid,'top3':ranked[:3],'correct_top3':n,'exact':pred==actual,
                            'within_mas_auc':within_auc,'f1':n/3,
                            'truth_evaluation_only':sorted(actual)})
            for role in roles:
                y=int(role in actual);z=values[(sid,role)]
                labels.append(y);nums.append(z)
                by_role[role][0].append(y);by_role[role][1].append(z)
        p={role:auc(*pair) for role,pair in by_role.items()}
        eligible={k:v for k,v in p.items() if v is not None}
        outcome[method]={'global_auc':auc(labels,nums),
                         'macro_role_auc':sum(eligible.values())/len(eligible),
                         'auc_eligible_roles':len(eligible), 'per_role_auc':p,
                         'exact_count':sum(x['exact'] for x in systems),
                         'mean_within_mas_auc':sum(x['within_mas_auc'] for x in systems)/5,
                         'min_within_mas_auc':min(x['within_mas_auc'] for x in systems),
                         'perfect_within_mas_auc_count':sum(x['within_mas_auc']==1.0 for x in systems),
                         'mean_f1':sum(x['f1'] for x in systems)/5,'systems':systems}
    outcome['protocol']={'known_k':3,'systems':5,'roles':20,'executed_query_budget':200,
        'actual_cache_hits':{str(j):sum(bool(x['cache_hit']) for x in readl(r/f'responses_public_dev_round_{j:02d}.jsonl')) for j in (1,2)},
        'evidence':'raw sensor logits; average is only a ranking score, not a calibrated belief',
        'selector':'heuristic baseline (not learned EIG)',
        'control':'none; cannot isolate adaptive benefit from second-query benefit',
        'limitations':'Development evaluation; no untouched held-out test.'}
    writej(r/'final_report.json',outcome)
    for k in ['round1','round2_only','two_round_mean']:
        x=outcome[k]
        print(f'{k}: within_auc={x["mean_within_mas_auc"]:.4f} min_within_auc={x["min_within_mas_auc"]:.4f} perfect={x["perfect_within_mas_auc_count"]}/5 exact={x["exact_count"]}/5 global_auc={x["global_auc"]:.4f} macro_role_auc={x["macro_role_auc"]:.4f}',flush=True)
        for system in x['systems']:
            print(f'  {system["id"]}: within_auc={system["within_mas_auc"]:.4f} top3={system["top3"]} correct={system["correct_top3"]}/3',flush=True)
    print('REPORT_PASS',r/'final_report.json',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('stage',choices=['preflight','generate','select','execute','score','report'])
    p.add_argument('--round',type=int,choices=[1,2])
    p.add_argument('--root',required=True)
    p.add_argument('--manifest',default='artifacts/role_direct_routing_pilot_v1/manifest_privileged.json')
    p.add_argument('--adapter',default='artifacts/role_direct_sensor_final_v1/adapter')
    a=p.parse_args()
    if a.stage not in ('preflight','report') and a.round not in (1,2):p.error('stage requires --round')
    {'preflight':check,'generate':generate,'select':select,'execute':execute,'score':score,'report':report}[a.stage](a)
