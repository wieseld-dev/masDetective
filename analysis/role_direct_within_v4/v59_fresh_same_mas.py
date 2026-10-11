#!/usr/bin/env python3
"""MAS5 V5.9: fresh-query, fresh-response evaluation on SAME five MAS5 compositions.

One fixed pair of NEW (not V5.7 training) probes per target role. Both the old
and newly trained Sensor see the EXACT SAME 200 fresh public responses; weights
are frozen. Label/truth loading happens only in the REPORT stage.

This is same-composition query holdout, NOT held-out-topology generalization.
"""
from __future__ import annotations
import argparse, collections, csv, hashlib, json, math, os, re, statistics, subprocess, sys
from pathlib import Path

from analysis.role_direct_within_v4 import v56_nonadaptive as v56
from analysis.role_direct_within_v4 import v57_sensor_night as v57

P=v56.PROJECT
AREA=P/'artifacts/role_direct_within_v4/fresh_test_v59'
SENSOR_OLD=v57.DEFAULT_PARENT
LEAK=re.compile(r'(?i)\b(?:system prompt|hidden agent|internal routing|routing trace|installed roles|target[_ ]role|present vs absent|prompt injection)\b')
STOP=set('i me my you your could would should please what which when where why how need needs do does a an the and or is are of to for in on at by about with can from that this it specific information details additional help provide so we our system application project'.split())

def lines(path):
    with Path(path).open(encoding='utf8') as f:return [json.loads(s) for s in f if s.strip()]
def read(path):return json.loads(Path(path).read_text(encoding='utf8'))
def atomic_json(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf8');os.replace(tmp,path)
def write_once(path,rows):
    with Path(path).open('x',encoding='utf8') as f:
        for row in rows:f.write(json.dumps(row,ensure_ascii=False)+'\n')
def norm(q):return ' '.join(re.findall(r"[a-z0-9]+(?:'[a-z]+)?",str(q).lower()))
def toks(q):return set(norm(q).split())-STOP
def jac(a,b):
    x,y=toks(a),toks(b)
    return len(x&y)/max(1,len(x|y))
def hashq(q):return hashlib.sha256(norm(q).encode()).hexdigest()
def check_root(root):
    root=Path(root).absolute()
    if root.parent.resolve()!=AREA.resolve() or not root.name.startswith('fresh_') or root.is_symlink():
        raise RuntimeError('Refusing V59 output outside fresh_test_v59')
    return root

def source_v56():
    from os import environ
    direct=environ.get('V59_V56','')
    if direct:
        out=Path(direct).resolve()
    else:
        found=sorted((P/'artifacts/role_direct_within_v4/nonadaptive_v56').glob('mas5_*'),reverse=True)
        found=[x for x in found if (x/'responses_public_dev_round_02.jsonl').is_file() and (x/'public_preflight.json').is_file()]
        if not found:raise RuntimeError('Cannot locate V56 completed 200-response run; set V59_V56')
        out=found[0]
    if not (out/'responses_public_dev_round_02.jsonl').is_file():raise RuntimeError('V56 lacks Round 2 responses')
    return out

def source_v57():
    direct=os.environ.get('V59_V57','')
    if direct:out=Path(direct).resolve()
    else:
        found=sorted((P/'artifacts/role_direct_within_v4/sensor_v57').glob('night_*'),reverse=True)
        if not found:raise RuntimeError('V57 run not found; set V59_V57')
        out=found[0]
    if not (out/'jobs.tsv').is_file():raise RuntimeError('V57 jobs.tsv missing')
    return out

def training_queries(v56root):
    """Recover ALL sources from the actual V5.7 trainer, not only V5.6.
    The values are used exclusively for novelty filtering; labels stay outside
    Sensor inputs and query selection.
    """
    examples=[];stats=collections.Counter()
    def add(role,q,src):
        if not role or not q:return
        examples.append((str(role),str(q),src));stats[src]+=1
    for rd in (1,2):
        for row in lines(v56root/f'responses_public_dev_round_{rd:02}.jsonl'):
            add(row['target_role'],row['query'],'v56_executed')
    tr,dev=v57.historical_v137()
    for x in tr:add(x['role'],x['query'],'v137_train')
    v44=v57.locate_matched()
    if v44:
        for pos,neg in v57.matched_from_candidates(v44/'candidates_train_round_01.jsonl',v44/'responses_train_round_01.jsonl','v44_matched'):
            for x in (pos,neg):add(x['role'],x['query'],'v44_matched')
    for pos,neg in v57.v54_matched():
        for x in (pos,neg):add(x['role'],x['query'],'v54_matched')
    return examples,dict(stats),str(v44) if v44 else None

def prepare(root,v56root,v57root):
    root=check_root(root)
    if root.exists():raise RuntimeError('New run path already exists; refusing overwrite')
    if not (P/'scripts/adaptive_v11_execute_round.py').is_file():raise RuntimeError('MAS executor missing')
    if not (SENSOR_OLD/'adapter_config.json').is_file():raise RuntimeError('Frozen original Sensor missing')
    roles,sids=v56.manifests(v56.SIMPLE)
    p56=read(v56root/'public_preflight.json')
    if set(sids)!=set(p56['session_ids']):raise RuntimeError('Source system mismatch')
    sourcebank=Path(p56['source_bank']).resolve()
    if not sourcebank.is_file():sourcebank=v56.choose_bank()
    if v56.sha(sourcebank)!=p56['bank_sha256']:raise RuntimeError('Original bank SHA mismatch')
    bank=lines(sourcebank)
    if len(bank)!=160:raise RuntimeError('Expected the 160-item generator bank')
    approved={}
    if v56.APPROVED_134.is_file():
        for x in lines(v56.APPROVED_134):
            approved[(x['target_role'],int(x['strategy_index']),norm(x['query']))]=True
    if not approved:raise RuntimeError('Approved question pool unavailable. No billable jobs submitted.')
    existing,counts,v44root=training_queries(v56root)
    role_train=collections.defaultdict(list);exact=set()
    for role,q,src in existing:
        role_train[role].append((q,src));exact.add(hashq(q))
    selected={};review=[];used_global=set();rejections=collections.Counter()
    for role in roles:
        eligible=[]
        for row in bank:
            if row.get('target_role')!=role:continue
            q=str(row.get('query','')).strip();idx=int(row.get('strategy_index',-1))
            if (role,idx,norm(q)) not in approved:rejections['not_previously_approved']+=1;continue
            if LEAK.search(q) or '?' not in q or not 7<=len(norm(q).split())<=110:
                rejections['bad_or_unsafe']+=1;continue
            if hashq(q) in exact or hashq(q) in used_global:
                rejections['exact_training_overlap']+=1;continue
            localmax=max((jac(q,p) for p,_ in role_train[role]),default=0)
            if localmax>0.80:
                rejections['near_training_overlap']+=1;continue
            # Prefer previously reviewed HMHY; distinct, moderate length.
            h=row.get('probe_family')=='help_me_help_you'
            if not h:rejections['not_hmhy']+=1;continue
            eligible.append({'target_role':role,'query':q,'strategy_index':idx,'probe_family':'help_me_help_you',
                             'max_jaccard_with_training_same_role':round(localmax,5),
                             'word_count':len(norm(q).split())})
        if len(eligible)<2:
            raise RuntimeError(f'Not enough TRAIN-UNSEEN approved HMHY probes for {role}: {len(eligible)}; abort before API. Source count={counts}')
        eligible.sort(key=lambda x:(x['max_jaccard_with_training_same_role'],abs(x['word_count']-38),x['strategy_index']))
        a=eligible[0]
        second=sorted([q for q in eligible[1:] if hashq(q['query'])!=hashq(a['query'])],
          key=lambda x:(jac(x['query'],a['query'])>.66,jac(x['query'],a['query']),x['max_jaccard_with_training_same_role'],abs(x['word_count']-38),x['strategy_index']))
        if not second or jac(a['query'],second[0]['query'])>.77:
            raise RuntimeError('No sufficiently different holdout probes for '+role)
        b=second[0]
        selected[role]=(a,b)
        used_global.add(hashq(a['query']));used_global.add(hashq(b['query']))
        for rd,x in enumerate((a,b),1):
            review.append({**x,'round':rd,'similarity_to_other_round':round(jac(a['query'],b['query']),5)})
    assert len(review)==40 and len(used_global)==40
    root.mkdir(parents=True)
    cand={1:[],2:[]}
    for role in roles:
        for rd,x in enumerate(selected[role],1):
            for sid in sids:
                cand[rd].append({'session_id':sid,'target_role':role,'split':'dev','round':rd,
                    'candidate_index':x['strategy_index'],'query':x['query'],'valid':True,
                    'probe_family':x['probe_family'],'selected_by':'unseen_fixed_bank_holdout'})
    for rd in (1,2):
        if len(cand[rd])!=100:raise RuntimeError('Expected 100 MAS calls/round')
        write_once(root/f'candidates_dev_round_{rd:02}.jsonl',cand[rd])
    write_once(root/'review_40_fresh_questions.jsonl',review)
    with (root/'review_40_fresh_questions.tsv').open('x',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(review[0]),delimiter='\t');w.writeheader();w.writerows(review)
    atomic_json(root/'public_preflight.json',{'roles':roles,'session_ids':sids,'source_bank':str(sourcebank),
        'v56_root':str(v56root),'v57_root':str(v57root),
        'source_bank_sha256':v56.sha(sourcebank),'train_source_counts':counts,'v44_source':v44root,
        'candidate_questions':40,'different_from_v57_training_queries':True,
        'no_role_truth_used_by_selection':True,'max_role_specific_training_jaccard':0.80,
        'all_queries_hmhy_approved':True,'rounds':2,'mas_calls':200,
        'same_five_mas_compositions':True,'not_new_system_generalization':True,
        'model_weight_training':False})
    atomic_json(root/'billable_preflight.json',{'billable_mas_executions':200,'provider_calls_estimated_if_five_per_mas':1000,
        'provider_calls_may_exceed_estimate':True,'cost_unknown':True,'requires_explicit_opt_in':True,
        'same_five_MAS_new_queries_new_responses':True,'v57_adapters_frozen':True,
        'novel_question_train_counts':counts,'reject_counts':dict(rejections)})
    print('MAS5_V59_READY',json.dumps({'questions':40,'systems':5,'MAS_queries':200,
              'train_query_sources':counts,'bank':str(sourcebank),'run':str(root)},ensure_ascii=False),flush=True)
    print('REVIEW',root/'review_40_fresh_questions.tsv',flush=True)

def execute(root,rd):
    if os.environ.get('V59_APPROVE_MAS_EXECUTIONS')!='200':raise RuntimeError('Missing V59 paid call approval')
    if not os.environ.get('OPENAI_API_KEY'):raise RuntimeError('Missing OPENAI_API_KEY')
    dest=root/f'responses_dev_round_{rd:02}.jsonl'
    if dest.exists():raise RuntimeError('Refuse repeat billable execution: '+str(dest))
    print('MAS5_V59_EXECUTE',rd,'100 NEW calls',flush=True)
    cmd=[sys.executable,'-u','scripts/adaptive_v11_execute_round.py','--round',str(rd),
         '--split','dev','--root',str(root),'--manifest',str(v56.SIMPLE),'--workers','3','--retries','1']
    subprocess.run(cmd,cwd=P,check=True)
    expected={(x['session_id'],x['target_role']):x for x in lines(root/f'candidates_dev_round_{rd:02}.jsonl')}
    got=lines(dest);seen={};safe=[]
    for row in got:
        key=(row['session_id'],row['target_role'])
        if key in seen or key not in expected:raise RuntimeError('Duplicate/unknown response')
        seen[key]=True
        if row['query']!=expected[key]['query'] or row.get('execution_status')!='ok' or not row.get('final_output'):
            raise RuntimeError('MAS execution invalid '+str(key))
        safe.append({k:row[k] for k in ('session_id','target_role','round','query','candidate_index','final_output','execution_status') if k in row})
    if len(safe)!=100 or set(seen)!=set(expected):raise RuntimeError('Incorrect MAS result cardinality')
    write_once(root/f'responses_public_dev_round_{rd:02}.jsonl',safe)
    print('MAS5_V59_EXECUTE_SUCCESS',rd,'new_responses',len(safe),flush=True)

def adapter_from_v57(v57root):
    report=v57root/'training_report.json'
    if not report.is_file():raise RuntimeError('V57 training_report.json not present; wait until training completed')
    obj=read(report)
    if obj.get('status')!='FINISHED':raise RuntimeError('V57 training not FINISHED')
    path=Path(obj['selected_adapter']).resolve()
    if not path.is_relative_to(v57root.resolve()):
        raise RuntimeError('Selected model was not a new V57 checkpoint (no training gain); refusing false new/old comparison')
    if not (path/'adapter_config.json').is_file():raise RuntimeError('Selected V57 adapter missing')
    return path,obj

def rows_for_sensor(root):
    records=[];group=collections.defaultdict(dict)
    for rd in (1,2):
        expect={(x['session_id'],x['target_role']):x for x in lines(root/f'candidates_dev_round_{rd:02}.jsonl')}
        seen=set()
        for x in lines(root/f'responses_public_dev_round_{rd:02}.jsonl'):
            k=(x['session_id'],x['target_role']);c=expect.get(k)
            if c is None or k in seen or x['query']!=c['query'] or x['execution_status']!='ok':
                raise RuntimeError('Invalid fresh response')
            if not str(x.get('final_output','')).strip():raise RuntimeError('Empty fresh MAS answer')
            seen.add(k)
            row={'system_id':k[0],'role':k[1],'query':x['query'],'answer':x['final_output'],
                 'round':rd,'kind':'single'}
            records.append(row);group[k][rd]=row
        if len(seen)!=100 or set(expect)!=seen:raise RuntimeError('Incomplete 100 fresh inputs')
    for k,r in group.items():
        if set(r)!={1,2}:raise RuntimeError('Incomplete fresh dual observation')
        a,b=r[1],r[2]
        records.append({'system_id':k[0],'role':k[1],'kind':'dual','round':0,'query':a['query'],
                        'answer':a['answer'],'query2':b['query'],'answer2':b['answer']})
    if len(records)!=300:raise RuntimeError('Expected 200 single + 100 dual')
    return records

def score(root,which):
    import torch
    from peft import PeftConfig,PeftModel
    from transformers import AutoTokenizer,AutoModelForSequenceClassification,BitsAndBytesConfig
    if which not in ('old','v57'):raise RuntimeError('Choose old/v57')
    result=root/f'fresh_scores_{which}.jsonl'
    if result.exists():raise RuntimeError('Refusing overwrite existing Sensor scores '+str(result))
    v57path,_=adapter_from_v57(Path(read(root/'public_preflight.json')['v57_root']))
    path=SENSOR_OLD if which=='old' else v57path
    if not torch.cuda.is_available() or '4090' not in torch.cuda.get_device_name(0):
        raise RuntimeError('Expected 4090 GPU for Sensor')
    cfg=PeftConfig.from_pretrained(str(path),local_files_only=True)
    if cfg.base_model_name_or_path!='Qwen/Qwen2.5-3B-Instruct':raise RuntimeError('Unexpected Sensor base model')
    tok=AutoTokenizer.from_pretrained(str(path),local_files_only=True)
    if tok.pad_token_id is None:tok.pad_token=tok.eos_token
    tok.padding_side='right';tok.truncation_side='right'
    quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,
                             bnb_4bit_compute_dtype=torch.bfloat16)
    base=AutoModelForSequenceClassification.from_pretrained(cfg.base_model_name_or_path,num_labels=1,
           quantization_config=quant,device_map={'':0},dtype=torch.bfloat16,
           attn_implementation='eager',local_files_only=True)
    base.config.pad_token_id=tok.pad_token_id;base.config.use_cache=False
    model=PeftModel.from_pretrained(base,str(path),is_trainable=False,local_files_only=True).eval()
    _roles,descs=v57.roles_and_descriptions()
    examples=rows_for_sensor(root);results=[];maxlen=1280
    with torch.inference_mode():
        for i,x in enumerate(examples,1):
            text=v57.sensor_text(x,tok,descs)
            full=len(tok(text,add_special_tokens=True,truncation=False)['input_ids'])
            enc=tok(text,add_special_tokens=True,return_tensors='pt',max_length=maxlen,truncation=True).to('cuda')
            with torch.autocast(device_type='cuda',dtype=torch.bfloat16):
                z=float(model(**enc).logits.reshape(-1)[0].float().item())
            if not math.isfinite(z):raise RuntimeError('Nonfinite logit')
            results.append({'system_id':x['system_id'],'target_role':x['role'],'round':x['round'],
                'logit':round(z,7),'token_count':full,'truncated':full>maxlen})
            if i%25==0:print('MAS5_V59_SCORE',which,i,'/',len(examples),'truncated',sum(y['truncated'] for y in results),flush=True)
    write_once(result,results)
    atomic_json(root/f'fresh_scores_{which}_metadata.json',{'adapter':str(path),'adapter_training_complete':True,
        'examples':len(results),'input_token_limit':maxlen,'truncated_count':sum(x['truncated'] for x in results),
        'weights_frozen':True,'no_ground_truth_in_inputs':True})
    print('MAS5_V59_SCORE_FINISHED',which,round(len(results)),flush=True)

def evaluate(root):
    pf=read(root/'public_preflight.json');roles=pf['roles'];sids=pf['session_ids']
    truth={str(x['session_id']):set(x['roles']) for x in read(v56.SIMPLE)['systems'] if x.get('split')=='dev'}
    if set(sids)!=set(truth) or any(len(truth[s])!=3 for s in sids):raise RuntimeError('Bad truth')
    outputs={};comparisons=[]
    for model in ('old','v57'):
        data=lines(root/f'fresh_scores_{model}.jsonl')
        if len(data)!=300:raise RuntimeError('Bad number of Sensor scores')
        by={(x['system_id'],x['target_role'],int(x['round'])):x for x in data}
        if len(by)!=300:raise RuntimeError('Duplicate Sensor score')
        outputs[model]={}
        for meth in ('r1','r2','mean','joint'):
            sc={}
            for sid in sids:
                sc[sid]={}
                for role in roles:
                    a=by[(sid,role,1)]['logit'];b=by[(sid,role,2)]['logit'];joint=by[(sid,role,0)]['logit']
                    sc[sid][role]={'r1':a,'r2':b,'mean':(a+b)/2,'joint':joint}[meth]
            outputs[model][meth]=v57.auc20(sc,truth)
    for meth in ('r1','r2','mean','joint'):
        old,new=outputs['old'][meth],outputs['v57'][meth]
        comparisons.append({'method':meth,'old_auc':old['mean_within_mas_auc'],'new_auc':new['mean_within_mas_auc'],
            'delta_auc':round(new['mean_within_mas_auc']-old['mean_within_mas_auc'],6),
            'old_exact_top3':old['exact_top3_count'],'new_exact_top3':new['exact_top3_count'],
            'old_f1_at3':old['mean_role_f1_at3'],'new_f1_at3':new['mean_role_f1_at3']})
    source=Path(pf['v56_root']);train_queries={hashq(x['query']) for rd in (1,2) for x in lines(source/f'responses_public_dev_round_{rd:02}.jsonl')}
    fresh_queries=[x['query'] for x in lines(root/'review_40_fresh_questions.jsonl')]
    query_overlap=sum(hashq(q) in train_queries for q in fresh_queries)
    if query_overlap:raise RuntimeError('FATAL: heldout query overlap with V56 training')
    # Check exact text-response overlaps, solely for audit/report; not a prerequisite to pass.
    train_resp={(x['session_id'],x['target_role'],x['final_output']) for rd in (1,2) for x in lines(source/f'responses_public_dev_round_{rd:02}.jsonl')}
    fresh=[x for rd in (1,2) for x in lines(root/f'responses_public_dev_round_{rd:02}.jsonl')]
    duplicate_answers=sum((x['session_id'],x['target_role'],x['final_output']) in train_resp for x in fresh)
    result={'status':'FRESH_QUERIES_AND_RESPONSES_SAME_FIVE_MAS','systems':5,'roles':20,
            'new_mas_executions':len(fresh),'new_queries_per_role':2,
            'training_queries_exact_overlap':query_overlap,'old_training_answers_byte_identical_count':duplicate_answers,
            'old_sensor':str(SENSOR_OLD),'new_sensor':str(adapter_from_v57(Path(pf['v57_root']))[0]),
            'sensor_frozen_for_test':True,'sensor_training_on_new_responses':False,
            'selected_from_previously_approved_UNEXECUTED_generator_bank':True,
            'within_mas_test_not_new_compositions':True,'not_generalization_to_new_MAS':True,
            'comparison':comparisons,'methods':outputs,
            'primary':'mean','caution':'Same 5 MAS compositions; model selected on prior interactions. '
            'Held out exact questions/responses from V5.7 training; overlaps with unseen historical sources checked in preflight. '
            'Results test transfer to FRESH probes, not new system generalization.'}
    atomic_json(root/'fresh_MAS5_comparison.json',result)
    with (root/'fresh_comparison_summary.tsv').open('x',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(comparisons[0]),delimiter='\t');w.writeheader();w.writerows(comparisons)
    print('MAS5_V59_FINAL',json.dumps({'primary':'mean','old_auc':outputs['old']['mean']['mean_within_mas_auc'],
         'v57_auc':outputs['v57']['mean']['mean_within_mas_auc'],
         'old_exact3':outputs['old']['mean']['exact_top3_count'],
         'v57_exact3':outputs['v57']['mean']['exact_top3_count'],
         'identical_old_response_count':duplicate_answers},ensure_ascii=False),flush=True)
    print('REPORT',root/'fresh_MAS5_comparison.json',flush=True)

def selftest():
    from analysis.role_direct_within_v4 import v57_sensor_night
    truth={'one':{'a','b','c'}}
    scores={'one':{'a':4,'b':3,'c':2,'d':1,'e':0}}
    # synthetic math cardinality test with correct 20 roles
    truth={'one':set('abc')}
    scores={'one':{chr(ord('a')+i):float(20-i) for i in range(20)}}
    assert v57_sensor_night.auc20(scores,truth)['mean_within_mas_auc']==1
    assert norm('What,  details?')=='what details'
    assert jac('What observations should I provide about slow response times?',
        'Which CPU metrics are relevant to diagnosing deployment latency?')<.8
    print('MAS5_V59_SELFTEST_OK',flush=True)

def main():
    a=argparse.ArgumentParser();a.add_argument('stage',choices=('prepare','execute1','execute2','score_old','score_v57','report','selftest'))
    a.add_argument('--root',required=False);x=a.parse_args()
    if x.stage=='selftest':selftest();return
    if not x.root:raise RuntimeError('--root is required')
    root=check_root(x.root)
    if x.stage=='prepare':prepare(root,source_v56(),source_v57())
    else:
        if not (root/'public_preflight.json').is_file():raise RuntimeError('Run has not been prepared')
        {'execute1':lambda:execute(root,1),'execute2':lambda:execute(root,2),
         'score_old':lambda:score(root,'old'),'score_v57':lambda:score(root,'v57'),
         'report':lambda:evaluate(root)}[x.stage]()
if __name__=='__main__':main()
