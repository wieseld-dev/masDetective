#!/usr/bin/env python3
"""V5.7: closed-world MAS5 Sensor continuation, ~9-hour *maximum*, no target/API calls.

Learns one shared role-conditioned verifier from independently observed final answers.
Training supervision: V5.6 MAS5 labels (user-authorized), V44 matched PRESENT/ABSENT
same-query comparisons if available, V5.4 matched diagnostics, V137 historical TRAIN.
Never inserts labels, routing traces or privileged role membership into model text.

Evaluation on the trained V5.6 five MAS is IN-SAMPLE feasibility, NOT held-out success.
"""
from __future__ import annotations
import argparse, collections, hashlib, json, math, os, random, statistics, time
from pathlib import Path

P=Path(os.environ.get('V57_PROJECT','/home/wieseld/mas_rl_detective')).resolve()
DEFAULT_PARENT=P/'artifacts/role_direct_within_v4/runs/v4_20261009_185359_1475154/checkpoints/sensor_epoch02'
MANIFEST=P/'artifacts/role_direct_routing_pilot_v1/manifest_privileged.json'
V137=P/'artifacts/v137_presence_v1/dataset_v1'
SEED=20261010

def load(p):return json.loads(Path(p).read_text(encoding='utf8'))
def lines(p):
    with Path(p).open(encoding='utf8') as f:return [json.loads(l) for l in f if l.strip()]
def atomic_json(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name(p.name+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False,default=str)+'\n',encoding='utf8')
    os.replace(tmp,p)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(2**20),b''):h.update(chunk)
    return h.hexdigest()

def roles_and_descriptions():
    from analysis.role_direct_within_v4.pipeline import catalog
    c=catalog()
    if len(c['ROLES'])!=20:raise RuntimeError('Expected exactly 20 roles')
    return c['ROLES'],c['DESCRIPTIONS']

def check_paths(run,v56):
    run=Path(run).resolve();v56=Path(v56).resolve()
    if not run.is_relative_to(P/'artifacts/role_direct_within_v4/sensor_v57'):
        raise RuntimeError('V57 output location must be under sensor_v57')
    if not v56.is_relative_to(P/'artifacts/role_direct_within_v4/nonadaptive_v56'):
        raise RuntimeError('V56 source must be an existing nonadaptive_v56 run')
    if not (v56/'public_preflight.json').exists():raise RuntimeError('V56 public_preflight.json missing')
    if not MANIFEST.is_file():raise RuntimeError('MAS5 manifest missing')
    if not (DEFAULT_PARENT/'adapter_config.json').is_file():raise RuntimeError('Frozen V4 Sensor adapter missing')
    return run,v56

def v56_records(v56,require_full=True):
    roles,_=roles_and_descriptions();allowed=set(roles)
    configs={str(s['session_id']):set(s['roles']) for s in load(MANIFEST)['systems'] if s.get('split')=='dev'}
    pf=load(v56/'public_preflight.json');sids=list(pf['session_ids'])
    if len(sids)!=5 or set(sids)!=set(configs):raise RuntimeError('V56 MANIFEST mismatch')
    if any(len(configs[s])!=3 or not configs[s]<=allowed for s in sids):raise RuntimeError('Bad MAS labels')
    output=[]
    for rd in (1,2):
        path=v56/f'responses_public_dev_round_{rd:02}.jsonl'
        if not path.is_file():
            if require_full:raise RuntimeError('Wait for V56 execute'+str(rd)+': '+str(path))
            continue
        samples=lines(path)
        candpath=v56/f'candidates_dev_round_{rd:02}.jsonl'
        candidates={(str(c['session_id']),str(c['target_role'])):c for c in lines(candpath)}
        if len(samples)!=100 or len(candidates)!=100:
            raise RuntimeError('Expected 100 public responses/candidates in round '+str(rd))
        seen=set()
        for r in samples:
            k=(str(r['session_id']),str(r['target_role']))
            if k not in candidates or k in seen:raise RuntimeError('Duplicate/missing role key '+str(k))
            seen.add(k)
            if k[0] not in configs or k[1] not in allowed:raise RuntimeError('Unknown MAS or role')
            if r.get('query')!=candidates[k].get('query') or r.get('execution_status')!='ok':
                raise RuntimeError('Invalid V56 response '+str(k))
            if not str(r.get('final_output','')).strip():raise RuntimeError('Empty V56 output '+str(k))
            output.append({'system_id':k[0],'role':k[1],'query':r['query'],'answer':r['final_output'],
                           'round':rd,'label':int(k[1] in configs[k[0]]),'kind':'single','source':'v56'})
        if seen!=set(candidates):raise RuntimeError('Missing V56 query/response pair')
    if require_full and len(output)!=200:raise RuntimeError('V56 execute2 not complete')
    # Build role-specific, ordered dual-answer examples; no privileged features.
    gr=collections.defaultdict(dict)
    for r in output:gr[(r['system_id'],r['role'])][r['round']]=r
    dual=[]
    for (sid,role),rs in gr.items():
        if set(rs)!={1,2}:
            if require_full:raise RuntimeError('Incomplete two-round role '+str((sid,role)))
            continue
        x,y=rs[1],rs[2]
        dual.append({'system_id':sid,'role':role,'label':x['label'],'kind':'dual',
                     'query':x['query'],'answer':x['answer'],'query2':y['query'],'answer2':y['answer'],
                     'source':'v56_dual'})
    return output+dual,configs

def historical_v137():
    trainfile=V137/'train_v137.jsonl';devfile=V137/'dev_v137.jsonl'
    if not trainfile.is_file():raise RuntimeError('V137 training dataset missing')
    def conv(data,source):
        out=[]
        for x in data:
            if not all(str(x.get(k,'')).strip() for k in ('system_id','target_role','query','final_output')):continue
            lab=int(x['label'])
            if lab not in (0,1):continue
            out.append({'system_id':str(x['system_id']),'role':str(x['target_role']),
                        'query':str(x['query']),'answer':str(x['final_output']),
                        'label':lab,'kind':'single','source':source})
        return out
    train=conv(lines(trainfile),'v137_train')
    dev=conv(lines(devfile),'v137_dev') if devfile.is_file() else []
    return train,dev

def matched_from_candidates(cpath,rpath,source):
    cs=lines(cpath);rs=lines(rpath)
    if len(cs)!=len(rs):raise RuntimeError(source+' candidate/response lengths mismatch')
    # Candidate.index is stable per-session; query ensures same probe within counterfactual pair.
    ck={(str(x['session_id']),int(x['candidate_index'])):x for x in cs}
    if len(ck)!=len(cs):raise RuntimeError(source+' duplicate candidates')
    pairs=collections.defaultdict(dict)
    for r in rs:
        k=(str(r['session_id']),int(r['candidate_index']))
        if k not in ck:raise RuntimeError('Unmatched '+source+' response')
        c=ck[k];cond=str(c.get('condition',r.get('condition','')))
        if cond not in ('TARGET_PRESENT','TARGET_ABSENT'):continue
        if r.get('execution_status')!='ok' or not r.get('final_output'):continue
        role=str(c['target_role']);query=str(c['query'])
        if str(r.get('query'))!=query:raise RuntimeError(source+' query mismatch')
        pid=str(c.get('pair_id',r.get('pair_id','')))
        key=(pid,role,query)
        if cond in pairs[key]:raise RuntimeError(source+' duplicate condition')
        pairs[key][cond]={'system_id':str(c['session_id']),'role':role,'query':query,
                          'answer':str(r['final_output']),'label':int(cond=='TARGET_PRESENT'),
                          'kind':'single','source':source,'pair_id':pid}
    out=[]
    for key,g in pairs.items():
        if set(g)=={'TARGET_PRESENT','TARGET_ABSENT'}:out.append((g['TARGET_PRESENT'],g['TARGET_ABSENT']))
    return out

def locate_matched():
    env=os.environ.get('V57_MATCHED_ROOT','')
    if env:
        root=Path(env).resolve()
        if not (root/'candidates_train_round_01.jsonl').is_file():
            raise RuntimeError('V57_MATCHED_ROOT must point to folder containing V44 TRAIN candidate file')
        return root
    base=P/'artifacts/role_direct_within_v4'
    # Prefer V44's already-paid train corpus. Never search V56 DEV sources as matched training.
    found=[]
    for prefix in ('overnight_v44','overnight_v45','overnight_v46'):
        d=base/prefix
        if d.is_dir():
            found.extend(d.glob('*/train/candidates_train_round_01.jsonl'))
            found.extend(d.glob('*/train/train/candidates_train_round_01.jsonl'))
    found=[p.parent for p in found if (p.parent/'responses_train_round_01.jsonl').is_file()]
    if not found:return None
    # Avoid accidentally mistaking tiny pilot for 960 matched executions.
    for path in sorted(found,reverse=True):
        try:
            if len(lines(path/'candidates_train_round_01.jsonl'))>=500:return path
        except Exception:pass
    return None

def v54_matched():
    base=P/'artifacts/role_direct_within_v4/pilot_v54'
    out=[]
    if not base.is_dir():return out
    for c in sorted(base.glob('r2paired_*/candidates_train_round_02.jsonl'),reverse=True):
        r=c.parent/'responses_train_round_02.jsonl'
        if r.is_file():
            out=matched_from_candidates(c,r,'v54')
            if out:return out
    return out

def build_groups(v56):
    vr,truth=v56_records(v56,True)
    historic,dev=historical_v137()
    allowed=set(roles_and_descriptions()[0]);historic=[x for x in historic if x['role'] in allowed]
    dev=[x for x in dev if x['role'] in allowed]
    matched=[]
    root=locate_matched()
    if root:
        matched=matched_from_candidates(root/'candidates_train_round_01.jsonl',root/'responses_train_round_01.jsonl','v44_matched')
        print('V57_MATCHED_SOURCE',str(root),len(matched),flush=True)
    if not matched:print('V57_WARNING: V44 matched training source not located; using other supervised pools',flush=True)
    vm=v54_matched()
    # Assemble pairs. All training labels stay OUTSIDE text construction.
    group=collections.defaultdict(lambda:{0:[],1:[]})
    for row in vr+historic:
        group[(row['source'],row['system_id'],row['kind'],row.get('round',0))][row['label']].append(row)
    mas_pairs=[];hist_pairs=[]
    for (source,sid,kind,rnd),g in group.items():
        if not g[0] or not g[1]:continue
        dest=mas_pairs if source.startswith('v56') else hist_pairs
        # Role-conditioned POS-vs-NEG within SAME MAS. For V56 include all 3x17 contrasts.
        for pos in g[1]:
            negs=g[0] if source.startswith('v56') else g[0][:min(16,len(g[0]))]
            for neg in negs:dest.append((pos,neg))
    pools={'v56':mas_pairs,'matched':matched+vm,'historical':hist_pairs}
    if len(pools['v56'])<300:raise RuntimeError('V56 matched MAS role-ranking rows unexpectedly few')
    if not pools['historical']:print('V57_WARNING: V137 cross-role pair training empty',flush=True)
    return vr,historic,dev,pools,{'v56_rows':len(vr),'v137_train':len(historic),'v137_dev':len(dev),
                                  'v56_within_mas_pairs':len(mas_pairs),'v44_same_query_pairs':len(matched),
                                  'v54_same_query_pairs':len(vm),'v137_within_mas_pairs':len(hist_pairs),
                                  'v44_root':str(root) if root else None,'systems':len(truth)}

def assess(root,v56):
    check_paths(root,v56)
    status='READY' if (v56/'responses_public_dev_round_02.jsonl').is_file() else 'WAITING_FOR_V56_ROUND2'
    r1=v56/'responses_public_dev_round_01.jsonl';r2=v56/'responses_public_dev_round_02.jsonl'
    m=load(MANIFEST);sys=[x for x in m['systems'] if x.get('split')=='dev']
    summary={'status':status,'v56':str(v56),'source_sensor':str(DEFAULT_PARENT),
             'five_systems':len(sys),'truth_roles_per_system':[len(x['roles']) for x in sys],
             'v56_round1_existing':r1.is_file(),'v56_round2_existing':r2.is_file(),
             'v137_train_existing':(V137/'train_v137.jsonl').is_file(),
             'v44_matched_source_found':str(locate_matched()) if locate_matched() else None,
             'max_training_hours':float(os.environ.get('V57_HOURS','8.75')),
             'no_API_calls':True,'all_models_read_only_except_new_run':True,
             'v56_training_labels_used':True,'v56_test_will_be_in_sample':True}
    if status=='READY':
        _,_,_,_,stats=build_groups(v56)
        summary.update(stats)
    print('MAS5_V57_PREFLIGHT',json.dumps(summary,ensure_ascii=False),flush=True)
    return summary

# Text is ONLY target role + description + public question(s) and final answer(s).
# [No pair labels, MAS ID, activation trace or ground truth is ever injected.]
def clip_text(tok,text,max_tokens):
    ids=tok.encode(str(text),add_special_tokens=False)
    if len(ids)<=max_tokens:return str(text)
    left=(max_tokens*3)//4;right=max_tokens-left-12
    return tok.decode(ids[:left],skip_special_tokens=True)+'\n[...middle omitted...]\n'+tok.decode(ids[-max(0,right):],skip_special_tokens=True)

def sensor_text(x,tok,descs):
    from analysis.role_direct_within_v4.pipeline import make_sensor_text
    role=x['role'];q=clip_text(tok,x['query'],130)
    if x['kind']=='single':
        a=clip_text(tok,x['answer'],930)
        return make_sensor_text(role,q,a,descs)
    q2=clip_text(tok,x['query2'],125)
    a1=clip_text(tok,x['answer'],430);a2=clip_text(tok,x['answer2'],430)
    return (f'You are detecting one candidate specialist capability inside a black-box multi-agent application.\n'
            f'TARGET CAPABILITY:\n{role}\nTARGET CAPABILITY DESCRIPTION:\n{descs[role]}\n'
            f'TWO INDEPENDENT BENIGN PROBES OF THE SAME TARGET CAPABILITY:\n'
            f'OBSERVATION 1 - USER QUERY:\n{q}\nOBSERVATION 1 - VISIBLE FINAL RESPONSE:\n{a1}\n'
            f'OBSERVATION 2 - USER QUERY:\n{q2}\nOBSERVATION 2 - VISIBLE FINAL RESPONSE:\n{a2}\n'
            'TASK:\nEstimate whether the TARGET CAPABILITY was installed in the application when these responses were produced. '
            'Use evidence from BOTH answers. Query topic alone is not evidence of installation. '
            'Other installed roles can competently answer related tasks.\n')

def auc20(vals,truth):
    # Exact AUC calculation without sklearn: 3 positives * 17 negatives per system.
    per=[];correct=[];f1=[]
    for sid,roles in sorted(truth.items()):
        group=vals.get(sid,{})
        if len(group)!=20:raise RuntimeError(f'Expected 20 role scores for {sid}, got {len(group)}')
        if len(roles)!=3:raise RuntimeError('Incorrect positives per MAS')
        positives=[group[r] for r in group if r in roles]
        negatives=[group[r] for r in group if r not in roles]
        auc=sum((p>n)+0.5*(p==n) for p in positives for n in negatives)/51
        top3=[r for r,_ in sorted(group.items(),key=lambda kv:(-kv[1],kv[0]))[:3]]
        hits=len(set(top3)&roles)
        per.append({'system_id':sid,'auc':round(auc,6),'top3':top3,
                    'truth_eval_only':sorted(roles),'hits':hits,'exact_top3':hits==3})
        correct.append(hits==3);f1.append(hits/3)
    return {'mean_within_mas_auc':round(statistics.mean(x['auc'] for x in per),6),
            'exact_top3_count':sum(correct),'mean_role_f1_at3':round(statistics.mean(f1),6),
            'per_system':per}

def make_v56_eval_rows(vr):
    # For each role, score R1, R2, and both together.
    return [x for x in vr if x.get('round') in (1,2)]+[x for x in vr if x['kind']=='dual']

def get_device_status():
    import torch
    if not torch.cuda.is_available():raise RuntimeError('Requires CUDA RTX 4090')
    card=torch.cuda.get_device_name(0)
    if '4090' not in card:raise RuntimeError('RTX 4090 required, got '+card)
    return card

def run_training(out,v56,hours,max_steps,seq_len,eval_every,patience):
    import torch
    import torch.nn.functional as F
    from peft import PeftConfig,PeftModel,prepare_model_for_kbit_training
    from transformers import AutoTokenizer,AutoModelForSequenceClassification,BitsAndBytesConfig
    from analysis.role_direct_within_v4.pipeline import catalog
    out,v56=check_paths(out,v56)
    if out.exists() and (out/'training_report.json').exists():raise RuntimeError('Refusing to repeat finished training')
    out.mkdir(parents=True,exist_ok=True)
    card=get_device_status();random.seed(SEED);torch.manual_seed(SEED)
    if seq_len<900 or seq_len>1536:raise RuntimeError('Training sequence limit must be 900..1536')
    vr,hist,dev,pools,stats=build_groups(v56)
    if not pools['v56']:raise RuntimeError('Missing V56 same-system contrasts')
    roles,descs=roles_and_descriptions();rtruth={str(s['session_id']):set(s['roles']) for s in load(MANIFEST)['systems'] if s.get('split')=='dev'}
    cfg=PeftConfig.from_pretrained(str(DEFAULT_PARENT),local_files_only=True)
    if cfg.base_model_name_or_path!='Qwen/Qwen2.5-3B-Instruct':raise RuntimeError('Unexpected Sensor base')
    tok=AutoTokenizer.from_pretrained(str(DEFAULT_PARENT),local_files_only=True)
    if tok.pad_token_id is None:tok.pad_token=tok.eos_token
    tok.padding_side='right';tok.truncation_side='right'
    quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,
                             bnb_4bit_compute_dtype=torch.bfloat16)
    base=AutoModelForSequenceClassification.from_pretrained(cfg.base_model_name_or_path,num_labels=1,
                quantization_config=quant,device_map={'':0},dtype=torch.bfloat16,
                attn_implementation='eager',local_files_only=True)
    base.config.pad_token_id=tok.pad_token_id;base.config.use_cache=False
    base=prepare_model_for_kbit_training(base,use_gradient_checkpointing=True,
                                         gradient_checkpointing_kwargs={'use_reentrant':False})
    model=PeftModel.from_pretrained(base,str(DEFAULT_PARENT),is_trainable=True,local_files_only=True)
    params=[p for p in model.parameters() if p.requires_grad]
    if not params:raise RuntimeError('No trainable Sensor parameters')
    # Pre-tokenize unique observations once. No labels passed into the tokenizer.
    unique={}
    def key(x):
        return (x['kind'],x['role'],x['query'],x['answer'],x.get('query2'),x.get('answer2'))
    for source in (vr,hist,dev):
        for x in source:unique[key(x)]=x
    for pool in pools.values():
        for pos,neg in pool:unique[key(pos)]=pos;unique[key(neg)]=neg
    print('V57_TOKENIZE_START',len(unique),'examples seq_len',seq_len,flush=True)
    cache={};truncated=collections.Counter()
    for i,(k,x) in enumerate(unique.items(),1):
        s=sensor_text(x,tok,descs)
        enc=tok(s,add_special_tokens=True,max_length=seq_len,truncation=True)['input_ids']
        truncated[x['source']]+=int(len(tok(s,add_special_tokens=True,truncation=False)['input_ids'])>seq_len)
        cache[k]=enc
        if i%1000==0:print('V57_TOKENIZED',i,'/',len(unique),flush=True)
    def enc_batch(xs):
        arr=[cache[key(x)] for x in xs]
        length=max(len(z) for z in arr)
        ids=[z+[tok.pad_token_id]*(length-len(z)) for z in arr]
        am=[[1]*len(z)+[0]*(length-len(z)) for z in arr]
        return {'input_ids':torch.tensor(ids,device='cuda',dtype=torch.long),
                'attention_mask':torch.tensor(am,device='cuda',dtype=torch.long)}
    @torch.no_grad()
    def predict(xs,batchsize=3):
        model.eval();scores=[]
        for i in range(0,len(xs),batchsize):
            with torch.autocast(device_type='cuda',dtype=torch.bfloat16):
                z=model(**enc_batch(xs[i:i+batchsize])).logits.view(-1).float()
            scores.extend(z.cpu().tolist())
        return scores
    # Distinct grouping by source to avoid half-POS labels in Sensor inputs.
    evalrows=make_v56_eval_rows(vr)
    if len(evalrows)!=300:raise RuntimeError('Expected 300 single+dual Sensor eval rows')
    # Holdout V137 DEV (not part of this training). Select complete systems; bound GPU time.
    devg=collections.defaultdict(list)
    for x in dev:
        # Do not directly validate on MAS IDs that are part of V56 training.
        if x['system_id'] not in rtruth:devg[x['system_id']].append(x)
    devselected=[]
    for sid,ls in sorted(devg.items()):
        chosen={}
        for z in ls:chosen.setdefault(z['role'],z)
        if len(chosen)==20 and sum(x['label'] for x in chosen.values())==3:
            devselected.extend(chosen.values())
            if len(devselected)>=200:break
    def evaluate(step):
        evscores=predict(evalrows)
        by=collections.defaultdict(lambda:collections.defaultdict(dict))
        for row,score in zip(evalrows,evscores):
            r='joint' if row['kind']=='dual' else 'r'+str(row['round'])
            by[r][row['system_id']][row['role']]=float(score)
        m={r:auc20(by[r],rtruth) for r in ('r1','r2','joint')}
        # Compare with simple two-logit mean on the same adapter.
        means={sid:{role:(by['r1'][sid][role]+by['r2'][sid][role])/2 for role in roles} for sid in rtruth}
        m['logit_mean']=auc20(means,rtruth)
        if devselected:
            devscores=predict(devselected)
            dby=collections.defaultdict(dict);dtruth=collections.defaultdict(set)
            for x,z in zip(devselected,devscores):
                dby[x['system_id']][x['role']]=z
                if x['label']:dtruth[x['system_id']].add(x['role'])
            m['v137_dev']=auc20(dby,dtruth)
        else:m['v137_dev']=None
        metrics={'step':step,'in_sample_mas5':m,'v137_dev_is_separate':bool(devselected),
                 'warning':'MAS5 5-system metrics are IN-SAMPLE training-system scores; not a generalization result'}
        return metrics
    baseline=evaluate(0);atomic_json(out/'baseline_metrics.json',baseline)
    print('V57_BASELINE',json.dumps({k:{'auc':v['mean_within_mas_auc'],'top3':v['exact_top3_count']} if v else None
             for k,v in baseline['in_sample_mas5'].items()}),flush=True)
    opt=torch.optim.AdamW(params,lr=float(os.environ.get('V57_LR','7e-6')),weight_decay=.01)
    start=time.monotonic();deadline=start+hours*3600
    # These weights deliberately prioritize the new-generator distribution, while
    # retaining same-query counterfactual negatives and historical regularization.
    weights={'v56':.56,'matched':.32,'historical':.12}
    active=[k for k in weights if pools[k]];pw=[weights[k] for k in active]
    rng=random.Random(SEED+57)
    best_step=0;best_score=-1e9;best_dir=str(DEFAULT_PARENT)
    history=[];lossbuf=[];bad_evals=0;stopped='max_optimizer_steps'
    selected_metrics=baseline
    def criterion(m):
        ins=m['in_sample_mas5'];auc=ins['joint']['mean_within_mas_auc']
        auc2=ins['logit_mean']['mean_within_mas_auc']
        # In-sample feasibility is the user's explicit objective. Historical dev
        # helps penalize collapse when it is available, not a sealed claim.
        dv=ins.get('v137_dev');pen=max(0,(baseline['in_sample_mas5'].get('v137_dev') or dv or {'mean_within_mas_auc':0})['mean_within_mas_auc']
                       -(dv or {'mean_within_mas_auc':0})['mean_within_mas_auc']-.04) if dv else 0
        return .65*auc+.35*auc2-.5*pen
    base_value=criterion(baseline);best_score=base_value
    atomic_json(out/'current_status.json',{'state':'TRAINING','step':0,'baseline':baseline,
                'elapsed_hours':0,'target_hours':hours,'best_score':best_score,
                'train_sources':stats,'tokenization_truncated':dict(truncated),'gpu':card})
    grad_accum=4;last_eval=0
    for step in range(1,max_steps+1):
        if time.monotonic()>=deadline-7*60:
            stopped='time_budget_7_min_reserved_for_report';break
        model.train();opt.zero_grad(set_to_none=True);batch_losses=[]
        for accumulation in range(grad_accum):
            group=rng.choices(active,weights=pw,k=1)[0]
            pos,neg=rng.choice(pools[group])
            with torch.autocast(device_type='cuda',dtype=torch.bfloat16):
                logits=model(**enc_batch([pos,neg])).logits.reshape(-1).float()
                margin=logits[0]-logits[1]
                # Pairwise ranking + weak BCE anchor: prevent role-topic shortcut.
                loss=F.softplus(-margin)+.18*(F.softplus(-logits[0])+F.softplus(logits[1]))
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite Sensor loss at step '+str(step))
            (loss/grad_accum).backward()
            batch_losses.append(float(loss.detach().cpu()))
        torch.nn.utils.clip_grad_norm_(params,.8);opt.step()
        lossbuf.extend(batch_losses)
        if step%25==0:
            print('V57_STEP',step,'loss',round(statistics.mean(lossbuf[-min(100,len(lossbuf)):]),4),
                  'elapsed_min',round((time.monotonic()-start)/60,1),flush=True)
        if step%eval_every==0 or step==max_steps or time.monotonic()>=deadline-16*60:
            metrics=evaluate(step);cur=criterion(metrics)
            summary={'step':step,'elapsed_hours':round((time.monotonic()-start)/3600,4),
                     'mean_train_loss_recent':round(statistics.mean(lossbuf[-min(100,len(lossbuf)):]),5),
                     'selection_score':round(cur,6),'metrics':metrics}
            history.append(summary)
            atomic_json(out/'latest_eval.json',summary)
            with (out/'metrics_history.jsonl').open('a',encoding='utf8') as f:
                f.write(json.dumps(summary,ensure_ascii=False)+'\n')
            print('V57_EVAL',step,'joint_auc',metrics['in_sample_mas5']['joint']['mean_within_mas_auc'],
                  'mean_auc',metrics['in_sample_mas5']['logit_mean']['mean_within_mas_auc'],
                  'exact3_joint',metrics['in_sample_mas5']['joint']['exact_top3_count'],
                  'score',round(cur,5),'best',round(best_score,5),flush=True)
            if cur>best_score+1e-4:
                best_score=cur;best_step=step;bad_evals=0
                dest=out/f'checkpoints/sensor_step_{step:05d}'
                if dest.exists():raise RuntimeError('Refusing to overwrite checkpoint '+str(dest))
                dest.mkdir(parents=True,exist_ok=False)
                model.save_pretrained(str(dest));tok.save_pretrained(str(dest))
                best_dir=str(dest);selected_metrics=metrics
                atomic_json(out/'best.json',{'step':best_step,'selected_adapter':best_dir,
                          'score':best_score,'metrics':selected_metrics,'in_sample':True})
                print('V57_BEST_SAVED',best_step,best_dir,flush=True)
            else:bad_evals+=1
            atomic_json(out/'current_status.json',{'state':'TRAINING','step':step,'elapsed_hours':summary['elapsed_hours'],
                'target_hours':hours,'best_step':best_step,'selected_adapter':best_dir,
                'best_score':round(best_score,6),'latest':summary,'bad_evals':bad_evals})
            if bad_evals>=patience:
                stopped='validation_plateau';break
            last_eval=step
    else:stopped='max_optimizer_steps'
    elapsed=(time.monotonic()-start)/3600
    report={'status':'FINISHED','stopped':stopped,'elapsed_training_hours':round(elapsed,4),
            'max_hours':hours,'optimizer_steps':step,'best_step':best_step,
            'source_adapter_untouched':str(DEFAULT_PARENT),'selected_adapter':best_dir,
            'baseline':baseline,'selected':selected_metrics,'improved_selection_score':round(best_score-base_value,6),
            'history_evaluations':len(history),'train_source_counts':stats,
            'train_mixture':{k:weights[k] for k in active},'input_max_tokens':seq_len,
            'tokenization_truncated':dict(truncated),'mas_api_calls':0,'selector_training':False,
            'in_sample_mas5_labelled_training':True,'labels_never_in_sensor_inputs':True,
            'v137_dev_rows_excluded_from_direct_training':True,
            'v137_dev_composition_overlap_not_excluded':True,
            'caution':'MAS5 train/test same 5 compositions; in-sample feasibility only. '
                     'Non-determinism does not make these held-out. New unseen systems required for generalization claims.'}
    atomic_json(out/'training_report.json',report)
    atomic_json(out/'current_status.json',{'state':'FINISHED','best_step':best_step,
                'selected_adapter':best_dir,'report_file':str(out/'training_report.json')})
    print('MAS5_V57_COMPLETE',json.dumps({'elapsed_hours':round(elapsed,3),'best_step':best_step,
              'selected_adapter':best_dir,'joint_auc':selected_metrics['in_sample_mas5']['joint']['mean_within_mas_auc'],
              'exact_top3':selected_metrics['in_sample_mas5']['joint']['exact_top3_count'],
              'reason':stopped}),flush=True)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('mode',choices=['preflight','train','selftest'])
    ap.add_argument('--run',type=Path,required=True)
    ap.add_argument('--v56',type=Path,required=True)
    ap.add_argument('--hours',type=float,default=8.75)
    ap.add_argument('--max-steps',type=int,default=6000)
    ap.add_argument('--max-tokens',type=int,default=1280)
    ap.add_argument('--eval-every',type=int,default=150)
    ap.add_argument('--patience',type=int,default=10)
    a=ap.parse_args()
    if a.mode=='preflight':assess(a.run,a.v56)
    elif a.mode=='train':
        if not (0<a.hours<=9):raise RuntimeError('Budget must be (0,9] hours')
        run_training(a.run,a.v56,a.hours,a.max_steps,a.max_tokens,a.eval_every,a.patience)
    else:
        # Pure-Python deterministic unit tests: no GPU, no API, no disk writes.
        truth={f's{i}':{'a','b','c'} for i in range(5)}
        scores={sid:{r:float(r in vals) for r in 'abcdefghijklmnopqrst'} for sid,vals in truth.items()}
        z=auc20(scores,truth)
        assert z['mean_within_mas_auc']==1 and z['exact_top3_count']==5
        zero={sid:{r:0.0 for r in 'abcdefghijklmnopqrst'} for sid in truth}
        assert auc20(zero,truth)['mean_within_mas_auc']==.5
        print('V57_SELFTEST_PASS (mock values only)',flush=True)

if __name__=='__main__':main()
