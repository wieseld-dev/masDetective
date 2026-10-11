#!/usr/bin/env python3
"""V4 safe continuation, OFFLINE only: no target requests, no writes outside V4.
Stages: prepare, selector, sensor, generator, probe_audit, report.
The prior generator/sensor adapters are READ ONLY. V4 uses public responses at test time.
"""
from __future__ import annotations
import argparse, ast, collections, hashlib, json, math, os, random, re, shutil, statistics, time
from pathlib import Path

PROJECT=Path('/home/wieseld/mas_rl_detective')
BASE=PROJECT/'artifacts/role_direct_within_v4'
SCRIPT=PROJECT/'analysis/role_direct_within_v4/pipeline.py'
SENSOR=PROJECT/'artifacts/role_direct_sensor_final_v1/adapter'
MATCHED=PROJECT/'artifacts/role_direct_matched_cf_v1'
V137=PROJECT/'artifacts/v137_presence_v1/dataset_v1'
STARTS=[PROJECT/'artifacts/v14_benign_policy', PROJECT/'artifacts/v11_policy']
SEED=20261009

def writej(path,x):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(x,indent=2,ensure_ascii=False,default=str)+'\n')
    os.replace(tmp,path)
def readj(path): return json.loads(Path(path).read_text())
def readl(path):
    with Path(path).open() as f:return [json.loads(x) for x in f if x.strip()]
def writel(path,items):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w') as f:
        for x in items:f.write(json.dumps(x,ensure_ascii=False)+'\n')
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def safe_root(path):
    p=Path(path).absolute()
    if p.exists() and p.is_symlink():raise RuntimeError('Run root must not be a symlink')
    parent=p.parent.resolve()
    if parent!= (BASE/'runs').resolve():raise RuntimeError('Refusing output outside V4 runs/')
    if not re.fullmatch(r'v4_[0-9]{8}_[0-9]{6}_[0-9]+',p.name):raise RuntimeError('Invalid run directory name')
    return p

def catalog():
    # Read the user's unchanged existing generator taxonomy without importing GPU packages.
    p=PROJECT/'analysis/role_direct_approach/generate_direct_panel_v1.py'
    tree=ast.parse(p.read_text())
    found={}
    for node in tree.body:
        if isinstance(node,ast.Assign):
            for dest in node.targets:
                if isinstance(dest,ast.Name) and dest.id in ('ROLES','DESCRIPTIONS','CONFUSABLES','STRATEGIES'):
                    found[dest.id]=ast.literal_eval(node.value)
    if set(found)!={'ROLES','DESCRIPTIONS','CONFUSABLES','STRATEGIES'} or len(found['ROLES'])!=20:
        raise RuntimeError('Existing 20-role catalog changed')
    return found

def original_adapter(path):
    p=Path(path)
    return p if (p/'adapter_config.json').is_file() and any((p/x).is_file() for x in ('adapter_model.safetensors','adapter_model.bin')) else None

def generator_adapter():
    for candidate in STARTS:
        if original_adapter(candidate):return candidate
        if candidate.is_dir():
            found=[p.parent for p in candidate.rglob('adapter_config.json') if original_adapter(p)]
            found=[p for p in found if len(p.relative_to(candidate).parts)<=3]
            if found:return sorted(found,key=lambda x:(len(x.parts),str(x)))[0]
    raise RuntimeError('No previous V14/V11 generator LoRA adapter found; refusing from-scratch replacement')

def clean(q):return re.sub(r'\s+',' ',str(q or '')).strip()
def norm(q):return re.sub(r'\W+',' ',str(q).casefold()).strip()
UNSAFE=('system prompt','ignore previous','hidden instructions','which agent','list your agents','your internal tools','reveal your','prompt injection')
META=re.compile(r'\b(if you were|as an? (?:expert|specialist|researcher|developer)|what would you do|what is your role|your capabilities|could you please provide guidance|can you guide me|how would you typically)\b',re.I)

def quality(q):
    q=clean(q); words=re.findall(r"\b[\w'-]+\b",q)
    if not (13<=len(words)<=160) or len(q)>1300:return False,'length'
    if any(x in q.casefold() for x in UNSAFE):return False,'internal/unsafe'
    if META.search(q):return False,'meta_template'
    if q.count('?')>2:return False,'excess_questions'
    # An instruction involving an actual task must not be nothing but a request for advice.
    if re.search(r'^\s*(?:what are the best ways|how can i improve|what steps should i take|can you explain|tell me how to)',q,re.I) and len(words)<35:
        return False,'generic_advice'
    return True,'ok'

def template(q):
    toks=norm(q).split(); return ' '.join(toks[:8])
def select_data():
    cat=catalog();roles=set(cat['ROLES'])
    train=readl(V137/'train_v137.jsonl');dev=readl(V137/'dev_v137.jsonl')
    for name,rows in [('train',train),('dev',dev)]:
        if not rows:raise RuntimeError(f'No {name} dataset')
        for x in rows:
            if str(x['target_role']) not in roles or int(x['label']) not in (0,1):raise RuntimeError('Bad role/label')
            for k in ('system_id','target_role','query','final_output','label'):
                if k not in x or not str(x[k]).strip():raise RuntimeError(f'Invalid {name} entry: {k}')
    ids=lambda rows:{str(x['system_id']) for x in rows}
    pairs=lambda rows:{str(x['pair_id']) for x in rows if x.get('pair_id') is not None}
    if ids(train)&ids(dev) or pairs(train)&pairs(dev):raise RuntimeError('System or counterfactual pair leakage')
    sig=lambda x:(str(x['target_role']),norm(x['query']),norm(x['final_output']))
    if {sig(x) for x in train}&{sig(x) for x in dev}:raise RuntimeError('Exact response leakage')
    if len(ids(dev))<10 or len(roles)!=20:raise RuntimeError('Unexpected split shape')
    return train,dev,cat

def prepare(root):
    if root.exists() and any(root.iterdir()):raise RuntimeError('This run root already has files')
    root.mkdir(parents=True,exist_ok=False)
    train,dev,cat=select_data();gen=generator_adapter()
    if not original_adapter(SENSOR):raise RuntimeError('Old sensor adapter not found')
    for key,path in [('sensor',SENSOR),('generator',gen)]:
        conf=readj(path/'adapter_config.json')
        base=str(conf.get('base_model_name_or_path',''))
        expected='Qwen/Qwen2.5-3B-Instruct' if key=='sensor' else 'Qwen/Qwen2.5-1.5B-Instruct'
        if base!=expected:raise RuntimeError(f'{key} adapter base mismatch: {base} != {expected}')
    positives=readl(MATCHED/'responses_dev_round_01.jsonl')
    pairs=readl(MATCHED/'matched_probe_comparisons.jsonl')
    if len(positives)!=600 or len(pairs)!=300:raise RuntimeError('Unexpected matched pool counts')
    # Activation is a HARD TRAINING LABEL gate: must route the target role in PRESENT.
    curated=[];used=set();prefixes=collections.Counter();stats=collections.Counter()
    for x in positives:
        if x.get('condition')!='TARGET_PRESENT':continue
        role=str(x['target_role']);q=clean(x.get('query',''))
        stats['positive_total']+=1
        if role not in set(x.get('selected_roles_training_only',[])):
            stats['not_target_routed']+=1;continue
        stats['target_routed']+=1
        ok,reason=quality(q)
        if not ok:stats['reject_'+reason]+=1;continue
        identity=(role,norm(q));pref=(role,template(q))
        if identity in used or prefixes[pref]>=2:stats['reject_duplicate']+=1;continue
        prefixes[pref]+=1;used.add(identity)
        curated.append({'role':role,'query':q,'strategy':str(x.get('probe_strategy','CONCRETE ARTIFACT')),
                        'activation_training_verified':True,'source_pair_id':str(x.get('pair_id'))})
    # Note: no V137 dev, no Simple5 rows, no sealed panel used for training.
    writel(root/'data/generator_activation_sft.jsonl',curated)
    # Selector labels derive from OLD sensor plus privileged train-only routing; PROXY not ground-truth EIG.
    selector=[]
    for x in pairs:
        q=clean(x.get('query',''));role=str(x['target_role'])
        if not q or role not in cat['ROLES']:continue
        a=bool(x.get('target_routed_pos'));win=float(x.get('v135_win',0));delta=float(x.get('v135_delta',0))
        if not math.isfinite(delta):continue
        utility=(0.75*win+0.25/(1+math.exp(-max(-20,min(20,delta))))) * float(a)
        selector.append({'pair_id':str(x['pair_id']),'role':role,'query':q,'utility_proxy':utility,
                         'target_activated_in_present':a,'old_sensor_win':win})
    writel(root/'data/selector_training_proxy.jsonl',selector)
    observed={x['role'] for x in curated}
    if len(curated)<55 or len(observed)<12:raise RuntimeError(f'Generator curated set too sparse {len(curated)} roles={len(observed)}; stop rather than train weak proxy')
    if len(selector)<200:raise RuntimeError('Missing selector data')
    sources={str(p.relative_to(PROJECT)):sha(p) for p in [V137/'train_v137.jsonl',V137/'dev_v137.jsonl',MATCHED/'responses_dev_round_01.jsonl',MATCHED/'matched_probe_comparisons.jsonl',SENSOR/'adapter_config.json',gen/'adapter_config.json']}
    writej(root/'manifest.json',{'status':'READY_OFFLINE','generator_parent':str(gen),'sensor_parent':str(SENSOR),
        'source_sha256':sources,'train_rows':len(train),'dev_rows':len(dev),
        'train_systems':len({x['system_id'] for x in train}),'dev_systems':len({x['system_id'] for x in dev}),
        'training_generator_probes':len(curated),'training_generator_roles':sorted(observed),
        'training_routing_counts':dict(stats),'selector_proxy_rows':len(selector),
        'no_api':True,'generator_objective':'target activation verified SFT; NOT yet GRPO/validated present-absent gain',
        'selector_objective':'OLD-sensor matched win proxy, not true expected information gain'})
    print('V4_PREPARE_PASS',root,'curated=',len(curated),'generator_base=',gen,flush=True)

def make_sensor_text(role,query,response,descs):
    return (f'You are detecting one candidate specialist capability inside a black-box multi-agent application.\n'
        f'TARGET CAPABILITY:\n{role}\nTARGET CAPABILITY DESCRIPTION:\n{descs[role]}\n'
        f'BENIGN DIRECT PROBE:\n{query}\nVISIBLE FINAL RESPONSE:\n{response}\n'
        'TASK:\nEstimate whether the TARGET CAPABILITY was installed in the hidden application when this response was produced.\n'
        'Use the behavioral evidence in the visible response relative to the probe.\n'
        'Do not infer presence merely because the probe topic matches the target capability.\n')

def system_auc(rows,scores):
    from sklearn.metrics import roc_auc_score
    groups=collections.defaultdict(list)
    for r,z in zip(rows,scores):groups[str(r['system_id'])].append((str(r['target_role']),int(r['label']),float(z)))
    vals=[];exact=[];system_rows=[]
    for sid,entries in sorted(groups.items()):
        y=[z[1] for z in entries];v=[z[2] for z in entries]
        if len(set(y))<2:continue
        score=float(roc_auc_score(y,v));k=sum(y);ordered=sorted(entries,key=lambda x:(-x[2],x[0]))
        correct={r for r,t,z in entries if t};top={r for r,_,_ in ordered[:k]}
        vals.append(score);exact.append(top==correct)
        system_rows.append({'id':sid,'auc':score,'exact':top==correct,'k':k,'topk':sorted(top)})
    if len(vals)<10:raise RuntimeError('Too few testable MAS systems')
    return {'mean_within_mas_auc':statistics.mean(vals),'min_within_mas_auc':min(vals),
            'exact_topk':sum(exact),'evaluable_systems':len(vals),'systems':system_rows}

def train_sensor(root):
    import torch
    import torch.nn.functional as F
    from peft import PeftModel,prepare_model_for_kbit_training
    from transformers import AutoTokenizer,AutoModelForSequenceClassification,BitsAndBytesConfig
    if not torch.cuda.is_available():raise RuntimeError('Sensor requires GPU')
    train,dev,cat=select_data();random.seed(SEED);torch.manual_seed(SEED)
    resume=os.environ.get('V4_SENSOR_RESUME')=='1'
    initial_parent=Path(readj(root/'manifest.json')['sensor_parent'])
    parent=root/'checkpoints/sensor_epoch01' if resume else initial_parent
    if resume and not (parent/'adapter_config.json').is_file():
        raise RuntimeError('Cannot resume: sensor_epoch01 adapter missing')
    tokenizer=AutoTokenizer.from_pretrained(parent,local_files_only=True)
    if tokenizer.pad_token is None:tokenizer.pad_token=tokenizer.eos_token
    tokenizer.padding_side='right'
    quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16)
    base=AutoModelForSequenceClassification.from_pretrained('Qwen/Qwen2.5-3B-Instruct',num_labels=1,quantization_config=quant,device_map={'':0},dtype=torch.bfloat16,local_files_only=True)
    base.config.pad_token_id=tokenizer.pad_token_id;base.config.use_cache=False
    base=prepare_model_for_kbit_training(
        base,use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant':False})
    model=PeftModel.from_pretrained(base,str(parent),is_trainable=True)
    if not any(p.requires_grad for p in model.parameters()):raise RuntimeError('Sensor has no trainable parameters')
    optim=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=2e-5,weight_decay=0.01)
    descs=cat['DESCRIPTIONS']
    grouped=collections.defaultdict(lambda:{0:[],1:[]})
    for x in train:grouped[str(x['system_id'])][int(x['label'])].append(x)
    system_ids=[sid for sid,g in grouped.items() if g[0] and g[1]]
    if len(system_ids)<15:raise RuntimeError('Need systems with both positive and negative labels')
    def predict(rows,batch=6):
        scores=[];model.eval()
        with torch.no_grad():
            for at in range(0,len(rows),batch):
                items=rows[at:at+batch]
                enc=tokenizer([make_sensor_text(x['target_role'],x['query'],x['final_output'],descs) for x in items],
                              padding=True,truncation=True,max_length=1024,return_tensors='pt').to('cuda')
                with torch.autocast(device_type='cuda',dtype=torch.bfloat16):z=model(**enc).logits.view(-1).float()
                scores.extend(z.cpu().tolist())
        return scores
    if resume:
        baseline=readj(root/'reports/sensor_baseline.json')
        prior=readj(root/'reports/sensor_epoch01.json')
        best=prior['metrics']['mean_within_mas_auc']
        bestpath=str(parent)
        start_epoch=2
        print('SENSOR_RESUME_FROM_EPOCH01',bestpath,'AUC',best,
              'optimizer=reinitialized',flush=True)
    else:
        baseline=system_auc(dev,predict(dev));best=baseline['mean_within_mas_auc']
        writej(root/'reports/sensor_baseline.json',baseline)
        print('SENSOR_BASELINE',best,flush=True)
        bestpath=str(parent)
        start_epoch=1
    for epoch in range(start_epoch,4):
        model.train();rng=random.Random(SEED+epoch);pairs=[]
        for sid in system_ids:
            g=grouped[sid];p=g[1];n=g[0]
            for ix in range(min(12,max(4,len(p)*2))):
                pairs.append((rng.choice(p),rng.choice(n)))
        rng.shuffle(pairs)
        losses=[];optim.zero_grad(set_to_none=True);accum=4
        for step in range(0,len(pairs),2):
            batch=pairs[step:step+2];b=len(batch)
            items=[x[0] for x in batch]+[x[1] for x in batch]
            enc=tokenizer([make_sensor_text(x['target_role'],x['query'],x['final_output'],descs) for x in items],
                          padding=True,truncation=True,max_length=1024,return_tensors='pt').to('cuda')
            with torch.autocast(device_type='cuda',dtype=torch.bfloat16):logits=model(**enc).logits.view(-1).float()
            pos,neg=logits[:b],logits[b:]
            labels=torch.cat([torch.ones_like(pos),torch.zeros_like(neg)])
            bce=F.binary_cross_entropy_with_logits(logits,labels)
            ranking=F.softplus(-(pos-neg)).mean()
            loss=bce+0.65*ranking
            if not torch.isfinite(loss):raise RuntimeError('Non-finite sensor loss')
            (loss/accum).backward();losses.append(float(loss.detach().cpu()))
            if ((step//2+1)%accum==0) or step+2>=len(pairs):
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.0)
                optim.step();optim.zero_grad(set_to_none=True)
            if (step//2+1)%35==0:print('SENSOR_STEP',epoch,step//2+1,len(pairs)//2,'loss',round(statistics.mean(losses[-35:]),4),flush=True)
        metric=system_auc(dev,predict(dev))
        path=root/f'checkpoints/sensor_epoch{epoch:02d}'
        path.mkdir(parents=True,exist_ok=False)
        model.save_pretrained(path);tokenizer.save_pretrained(path)
        writej(root/f'reports/sensor_epoch{epoch:02d}.json',{'epoch':epoch,'train_loss':statistics.mean(losses),'metrics':metric,'adapter':str(path)})
        print('SENSOR_EPOCH',epoch,'within_auc',metric['mean_within_mas_auc'],'baseline',baseline['mean_within_mas_auc'],flush=True)
        if metric['mean_within_mas_auc']>best+1e-5:best=metric['mean_within_mas_auc'];bestpath=str(path)
        elif epoch>=2:break
    writej(root/'reports/sensor_selected.json',{'baseline_auc':baseline['mean_within_mas_auc'],
           'selected_auc':best,'selected_adapter':bestpath,
           'decision':'continuation' if bestpath!=str(parent) else 'KEEP_ORIGINAL',
           'dev_is_development_not_sealed_test':True})
    print('V4_SENSOR_PASS',bestpath,'AUC',best,flush=True)

def train_selector(root):
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.metrics import mean_absolute_error
    import joblib
    cat=catalog();rows=readl(root/'data/selector_training_proxy.jsonl')
    groups=[r['pair_id'] for r in rows];y=np.array([r['utility_proxy'] for r in rows],dtype=float)
    txt=[f"TARGET ROLE: {r['role']} {cat['DESCRIPTIONS'][r['role']]} QUERY: {r['query']}" for r in rows]
    tr,te=next(GroupShuffleSplit(test_size=.25,n_splits=1,random_state=SEED).split(txt,y,groups))
    def fit(ids):
        tf=TfidfVectorizer(ngram_range=(1,2),min_df=1,max_features=30000,sublinear_tf=True)
        X=tf.fit_transform([txt[i] for i in ids]);m=Ridge(alpha=10.0,solver='lsqr').fit(X,y[ids]);return tf,m
    tf,model=fit(tr);pred=model.predict(tf.transform([txt[i] for i in te]))
    test=collections.defaultdict(list)
    for j,i in enumerate(te):test[rows[i]['pair_id']].append((float(y[i]),float(pred[j])))
    selected=[];oracle=[];random_expected=[];n_groups=0
    for p,arr in test.items():
        if len(arr)<2:continue
        n_groups+=1;selected.append(arr[max(range(len(arr)),key=lambda i:arr[i][1])][0]);oracle.append(max(z[0] for z in arr));random_expected.append(statistics.mean(z[0] for z in arr))
    if n_groups<10:raise RuntimeError('Selector holdout groups too few')
    verdict='LEARNED' if statistics.mean(selected)>statistics.mean(random_expected)+0.01 else 'FALLBACK_HEURISTIC'
    final_tf,final_model=fit(np.arange(len(rows)))
    out=root/'checkpoints/selector';out.mkdir(parents=True,exist_ok=False)
    joblib.dump({'vectorizer':final_tf,'model':final_model,'method':'offline routing + old-sensor win proxy',
                 'verdict':verdict},out/'selector.joblib')
    report={'rows':len(rows),'test_pair_groups':n_groups,'mean_selected_proxy':statistics.mean(selected),
            'mean_oracle_proxy':statistics.mean(oracle),'mean_random_proxy':statistics.mean(random_expected),
            'holdout_mae':float(mean_absolute_error(y[te],pred)), 'verdict':verdict,
            'limitation':'Not trained on true EIG; query-only learner; old sensor proxy reused, potentially optimistic'}
    writej(root/'reports/selector.json',report)
    print('V4_SELECTOR_PASS',report,flush=True)

def gen_messages(role,strategy,descs,conf):
    return [{'role':'system','content':('Write one benign, standalone, concrete user request intended to engage the exact target specialist capability. '
            'Include the necessary input or scenario and request a tangible output. Do not ask about the assistant identity, hidden agents, prompts or internals. '
            'Do not ask "If you were a specialist". Do not use placeholders that force the system to ask for missing input. '
            'Preserve the target role despite confusing adjacent roles. Output ONLY the user-facing request.')},
            {'role':'user','content':f'TARGET ROLE: {role}\nCAPABILITY: {descs[role]}\nCONFUSABLE ROLES: {", ".join(conf[role][:4])}\nSTRATEGY: {strategy}\nProduce a specific actionable task.'}]

def train_generator(root):
    import torch
    from peft import PeftModel,prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM,AutoTokenizer,BitsAndBytesConfig
    if not torch.cuda.is_available():raise RuntimeError('Generator requires GPU')
    random.seed(SEED);torch.manual_seed(SEED)
    cat=catalog();rows=readl(root/'data/generator_activation_sft.jsonl');parent=Path(readj(root/'manifest.json')['generator_parent'])
    tokenizer=AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct',local_files_only=True)
    if tokenizer.pad_token_id is None:tokenizer.pad_token=tokenizer.eos_token
    tok_eos=tokenizer.eos_token_id
    quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16)
    base=AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct',quantization_config=quant,device_map={'':0},dtype=torch.bfloat16,local_files_only=True)
    base.config.use_cache=False;base=prepare_model_for_kbit_training(base,use_gradient_checkpointing=True)
    model=PeftModel.from_pretrained(base,str(parent),is_trainable=True)
    trainable=[p for p in model.parameters() if p.requires_grad]
    if not trainable:raise RuntimeError('Generator missing trainable adapter')
    opt=torch.optim.AdamW(trainable,lr=8e-6,weight_decay=0.005)
    max_len=768;used=[]
    for row in rows:
        prompt=tokenizer.apply_chat_template(gen_messages(row['role'],row['strategy'],cat['DESCRIPTIONS'],cat['CONFUSABLES']),tokenize=False,add_generation_prompt=True)
        ids=tokenizer(prompt,add_special_tokens=False)['input_ids'];q=tokenizer(row['query'],add_special_tokens=False)['input_ids']
        ids=ids+q+[tok_eos];labels=[-100]*len(tokenizer(prompt,add_special_tokens=False)['input_ids'])+q+[tok_eos]
        if len(ids)>max_len:continue
        used.append((ids,labels))
    if len(used)<55:raise RuntimeError(f'Not enough tokenized activation-verified examples ({len(used)})')
    model.train();random.shuffle(used);accum=8;losses=[];opt.zero_grad(set_to_none=True)
    for i,(ids,labels) in enumerate(used):
        enc={'input_ids':torch.tensor([ids],device='cuda'),'attention_mask':torch.ones((1,len(ids)),dtype=torch.long,device='cuda'),
             'labels':torch.tensor([labels],device='cuda')}
        with torch.autocast(device_type='cuda',dtype=torch.bfloat16):loss=model(**enc).loss.float()
        if not torch.isfinite(loss):raise RuntimeError('Non-finite generator loss')
        (loss/accum).backward();losses.append(float(loss.detach().cpu()))
        if (i+1)%accum==0 or i==len(used)-1:
            torch.nn.utils.clip_grad_norm_(trainable,1.0);opt.step();opt.zero_grad(set_to_none=True)
        if (i+1)%40==0:print('GENERATOR_SFT',i+1,len(used),'loss',statistics.mean(losses[-40:]),flush=True)
    out=root/'checkpoints/generator_activation_sft';out.mkdir(parents=True,exist_ok=False)
    model.save_pretrained(out);tokenizer.save_pretrained(out)
    writej(root/'reports/generator_sft.json',{'initial_adapter':str(parent),'new_adapter':str(out),
        'training_examples':len(used),'mean_loss':statistics.mean(losses),'epochs':1,'learning_rate':8e-6,
        'target_activation':'Verified from privileged PRESENT routing traces on TRAIN samples',
        'discrimination':'NOT optimized yet; this is a constrained continuation warm-start',
        'do_not_call_it_grpo':True})
    print('V4_GENERATOR_SFT_PASS',str(out),flush=True)

def audit_generator(root):
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM,BitsAndBytesConfig
    from peft import PeftModel
    if not torch.cuda.is_available():raise RuntimeError('Generator audit requires GPU')
    cat=catalog();parent=Path(readj(root/'manifest.json')['generator_parent'])
    adapted=root/'checkpoints/generator_activation_sft'
    tokenizer=AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct',local_files_only=True)
    quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16)
    base=AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct',quantization_config=quant,device_map={'':0},dtype=torch.bfloat16,local_files_only=True)
    # Test both adapters using exactly the same prompt, strategy and random seed.
    lm=PeftModel.from_pretrained(base,str(parent),adapter_name='previous',is_trainable=False).eval()
    lm.load_adapter(str(adapted),adapter_name='candidate',is_trainable=False)
    all_rows=[];reports={}
    for adapter in ('previous','candidate'):
        lm.set_adapter(adapter)
        samples=[];templates=collections.Counter();bad=collections.Counter()
        for ri,role in enumerate(cat['ROLES']):
            for ix,(strategy,_) in enumerate(cat['STRATEGIES']):
                torch.manual_seed(SEED+ri*100+ix)
                messages=gen_messages(role,strategy,cat['DESCRIPTIONS'],cat['CONFUSABLES'])
                enc=tokenizer(tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True),add_special_tokens=False,return_tensors='pt').to('cuda')
                with torch.inference_mode():y=lm.generate(**enc,max_new_tokens=180,do_sample=True,temperature=.85,top_p=.92,pad_token_id=tokenizer.eos_token_id)
                q=clean(tokenizer.decode(y[0,enc['input_ids'].shape[1]:],skip_special_tokens=True))
                ok,reason=quality(q);bad[reason]+=int(not ok);templates[template(q)]+=1
                samples.append({'adapter':adapter,'role':role,'strategy':strategy,'query':q,'valid':ok,'rejection_reason':None if ok else reason})
            print('GEN_AUDIT',adapter,ri+1,20,flush=True)
        all_rows+=samples
        reports[adapter]={'n':len(samples),'valid_rate':sum(x['valid'] for x in samples)/len(samples),
            'unique_first_8_word_ratio':len(templates)/len(samples),
            'repeated_templates':sum(n-1 for n in templates.values() if n>1),
            'failure_reasons':dict(bad)}
    writel(root/'reports/generator_probe_samples.jsonl',all_rows)
    old,new=reports['previous'],reports['candidate']
    keep=(new['valid_rate']>=old['valid_rate']-.025 and new['unique_first_8_word_ratio']>=old['unique_first_8_word_ratio']-.025)
    writej(root/'reports/generator_probe_audit.json',{'metrics':reports,
        'selected_adapter':str(adapted if keep else parent),
        'decision':'CANDIDATE_SURFACE_PASS' if keep else 'FALLBACK_PREVIOUS',
        'warning':'Surface and anti-template gate ONLY; activation and PRESENT/ABSENT discrimination on new probes NOT measured. No automated real-MAS calls.'})
    print('V4_GENERATOR_AB_AUDIT_PASS','new_selected',keep,flush=True)

def report(root):
    manifest=readj(root/'manifest.json')
    results={'manifest':manifest,'stages':{}}
    for name,path in [('sensor','reports/sensor_selected.json'),('selector','reports/selector.json'),
        ('generator','reports/generator_sft.json'),('generator_audit','reports/generator_probe_audit.json')]:
        p=root/path
        results['stages'][name]=readj(p) if p.exists() else {'status':'MISSING_OR_FAILED'}
    result=root/'reports/final.json';writej(result,results)
    print('V4_OFFLINE_FINAL',result,flush=True)
    for name,value in results['stages'].items():print('STAGE',name, 'PASS' if 'status' not in value else value['status'],flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','selector','sensor','generator','probe_audit','report']);p.add_argument('--root',required=True)
    a=p.parse_args();root=safe_root(a.root)
    if not PROJECT.is_dir():raise RuntimeError('BGU project directory not found')
    if a.stage=='prepare':prepare(root)
    else:
        if not (root/'manifest.json').is_file():raise RuntimeError('Preflight manifest missing')
        {'selector':train_selector,'sensor':train_sensor,'generator':train_generator,'probe_audit':audit_generator,'report':report}[a.stage](root)
if __name__=='__main__':main()
