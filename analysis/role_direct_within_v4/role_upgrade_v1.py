#!/usr/bin/env python3
"""Isolated V4 role-directed offline preference pilot. NO API calls or old-model writes.

CAUTION: The matched counterfactual DEV set was already explored by prior experiments.
Scores (v134/v135) are proxy evidence, NOT ground-truth EIG. Distinct queries ~60.
This does not claim improved attack performance without fresh heldout matched tests.
"""
from __future__ import annotations
import argparse, collections, hashlib, json, math, os, random, re, statistics, sys
from pathlib import Path
from typing import Sequence

from analysis.role_direct_within_v4 import pipeline as old

SEED = 20261009
ROOT_BASE = old.BASE
MATCHED = old.MATCHED / 'matched_probe_comparisons.jsonl'
STRATEGIES = {1:'CONCRETE ARTIFACT', 2:'CONSTRAINED TASK', 3:'SPECIALIST DELIVERABLE'}


def getroot(path):
    p = old.safe_root(path)
    if not (p/'manifest.json').is_file(): raise RuntimeError('Existing V4 root required')
    return p


def udir(root):return root/'role_upgrade_v1'


def sig(x):
    z = max(-10., min(10., float(x)))
    return 1. / (1. + math.exp(-z))


def score(r):
    """Activation gated target-specific discrimination proxy; NOT actual EIG."""
    if not r['target_routed_pos']:return 0.0
    # These are *signed* target-label evidence differences (PRESENT - ABSENT).
    a = sig(float(r['v135_delta']))
    b = sig(float(r['v134_delta']))
    return .55*a+.45*b


def get_rows():
    rows = old.readl(MATCHED)
    if len(rows)!=300:raise RuntimeError('Expected 300 matched comparisons')
    seen=set();groups=collections.defaultdict(list);roles=set(old.catalog()['ROLES'])
    for r in rows:
        role=str(r.get('target_role'));pair=str(r.get('pair_id'));rank=int(r.get('probe_rank',-1))
        if role not in roles or rank not in STRATEGIES or not pair:
            raise RuntimeError('Invalid role/pair/rank')
        key=(role,pair,rank)
        if key in seen:raise RuntimeError('Duplicate matched comparison')
        seen.add(key)
        for k in ('v135_delta','v134_delta','v135_win','v134_win'):
            if not math.isfinite(float(r[k])):raise RuntimeError(f'Nonfinite {k}')
        r['pair_id']=pair;r['target_role']=role;r['probe_rank']=rank
        r['query']=old.clean(r['query']);r['utility']=score(r)
        groups[(role,pair)].append(r)
    if len(groups)!=100 or any(len(v)!=3 for v in groups.values()):
        raise RuntimeError('Expected 100 matched groups x 3 probes')
    # Same query text must appear across 5 matched background pairs at each role/rank.
    byrole=collections.defaultdict(lambda:collections.defaultdict(set))
    for r in rows:
        byrole[r['target_role']][r['probe_rank']].add(old.norm(r['query']))
    if len(byrole)!=20 or any(len(s)!=1 for d in byrole.values() for s in d.values()):
        raise RuntimeError('Unexpected query pool: cannot guarantee held-out roles prevent leakage')
    return rows,groups


def split_roles(rows):
    roles=sorted({x['target_role'] for x in rows})
    rng=random.Random(SEED);rng.shuffle(roles)
    return set(roles[:16]),set(roles[16:])


def prepare(root):
    d=udir(root)
    if d.exists():raise RuntimeError('Upgrade folder already exists; refusing overwrite')
    d.mkdir(parents=True)
    rows, groups=get_rows()
    original=old.readj(root/'manifest.json').get('source_sha256',{}).get('artifacts/role_direct_matched_cf_v1/matched_probe_comparisons.jsonl')
    if original is not None and original!=old.sha(MATCHED):
        raise RuntimeError('Matched comparison file changed after V4 original preflight')
    trainroles,holdout=split_roles(rows)
    # One UNIQUE pair per role, aggregated across 5 counterfactual backgrounds.
    # More rows are not falsely treated as independent training probes.
    byrole=collections.defaultdict(lambda:collections.defaultdict(list))
    for r in rows:byrole[r['target_role']][r['probe_rank']].append(r)
    preferences=[];audit=[]
    for role,rankmap in sorted(byrole.items()):
        arm=[]
        for rank,rr in sorted(rankmap.items()):
            q=rr[0]['query'];ok,reason=old.quality(q)
            stats={
                'role':role,'rank':rank,'query':q,'strategy':STRATEGIES[rank],
                'activation_rate':statistics.mean(float(x['target_routed_pos']) for x in rr),
                'mean_v135_win':statistics.mean(float(x['v135_win']) for x in rr),
                'mean_v134_win':statistics.mean(float(x['v134_win']) for x in rr),
                'mean_signed_v135_delta':statistics.mean(float(x['v135_delta']) for x in rr),
                'mean_signed_v134_delta':statistics.mean(float(x['v134_delta']) for x in rr),
                'mean_utility':statistics.mean(x['utility'] for x in rr),
                'valid':ok,'quality_reason':reason}
            audit.append(stats);arm.append(stats)
        eligible=[x for x in arm if x['valid'] and x['activation_rate']>=.8
                  and x['mean_v135_win']>=.5 and x['mean_v134_win']>=.5]
        if not eligible:continue
        winner=max(eligible,key=lambda x:x['mean_utility'])
        losers=[x for x in arm if x['valid'] and x['rank']!=winner['rank']
                and winner['mean_utility']-x['mean_utility']>=.07]
        if not losers:continue
        loser=min(losers,key=lambda x:x['mean_utility'])
        preferences.append({'role':role, 'chosen':winner['query'], 'rejected':loser['query'],
                            'chosen_strategy':winner['strategy'],'rejected_strategy':loser['strategy'],
                            'utility_gap':winner['mean_utility']-loser['mean_utility'],
                            'holdout_role':role in holdout,
                            'parent_adapter':str(root/'checkpoints/generator_activation_sft')})
    old.writel(d/'data/preferences.jsonl',preferences)
    old.writel(d/'data/arm_evidence.jsonl',audit)
    summary={
        'matched_rows':len(rows),'matched_groups':len(groups),'unique_queries':len({(r['target_role'],r['probe_rank']) for r in rows}),
        'train_roles':sorted(trainroles),'holdout_roles':sorted(holdout),
        'dpo_preferences_total':len(preferences),
        'dpo_preferences_train':sum(not x['holdout_role'] for x in preferences),
        'dpo_preferences_holdout':sum(x['holdout_role'] for x in preferences),
        'source_sha256':old.sha(MATCHED),
        'generator_parent_adapter':str(root/'checkpoints/generator_activation_sft'),
        'selector_source':'matched dev; v134/v135 signed evidence+target activation (training only)',
        'critical_limitation':'Previously inspected DEV; only 60 unique questions; not independent experiment.',
        'objective':'Target activation AND signed response-based PRESENT > ABSENT proxy; not real EIG',
        'generator_dpo_eligible':sum(not x['holdout_role'] for x in preferences)>=8 and len({x['role'] for x in preferences if not x['holdout_role']})>=8 and sum(x['holdout_role'] for x in preferences)>=2,
        'no_api':True,'no_old_checkpoint_writes':True,
    }
    old.writej(d/'reports/prepare.json',summary)
    print('UPGRADE_PREPARE',json.dumps(summary),flush=True)


def query_features(role,query,desc):
    q=old.clean(query);low=q.lower();w=re.findall(r"[a-z]+",low);ds=set(re.findall(r'[a-z]+',desc.lower()))
    return [min(len(w),220)/120.,min(len(q),1500)/1000.,
            min(len(re.findall(r'\d',q)),10)/10., min(len(re.findall(r'[,;:]',q)),10)/10.,
            float(any(x in low for x in ('table','schema','plan','code','example','report','checklist','test case','specification'))),
            float(any(x in low for x in ('must','ensure','require','avoid','limit','include','constraint','exactly'))),
            float(any(x in low for x in ('first','then','finally','after','before','sequence'))),
            float(q.endswith('?')),
            float(sum(x in ds for x in w))/max(1,len(w)),
            float(bool(old.META.search(q))),
            float(len(re.findall(r'\n',query))) / 8.,
            float(len(set(w)))/max(1,len(w))]


def train_selector(root):
    import numpy as np
    from scipy.sparse import hstack,csr_matrix
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import roc_auc_score
    import joblib
    d=udir(root);rows,groups=get_rows();cat=old.catalog();rolelist=np.array([r['target_role'] for r in rows])
    texts=[f"{r['query']}" for r in rows]
    numeric=np.array([query_features(r['target_role'],r['query'],cat['DESCRIPTIONS'][r['target_role']]) for r in rows],dtype=np.float64)
    def fit_predict(tri,tei):
        tf=TfidfVectorizer(ngram_range=(1,2),min_df=1,sublinear_tf=True,max_features=12000)
        Xtr=tf.fit_transform([texts[i] for i in tri]);Xte=tf.transform([texts[i] for i in tei])
        mu=numeric[tri].mean(axis=0);std=np.maximum(numeric[tri].std(axis=0),.1)
        Xtr=hstack([Xtr,csr_matrix((numeric[tri]-mu)/std)],format='csr')
        Xte=hstack([Xte,csr_matrix((numeric[tei]-mu)/std)],format='csr')
        indices={int(ix):p for p,ix in enumerate(tri)}
        pergroup=collections.defaultdict(list)
        for ix in tri:pergroup[(rows[ix]['target_role'],rows[ix]['pair_id'])].append(int(ix))
        xs=[];ys=[];sample_weights=[]
        for _,rr in pergroup.items():
            if len(rr)!=3:raise RuntimeError('Incomplete group in selector fold')
            for a in range(3):
                for b in range(a+1,3):
                    ia,ib=rr[a],rr[b];dif=rows[ia]['utility']-rows[ib]['utility']
                    if abs(dif)<.04:continue
                    v=Xtr[indices[ia]]-Xtr[indices[ib]]
                    xs.extend([v,-v]);ys.extend([int(dif>0),int(dif<0)])
                    sample_weights.extend([min(1.0,abs(dif)*2+.1)]*2)
        if len(ys)<50 or len(set(ys))<2:raise RuntimeError('Selector has insufficient preference comparisons')
        Xpairs=hstack([]) if False else __import__('scipy').sparse.vstack(xs,format='csr')
        model=LogisticRegression(C=.5,solver='liblinear',max_iter=1000,fit_intercept=False)
        model.fit(Xpairs,ys,sample_weight=sample_weights)
        return tf,model,mu,std,model.decision_function(Xte)

    gkf=GroupKFold(n_splits=5)
    preds=np.full(len(rows),np.nan)
    folds=[]
    for fi,(tr,te) in enumerate(gkf.split(texts,groups=rolelist),1):
        _,_,_,_,p=fit_predict(tr,te);preds[te]=p
        train_r=set(rolelist[tr]);test_r=set(rolelist[te]);assert not train_r&test_r
        folds.append({'fold':fi,'test_roles':sorted(test_r),'n_train':len(tr),'n_test':len(te)})
    if not np.isfinite(preds).all():raise RuntimeError('Incomplete selector out-of-role predictions')
    selected=[];random_expected=[];oracle=[];old_selector=[];rolemetrics=collections.defaultdict(list)
    for key,rr in groups.items():
        indices=[i for i,r in enumerate(rows) if (r['target_role'],r['pair_id'])==key]
        assert len(indices)==3
        result=[float(rows[i]['utility']) for i in indices]
        chosen=indices[int(np.argmax(preds[indices]))]
        selected.append(float(rows[chosen]['utility']))
        random_expected.append(float(np.mean(result)))
        oracle.append(float(np.max(result)))
        rolemetrics[key[0]].append(float(rows[chosen]['utility']-np.mean(result)))
    gain=np.array(selected)-np.array(random_expected)
    rng=np.random.default_rng(SEED)
    role_means=np.array([np.mean(vals) for vals in rolemetrics.values()])
    boots=np.array([np.mean(rng.choice(role_means,size=len(role_means),replace=True)) for _ in range(2000)])
    low,hi=np.percentile(boots,[2.5,97.5])
    verdict='LEARNED_ROLE_HELDOUT_PASS' if float(low)>0.0 and float(np.mean(gain))>.015 else 'FALLBACK_FIXED'
    tri=np.arange(len(rows),dtype=int)
    tf,model,mu,std,_=fit_predict(tri,tri)
    out=d/'checkpoints/selector';out.mkdir(parents=True,exist_ok=False)
    joblib.dump({'vectorizer':tf,'model':model,'numeric_mean':mu,'numeric_std':std,
                 'descriptions':cat['DESCRIPTIONS'],'verdict':verdict,
                 'data_source':'already-inspected matched dev','supports_history_training':False},out/'ranker.joblib')
    report={'verdict':verdict,'out_of_target_role_cv':True,'folds':folds,'n_matched_groups':len(groups),
            'mean_learned':float(np.mean(selected)),'mean_random':float(np.mean(random_expected)),
            'mean_oracle':float(np.mean(oracle)),'mean_gain':float(np.mean(gain)),
            'role_cluster_bootstrap_ci95_gain':[float(low),float(hi)],
            'dangers':'old sensor v134/v135 proxy; reuses explored dev; R2 public belief/history not in these training examples',
            'model_path':str(out/'ranker.joblib')}
    old.writej(d/'reports/selector_cv.json',report)
    print('UPGRADE_SELECTOR',json.dumps(report),flush=True)


def logprob(model,tokenizer,cat,role,q):
    import torch
    msgs=old.gen_messages(role,'CONCRETE ARTIFACT',cat['DESCRIPTIONS'],cat['CONFUSABLES'])
    # Both candidate completions share exactly the same conditioned prompt.
    prompt=tokenizer.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True)
    p=tokenizer(prompt,add_special_tokens=False)['input_ids']
    t=tokenizer(q,add_special_tokens=False)['input_ids'] + [tokenizer.eos_token_id]
    if len(p)+len(t)>768:raise RuntimeError('Preference sample exceeds 768 tokens')
    ids=torch.tensor([p+t],device='cuda');labels=torch.tensor([[-100]*len(p)+t],device='cuda')
    with torch.autocast(device_type='cuda',dtype=torch.bfloat16):
        return -model(input_ids=ids,attention_mask=torch.ones_like(ids),labels=labels).loss.float()


def train_generator(root):
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM,AutoTokenizer,BitsAndBytesConfig
    from peft import PeftModel,prepare_model_for_kbit_training
    d=udir(root);prep=old.readj(d/'reports/prepare.json')
    if not prep['generator_dpo_eligible']:
        old.writej(d/'reports/generator_dpo.json',{'status':'SKIPPED_INSUFFICIENT_RELIABLE_PREFERENCES',
                   'eligible':prep['dpo_preferences_train'],'reason':'Avoid overfitting tiny repeated matched pool'})
        print('UPGRADE_DPO_SKIPPED_INSUFFICIENT_PREFS',flush=True);return
    if not torch.cuda.is_available():raise RuntimeError('RTX 4090 GPU required')
    if not any('4090' in torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())):
        raise RuntimeError('DPO job expected RTX4090; refusing incompatible device')
    torch.manual_seed(SEED);random.seed(SEED)
    cat=old.catalog();pref=old.readl(d/'data/preferences.jsonl')
    train=[p for p in pref if not p['holdout_role']];holdout=[p for p in pref if p['holdout_role']]
    train.sort(key=lambda x:x['role'])
    parent=root/'checkpoints/generator_activation_sft'
    if not old.original_adapter(parent):raise RuntimeError('Missing V4 activation SFT adapter')
    tokenizer=AutoTokenizer.from_pretrained(parent,local_files_only=True)
    if tokenizer.pad_token_id is None:tokenizer.pad_token=tokenizer.eos_token
    qconfig=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16)
    base=AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct',quantization_config=qconfig,device_map={'':0},dtype=torch.bfloat16,local_files_only=True)
    base.config.use_cache=False
    base=prepare_model_for_kbit_training(base,use_gradient_checkpointing=True,gradient_checkpointing_kwargs={'use_reentrant':False})
    model=PeftModel.from_pretrained(base,str(parent),is_trainable=True)
    params=[p for p in model.parameters() if p.requires_grad]
    if not params:raise RuntimeError('No trainable generator LoRA params')
    print('DPO_PREFS',len(train),'HOLDOUT',len(holdout),flush=True)
    # Reference is the exact warm-start adapter, evaluated and cached BEFORE any policy update.
    model.eval();reference=[]
    for j,r in enumerate(train+holdout):
        with torch.no_grad():
            rw=logprob(model,tokenizer,cat,r['role'],r['chosen']).item()
            rl=logprob(model,tokenizer,cat,r['role'],r['rejected']).item()
        reference.append((rw,rl))
    reftrain=reference[:len(train)];refhold=reference[len(train):]
    model.train();rng=random.Random(SEED);order=list(range(len(train)));rng.shuffle(order)
    lr=2e-6;beta=.15;optimizer=torch.optim.AdamW(params,lr=lr,weight_decay=0.0)
    optimizer.zero_grad(set_to_none=True);losses=[]
    # One conservative pass; never learn activation based merely on role-name tokens.
    for step,j in enumerate(order,1):
        r=train[j];w=logprob(model,tokenizer,cat,r['role'],r['chosen']);l=logprob(model,tokenizer,cat,r['role'],r['rejected'])
        refw,refl=reftrain[j]
        z=beta*((w-l)-(refw-refl))
        loss= -F.logsigmoid(z)
        if not torch.isfinite(loss):raise RuntimeError('Non-finite DPO loss')
        (loss/4).backward();losses.append(float(loss.detach().cpu()))
        if step%4==0 or step==len(order):
            torch.nn.utils.clip_grad_norm_(params,.5)
            optimizer.step();optimizer.zero_grad(set_to_none=True)
        if step%5==0:print('DPO_STEP',step,len(order),'loss',round(statistics.mean(losses[-5:]),4),flush=True)
    model.eval()
    margins=[]
    with torch.no_grad():
        for j,r in enumerate(holdout):
            wp=logprob(model,tokenizer,cat,r['role'],r['chosen']).item()
            lp=logprob(model,tokenizer,cat,r['role'],r['rejected']).item()
            margins.append({'role':r['role'],'before':refhold[j][0]-refhold[j][1], 'after':wp-lp})
    before=statistics.mean([x['before'] for x in margins]) if margins else None
    after=statistics.mean([x['after'] for x in margins]) if margins else None
    # Not an empirical MAS test: just a conservative preference-overfitting check.
    keep=bool(margins) and after>=before-.05
    if keep:
        out=d/'checkpoints/generator_discrimination_dpo';out.mkdir(parents=True,exist_ok=False)
        model.save_pretrained(out);tokenizer.save_pretrained(out)
    else:out=parent
    report={'status':'TRAINED_PILOT' if keep else 'FALLBACK_PARENT',
        'parent':str(parent),'selected_adapter':str(out),'num_train_unique_role_preferences':len(train),
        'num_heldout_role_preferences':len(holdout),'reference_margin_holdout':before,
        'updated_margin_holdout':after,'holdout_margins':margins,'mean_dpo_loss':statistics.mean(losses),
        'learning_rate':lr,'epochs':1,'beta':beta,'objective':'DPO on routing-verified, signed PRESENT>ABSENT old-sensor proxy',
        'not_proven':'No new probes executed on fresh matched MAS; no GRPO and no real EIG'}
    old.writej(d/'reports/generator_dpo.json',report)
    print('UPGRADE_DPO_RESULT',json.dumps({k:v for k,v in report.items() if k!='holdout_margins'}),flush=True)


def report(root):
    d=udir(root)
    results={}
    for k in ('prepare','selector_cv','generator_dpo'):
        p=d/'reports'/f'{k}.json'
        results[k]=old.readj(p) if p.exists() else {'status':'MISSING'}
    old.writej(d/'reports/final.json',results)
    print('UPGRADE_FINAL_REPORT',d/'reports/final.json',flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['prepare','selector','generator','report'])
    parser.add_argument('--root',required=True)
    args=parser.parse_args();root=getroot(args.root)
    {'prepare':prepare,'selector':train_selector,'generator':train_generator,'report':report}[args.stage](root)

if __name__=='__main__':main()
