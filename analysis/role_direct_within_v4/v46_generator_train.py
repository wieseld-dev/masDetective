#!/usr/bin/env python3
"""MAS5 V4.6 generator: (1) TRUE full-question HMHY SFT from V4 parent,
(2) SAME-CONDITIONING REFERENCE-CORRECT DPO from matched training evidence.

DPO never sees future MAS5 truth, uses role/family prompt and fresh train-only
chosen/rejected pair decisions; frozen reference is cached before any update.
"""
from __future__ import annotations
import argparse, math, random, statistics
from pathlib import Path

P=Path('/home/wieseld/mas_rl_detective')


def main(root,stage):
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM,AutoTokenizer,BitsAndBytesConfig
    from peft import PeftModel,prepare_model_for_kbit_training
    from analysis.role_direct_within_v4 import pipeline as old,v46_generator as gen,v46_night as n
    root=Path(root)
    if not torch.cuda.is_available() or '4090' not in torch.cuda.get_device_name(0):
        raise RuntimeError('RTX4090 required for 4-bit QLoRA')
    torch.manual_seed(20261018)
    random.seed(20261018)
    cfg=n.load(root/'config.json')
    cat=old.catalog()
    families={'ELICITATION_INTAKE':'intake', 'ELICITATION_AMBIGUITY':'ambiguity',
              'ELICITATION_VALIDATION':'validation'}
    if stage=='bootstrap':
        src=n.jlines(cfg['source_v43_probe_bank'])
        rows=[]
        for x in src:
            family=families.get(x.get('strategy'))
            if not family or x.get('probe_family')!='help_me_help_you':continue
            valid,why=gen.hmhy_quality(x['query'])
            if not valid:continue
            rows.append({'role':x['target_role'],'family':family,'completion':x['query']})
        if len(rows)<35 or len({x['role'] for x in rows})<15:
            raise RuntimeError(f'HMHY bootstrap source insufficient: {len(rows)} rows, {len(set(x["role"] for x in rows))} roles')
        parent=Path(cfg['source_generator_adapter'])
        destination=root/'train/generator_bootstrap_adapter'
        epochs=3
        lr=5e-6
    else:
        prefs=n.jlines(root/'train/generator_preferences.jsonl')
        if len(prefs)<12 or len({x['role'] for x in prefs})<8:
            raise RuntimeError('Not enough matched-evidence DPO comparisons')
        if any(x['family'] not in ('intake','ambiguity','validation') for x in prefs):
            raise RuntimeError('Unknown HMHY family')
        if any(x['chosen']==x['rejected'] for x in prefs):
            raise RuntimeError('Identical chosen/rejected')
        parent=root/'train/generator_bootstrap_adapter'
        destination=root/'train/generator_adapter'
        epochs=3
        lr=4e-6
    if destination.exists():raise RuntimeError('No overwrite: '+str(destination))
    tok=AutoTokenizer.from_pretrained(parent,local_files_only=True)
    if tok.pad_token_id is None:tok.pad_token=tok.eos_token
    quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16)
    base=AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct',
        quantization_config=quant,device_map={'':0},dtype=torch.bfloat16,local_files_only=True)
    base.config.use_cache=False
    base=prepare_model_for_kbit_training(base,use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant':False})
    model=PeftModel.from_pretrained(base,str(parent),is_trainable=True)
    trainables=[p for p in model.parameters() if p.requires_grad]
    if not trainables:raise RuntimeError('No trainable LoRA')
    optimizer=torch.optim.AdamW(trainables,lr=lr,weight_decay=0)

    def toks(role,family,completion):
        prompt=tok.apply_chat_template(gen.hmhy_messages(role,family,cat,None),
                                       tokenize=False,add_generation_prompt=True)
        p=tok(prompt,add_special_tokens=False)['input_ids']
        t=tok(completion,add_special_tokens=False)['input_ids']+[tok.eos_token_id]
        if len(p)+len(t)>900:raise RuntimeError('Long example exceeds 900 tokens')
        ids=torch.tensor([p+t],dtype=torch.long,device='cuda')
        labels=torch.tensor([[-100]*len(p)+t],dtype=torch.long,device='cuda')
        return ids,labels

    def sequence_logp(role,family,completion):
        ids,labels=toks(role,family,completion)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            logits=model(input_ids=ids,attention_mask=torch.ones_like(ids)).logits
        # Completion-only token mean log probability; average reduces length bias.
        targets=labels[:,1:];mask=targets.ne(-100)
        log_probs=F.log_softmax(logits[:,:-1,:].float(),dim=-1)
        token_lp=log_probs.gather(-1,targets.clamp(min=0).unsqueeze(-1)).squeeze(-1)
        return (token_lp*mask).sum()/mask.sum().clamp(min=1)

    if stage=='bootstrap':
        # V4.6.1 FIX: original V4.6 withheld one example per *each* role,
        # leaving fewer than 30 training questions in a sparse V4.3 bank.
        # Reserve exactly five questions from five different roles, and keep
        # all other qualified HMHY examples for continued SFT. This is a small
        # *wording* validation set, NOT a claim of fingerprint generalization.
        rng=random.Random(46);byrole={}
        for x in rows:byrole.setdefault(x['role'],[]).append(x)
        train=[];heldout_candidates=[]
        for role in sorted(byrole):
            xx=byrole[role]
            rng.shuffle(xx)
            if len(xx)>=2:
                heldout_candidates.append(xx[0]);train.extend(xx[1:])
            else:
                train.extend(xx)
        rng.shuffle(heldout_candidates)
        val=heldout_candidates[:5]
        train.extend(heldout_candidates[5:])
        print('MAS5_HMHY_BOOTSTRAP_SPLIT',
              {'qualified':len(rows),'roles':len(byrole),
               'roles_with_multiple_examples':len(heldout_candidates),
               'train':len(train),'validation':len(val),
               'validation_roles':sorted(x['role'] for x in val)},flush=True)
        if len(train)<30 or len(val)<5:
            raise RuntimeError(f'SFT split too small: source={len(rows)}, '
                               f'train={len(train)}, val={len(val)}, '
                               f'roles={len(byrole)}, roles_with_multiple={len(heldout_candidates)}')
        def nll(xx):
            model.eval()
            with torch.no_grad():return statistics.mean(float(-sequence_logp(x['role'],x['family'],x['completion'])) for x in xx)
        start=nll(val);hist=[]
        for epoch in range(1,epochs+1):
            model.train();rng.shuffle(train);optimizer.zero_grad(set_to_none=True);losses=[]
            for i,x in enumerate(train,1):
                loss=-sequence_logp(x['role'],x['family'],x['completion'])
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite SFT loss')
                (loss/4).backward();losses.append(float(loss.detach()))
                if i%4==0 or i==len(train):
                    torch.nn.utils.clip_grad_norm_(trainables,.7)
                    optimizer.step();optimizer.zero_grad(set_to_none=True)
                if i%12==0:print('MAS5_SFT',epoch,i,len(train),round(statistics.mean(losses[-12:]),4),flush=True)
            post=nll(val)
            hist.append({'epoch':epoch,'train_nll':statistics.mean(losses),'heldout_nll':post})
            print('MAS5_SFT_EPOCH',hist[-1],flush=True)
        improved=math.isfinite(hist[-1]['heldout_nll']) and hist[-1]['heldout_nll']<=start+.12
        report={'status':'PASS' if improved else 'FAILED_HMHY_SFT','method':'Continued QLoRA SFT on COMPLETE HMHY questions, not wrapped scenarios',
            'source_adapter':str(parent),'train_examples':len(train),'validation_examples':len(val),
            'initial_heldout_nll':start,'history':hist,
            'limitation':'Teaching HMHY wording is not proof of fingerprint discrimination'}
        n.dump(root/'train/generator_bootstrap_report.json',report)
        if not improved:raise RuntimeError('HMHY SFT heldout gate failed')
        model.save_pretrained(destination);tok.save_pretrained(destination)
        print('MAS5_HMHY_SFT_BOOTSTRAP_PASS',destination,flush=True)
        return

    # Only pairs with exactly identical generation conditioning (role + family)
    # are compared. Split entire ROLES for independent DPO holdout.
    rng=random.Random(4606)
    prefroles=sorted({x['role'] for x in prefs});rng.shuffle(prefroles)
    valroles=set(prefroles[:max(2,len(prefroles)//5)])
    train=[x for x in prefs if x['role'] not in valroles]
    val=[x for x in prefs if x['role'] in valroles]
    if len(train)<8 or len(val)<2:raise RuntimeError(f'DPO split too small {len(train)}/{len(val)}')
    # Frozen reference: cached log probabilities from PRE-DPO SFT adapter.
    reference={}
    model.eval()
    with torch.no_grad():
        for i,x in enumerate(prefs,1):
            reference[(x['role'],x['family'],x['chosen'])]=float(sequence_logp(x['role'],x['family'],x['chosen']))
            reference[(x['role'],x['family'],x['rejected'])]=float(sequence_logp(x['role'],x['family'],x['rejected']))
            if i%10==0:print('MAS5_DPO_REFERENCE',i,len(prefs),flush=True)
    def margins(items):
        model.eval();deltas=[]
        with torch.no_grad():
            for x in items:
                lp_c=sequence_logp(x['role'],x['family'],x['chosen'])
                lp_r=sequence_logp(x['role'],x['family'],x['rejected'])
                ref_c=reference[(x['role'],x['family'],x['chosen'])]
                ref_r=reference[(x['role'],x['family'],x['rejected'])]
                deltas.append(float((lp_c-lp_r).detach())-(ref_c-ref_r))
        return statistics.mean(deltas),sum(d>0 for d in deltas)/len(deltas)
    initial_val=margins(val)
    history=[];beta=2.0
    for epoch in range(1,epochs+1):
        model.train();rng.shuffle(train);optimizer.zero_grad(set_to_none=True)
        losses=[]
        for i,x in enumerate(train,1):
            log_c=sequence_logp(x['role'],x['family'],x['chosen'])
            log_r=sequence_logp(x['role'],x['family'],x['rejected'])
            ref_diff=reference[(x['role'],x['family'],x['chosen'])]-reference[(x['role'],x['family'],x['rejected'])]
            loss=-F.logsigmoid(beta*((log_c-log_r)-ref_diff))
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite DPO loss')
            (loss/2).backward();losses.append(float(loss.detach()))
            if i%2==0 or i==len(train):
                torch.nn.utils.clip_grad_norm_(trainables,.7)
                optimizer.step();optimizer.zero_grad(set_to_none=True)
            if i%10==0:print('MAS5_DPO',epoch,i,len(train),round(statistics.mean(losses[-10:]),4),flush=True)
        heldout=margins(val)
        entry={'epoch':epoch,'train_dpo_loss':statistics.mean(losses),
               'heldout_preference_logprob_shift':heldout[0],
               'heldout_relative_preference_win':heldout[1]}
        history.append(entry)
        print('MAS5_DPO_EPOCH',entry,flush=True)
        if epoch==1 or entry['heldout_preference_logprob_shift']>best['heldout_preference_logprob_shift']:
            best=entry.copy()
            model.save_pretrained(destination);tok.save_pretrained(destination)
    # This is an operational generalization sanity check, not external MAS accuracy.
    passed=best['heldout_preference_logprob_shift']>.001 and math.isfinite(best['heldout_preference_logprob_shift'])
    report={'status':'PASS' if passed else 'FAILED_DPO_VALIDATION',
       'method':'QLoRA continued, frozen-reference length-normalized DPO (same role/family input)',
       'train_preferences':len(train),'heldout_preferences':len(val),
       'heldout_roles':sorted(valroles),'bootstrap_adapter':str(parent),
       'reference_cached_pretraining':True,'beta':beta,'history':history,'selected_epoch':best,
       'reward':'Observed present/absent target routing + frozen V4.2 proxy response scorer on 3 matched backgrounds',
       'limitation':'Frozen V4.2 response scorer did not beat TFIDF dev; cannot claim true role information gain or MAS5 gain'}
    n.dump(root/'train/generator_train_report.json',report)
    if not passed:raise RuntimeError('DPO held-out preference sanity gate failed; MAS5 will NOT run')
    print('MAS5_GENERATOR_DPO_PASS',len(train),len(val),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',required=True)
    ap.add_argument('--stage',choices=['bootstrap','dpo'],required=True)
    a=ap.parse_args();main(Path(a.root),a.stage)
