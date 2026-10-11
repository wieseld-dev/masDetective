#!/usr/bin/env python3
"""Conservative positive-evidence SFT continuation of existing V4 LoRA.
Direct tasks and Help-Me-Help-You are trained under MATCHING inference prompts.
No GRPO claim; no training on dev; only actual matched TRAIN probes accepted.
"""
from __future__ import annotations
import argparse, collections, math, random, statistics
from pathlib import Path

def main(root):
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer,BitsAndBytesConfig
    from peft import PeftModel,prepare_model_for_kbit_training
    from analysis.role_direct_within_v4 import pipeline as old,v44_generator as gen,v44_night as n
    root=Path(root)
    if '4090' not in torch.cuda.get_device_name(0):raise RuntimeError('4090 GPU required')
    data=n.jlines(root/'train/generator_positive_sft.jsonl')
    if len(data)<18 or len(set(r['role'] for r in data))<8:raise RuntimeError('Insufficient verified good query pool')
    cat=old.catalog();rng=random.Random(20261011)
    data.sort(key=lambda x:(x['role'],x['strategy_index']))
    # Hold out entire roles for an honest cross-role likelihood sanity check.
    roles=sorted(set(x['role'] for x in data));rng.shuffle(roles)
    hold_roles=set(roles[:max(2,len(roles)//5)])
    train=[x for x in data if x['role'] not in hold_roles]
    hold=[x for x in data if x['role'] in hold_roles]
    if len(train)<14 or len(hold)<2:raise RuntimeError('Insufficient holdout')
    # Oversample verified HMHY cases to >=75% of training updates; role holdout stays isolated.
    hmhy=[r for r in train if r['probe_family']=='help_me_help_you']
    direct=[r for r in train if r['probe_family']!='help_me_help_you']
    if len(hmhy)<6:raise RuntimeError('Insufficient reliable HMHY positives for MAS5 training')
    if direct:
        train=hmhy*max(1,math.ceil(3*len(direct)/len(hmhy)))+direct
    else:
        train=hmhy
    assert sum(r['probe_family']=='help_me_help_you' for r in train)/len(train)>=.75
    parent=Path(n.load(root/'config.json')['source_generator_adapter'])
    tok=AutoTokenizer.from_pretrained(parent,local_files_only=True)
    if tok.pad_token_id is None:tok.pad_token=tok.eos_token
    qcfg=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16)
    base=AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct',quantization_config=qcfg,
                                               device_map={'':0},dtype=torch.bfloat16,local_files_only=True)
    base.config.use_cache=False
    base=prepare_model_for_kbit_training(base,use_gradient_checkpointing=True,
                                         gradient_checkpointing_kwargs={'use_reentrant':False})
    model=PeftModel.from_pretrained(base,str(parent),is_trainable=True)
    params=[x for x in model.parameters() if x.requires_grad]
    if not params:raise RuntimeError('No trainable generator LoRA')
    def example(r):
        role=r['role'];strategy=r['strategy'];full=r['query']
        if r['probe_family']=='help_me_help_you':
            mapping={'ELICITATION_INTAKE':'intake','ELICITATION_AMBIGUITY':'ambiguity',
                     'ELICITATION_VALIDATION':'validation'}
            fam=mapping[strategy]
            msgs=gen.scenario_messages(role,fam,cat,None)
            head="I'm working on this situation: "
            if not full.startswith(head) or '. Before ' not in full:
                raise RuntimeError('Invalid HMHY formatting, refuse training: '+full[:70])
            completion=full[len(head):].split('. Before ',1)[0]+'.'
        else:
            msgs=old.gen_messages(role,strategy,cat['DESCRIPTIONS'],cat['CONFUSABLES'])
            msgs[0]['content']+=(' The task must be realistically solvable and demand a CHECKABLE specialist deliverable. '
                'When that specialist is present it should naturally be routed to their capability. '
                'Choose input details that leave room for observable output differences versus a system lacking that specialist. '
                'Neither mention the target role nor ask the target to identify its architecture.')
            strategy_ix=int(r['strategy_index'])
            msgs[1]['content']+=('\nSTRATEGY EXPLANATION: '+cat['STRATEGIES'][strategy_ix][1]+'\n'+gen.state_context(None,role)+
                '\nDo not repeat previous queries. Provide the user task only.')
            completion=full
        pref=tok.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True)
        p=tok(pref,add_special_tokens=False)['input_ids'];t=tok(completion,add_special_tokens=False)['input_ids']+[tok.eos_token_id]
        if len(p)+len(t)>800:raise RuntimeError('Training example exceeds safe token limit')
        ids=torch.tensor([p+t],dtype=torch.long,device='cuda')
        labels=torch.tensor([[-100]*len(p)+t],dtype=torch.long,device='cuda')
        return ids,labels
    def lossfn(r):
        ids,labels=example(r)
        with torch.autocast(device_type='cuda',dtype=torch.bfloat16):
            return model(input_ids=ids,attention_mask=torch.ones_like(ids),labels=labels).loss.float()
    def ev(data):
        model.eval()
        with torch.no_grad():out=[float(lossfn(r).detach().cpu()) for r in data]
        return statistics.mean(out)
    before=ev(hold);print('HMHY_GEN_HOLDOUT_BEFORE',round(before,5),flush=True)
    opt=torch.optim.AdamW(params,lr=5e-6,weight_decay=0.0)
    history=[]
    for epoch in range(1,3):
        rng.shuffle(train);model.train();opt.zero_grad(set_to_none=True);losses=[]
        for i,r in enumerate(train,1):
            loss=lossfn(r)
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite Generator loss')
            (loss/4).backward();losses.append(float(loss.detach().cpu()))
            if i%4==0 or i==len(train):
                torch.nn.utils.clip_grad_norm_(params,0.5);opt.step();opt.zero_grad(set_to_none=True)
            if i%20==0:print('HMHY_GEN_SFT_STEP',epoch,i,len(train),'loss',round(statistics.mean(losses[-20:]),4),flush=True)
        model.eval();after=ev(hold)
        history.append({'epoch':epoch,'train_loss':statistics.mean(losses),'holdout_nll':after})
        print('HMHY_GEN_SFT_EPOCH',history[-1],flush=True)
    after=history[-1]['holdout_nll']
    good=math.isfinite(after) and after<=before+.08
    if good:
        dest=root/'train/generator_adapter'
        if dest.exists():raise RuntimeError('Generator destination exists')
        model.save_pretrained(dest);tok.save_pretrained(dest)
    report={'status':'PASS' if good else 'FAILED_VALIDATION','base_adapter':str(parent),
       'method':'SFT on routing-verified and matched-response-positive probes, INCLUDING HMHY scenarios',
       'not_grpo_or_dpo':True,'positive_train_rows':len(train),'heldout_rows':len(hold),
       'positive_HMHY_rows':sum(x['probe_family']=='help_me_help_you' for x in data),
       'heldout_nll_before':before,'heldout_nll_after':after,'epochs':history,
       'limitations':'NLL is sanity check; real discrimination evaluated ONLY in subsequent Simple5'}
    n.dump(root/'train/generator_train_report.json',report)
    if not good:raise RuntimeError('Generator heldout NLL degraded; do NOT run Simple5')
    print('V44_GENERATOR_TRAIN_PASS',report['positive_train_rows'],flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);main(Path(ap.parse_args().root))
