#!/usr/bin/env python3
"""Matched-response QLoRA continuation from V4 epoch02, never fresh weights.
Paired IDENTICAL target-role query against POS/NEG MAS; contrastive rank loss.
Holdout both unseen probe strategy indices and unseen background pair indices.
"""
from __future__ import annotations
import argparse, collections, json, math, random, statistics
from pathlib import Path

def main(root):
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForSequenceClassification,AutoTokenizer,BitsAndBytesConfig
    from peft import PeftConfig,PeftModel,prepare_model_for_kbit_training
    from analysis.role_direct_within_v4 import v46_night as n
    from analysis.role_direct_within_v4.pipeline import catalog,make_sensor_text
    root=Path(root)
    if '4090' not in torch.cuda.get_device_name(0):raise RuntimeError('4090 required')
    torch.manual_seed(20261010);random.seed(20261010)
    cs=n.jlines(root/'train/candidates_train_round_01.jsonl')
    rs=n.jlines(root/'train/responses_train_round_01.jsonl')
    cmap={(x['session_id'],n.norm(x['query'])):x for x in cs}
    paired=collections.defaultdict(dict)
    for x in rs:
        key=(x['session_id'],n.norm(x['query']))
        c=cmap[key];pkey=(c['target_role'],c['pair_id'],n.norm(c['query']))
        if c['condition'] in paired[pkey]:raise RuntimeError('Duplicate matched condition')
        paired[pkey][c['condition']]={'role':c['target_role'],'query':c['query'],
                        'response':x['final_output'],'strategy_index':int(c['candidate_index']),
                        'pair_id':c['pair_id']}
    pairs=[]
    for group in paired.values():
        if set(group)!={'TARGET_PRESENT','TARGET_ABSENT'}:raise RuntimeError('Missing matched response')
        pairs.append((group['TARGET_PRESENT'],group['TARGET_ABSENT']))
    if len(pairs)!=480:raise RuntimeError('Expected 480 paired observations')
    # For validation, BOTH query arm and background pair must be withheld.
    train=[];val=[]
    for pair in pairs:
        x=pair[0];index=int(x['strategy_index']);which=int(x['pair_id'].rsplit('_',1)[-1])
        if index in (0,1,2,3,6,7) and which<=2:train.append(pair)
        if index in (4,5) and which==3:val.append(pair)
    if len(train)!=240 or len(val)!=40:raise RuntimeError(f'Split mismatch {len(train)} {len(val)}')
    dsc=catalog()['DESCRIPTIONS']
    parent=Path(n.load(root/'config.json')['source_sensor_adapter'])
    conf=PeftConfig.from_pretrained(parent,local_files_only=True)
    tok=AutoTokenizer.from_pretrained(parent,local_files_only=True)
    if tok.pad_token_id is None:tok.pad_token=tok.eos_token
    tok.padding_side='right';tok.truncation_side='right'
    qcfg=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16)
    base=AutoModelForSequenceClassification.from_pretrained(conf.base_model_name_or_path,num_labels=1,
         quantization_config=qcfg,device_map={'':0},dtype=torch.bfloat16,local_files_only=True)
    base.config.pad_token_id=tok.pad_token_id;base.config.use_cache=False
    base=prepare_model_for_kbit_training(base,use_gradient_checkpointing=True,
                              gradient_checkpointing_kwargs={'use_reentrant':False})
    model=PeftModel.from_pretrained(base,str(parent),is_trainable=True)
    params=[p for p in model.parameters() if p.requires_grad]
    if not params:raise RuntimeError('No trainable Sensor LoRA')
    def batch(pair):
        return tok([make_sensor_text(x['role'],x['query'],x['response'],dsc) for x in pair],
                    padding=True,truncation=True,max_length=576,return_tensors='pt').to('cuda')
    def evalpair(data):
        model.eval();wins=[];gaps=[]
        with torch.no_grad():
            for p in data:
                with torch.autocast(device_type='cuda',dtype=torch.bfloat16):
                    zz=model(**batch(p)).logits.float().flatten().cpu().tolist()
                gap=float(zz[0]-zz[1]);gaps.append(gap);wins.append(float(gap>0)+.5*float(gap==0))
        return {'pair_win':statistics.mean(wins),'mean_margin':statistics.mean(gaps),'n':len(gaps)}
    baseline=evalpair(val);print('MATCHED_SENSOR_BASELINE',baseline,flush=True)
    optimizer=torch.optim.AdamW(params,lr=8e-6,weight_decay=.01)
    best=None;history=[]
    for epoch in range(1,5):
        model.train();rng=random.Random(20261010+epoch);order=train.copy();rng.shuffle(order)
        optimizer.zero_grad(set_to_none=True);losses=[]
        for step,p in enumerate(order,1):
            with torch.autocast(device_type='cuda',dtype=torch.bfloat16):
                zz=model(**batch(p)).logits.float().flatten()
                loss=F.softplus(-(zz[0]-zz[1])) + .12*(F.softplus(-zz[0])+F.softplus(zz[1]))
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite Sensor loss')
            (loss/4).backward();losses.append(float(loss.detach().cpu()))
            if step%4==0 or step==len(order):
                torch.nn.utils.clip_grad_norm_(params,.75)
                optimizer.step();optimizer.zero_grad(set_to_none=True)
            if step%40==0:print('MATCHED_SENSOR_STEP',epoch,step,len(order),'loss',round(statistics.mean(losses[-40:]),4),flush=True)
        valmet=evalpair(val);valmet.update({'epoch':epoch,'train_loss':statistics.mean(losses)})
        history.append(valmet);print('MATCHED_SENSOR_EPOCH',valmet,flush=True)
        if best is None or (valmet['pair_win'],valmet['mean_margin'])>(best['pair_win'],best['mean_margin']):
            best=valmet.copy()
            dest=root/'train/sensor_adapter'
            # Save only BEST checkpoint; do not overwrite parent adapters.
            dest.mkdir(parents=True,exist_ok=True)
            model.save_pretrained(dest);tok.save_pretrained(dest)
    pass_gate=(best['pair_win']>=.56 and best['pair_win']>=baseline['pair_win']+.015)
    n.dump(root/'train/sensor_train_report.json',{
       'status':'PASS' if pass_gate else 'FAILED_VALIDATION',
       'baseline':baseline,'best_validation':best,'epochs':history,
       'train_pairs':len(train),'validation_pairs':len(val),
       'validation_is_new_probe_index_and_new_background':True,
       'objective':'matched same-query PRESENT/ABSENT pairwise contrastive + weak BCE',
       'caveat':'Still proxy matched win, not whole-MAS Within-AUC; previously inspected target family'})
    if not pass_gate:raise RuntimeError('Sensor validation not improved enough; do NOT run MAS5')
    print('V46_SENSOR_VALIDATION_PASS',best,flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);main(Path(ap.parse_args().root))
