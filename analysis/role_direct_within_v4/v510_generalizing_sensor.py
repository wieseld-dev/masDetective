#!/usr/bin/env python3
"""V5.10: role-composition generalization Sensor, audited held-out MAS configurations.

V137 TRAIN and optionally V56 MAS5: labels used only in optimizer, NOT model text.
V137 DEV split per distinct *composition* into validation + test (no train rows).
Checkpoint selection ONLY on validation compositions. Test is evaluated ONCE after
training in a separate scoring stage; it is not a genuinely prospective unseen test
because older V5.7 reports already exposed V137 Dev aggregate results.
"""
from __future__ import annotations
import argparse, collections, hashlib, json, math, os, random, statistics, time
from pathlib import Path

from analysis.role_direct_within_v4 import v57_sensor_night as v57
P = v57.P
OLD = v57.DEFAULT_PARENT
SEED = 5102026
BASE = P / 'artifacts/role_direct_within_v4/sensor_v510'


def dump(path, data): v57.atomic_json(path, data)
def getrows(path): return v57.lines(path)
def digest(text): return hashlib.sha256(str(text).encode('utf8')).hexdigest()
def tkey(r):
    return digest(json.dumps([r['role'],r['query'],r['answer'],r.get('query2'),r.get('answer2')],ensure_ascii=False))

def complete_systems(rows, roles):
    by=collections.defaultdict(lambda:collections.defaultdict(list))
    for r in rows:
        if r['role'] in roles and r['label'] in (0,1):
            by[r['system_id']][r['role']].append(r)
    systems={}; incomplete=[]
    for sid,group in sorted(by.items()):
        if set(group)!=set(roles):
            incomplete.append({'system_id':sid,'roles':len(group)})
            continue
        labels={role:{x['label'] for x in group[role]} for role in roles}
        if any(len(z)!=1 for z in labels.values()):
            incomplete.append({'system_id':sid,'inconsistent_labels':True});continue
        selected={role:group[role][0] for role in roles}
        positives={r for r in roles if selected[r]['label']==1}
        if len(positives)!=3:
            incomplete.append({'system_id':sid,'positives':len(positives)});continue
        systems[sid]=(selected,positives)
    return systems,incomplete

def partition(dev, roles):
    systems, invalid=complete_systems(dev,roles)
    # Split distinct role compositions, not only session IDs. Duplicate composition
    # must NEVER straddle validation/test.
    bycomp=collections.defaultdict(list)
    for sid,(_,pos) in systems.items():bycomp[tuple(sorted(pos))].append(sid)
    comps=list(sorted(bycomp));random.Random(SEED).shuffle(comps)
    if len(comps)<6 or len(systems)<8:
        raise RuntimeError(f'Need >=6 distinct compositions and >=8 complete V137 DEV systems. Got {len(comps)} compositions, {len(systems)} systems; no GPU jobs')
    half=max(2,len(comps)//2)
    val_comps=set(comps[:half]);test_comps=set(comps[half:])
    val={sid:sys for sid,sys in systems.items() if tuple(sorted(sys[1])) in val_comps}
    test={sid:sys for sid,sys in systems.items() if tuple(sorted(sys[1])) in test_comps}
    if len(val)<2 or len(test)<2:raise RuntimeError('Dev composition split too unbalanced; no GPU jobs')
    return val,test,{'dev_complete_systems':len(systems),'dev_invalid_systems':invalid,
                     'val_system_ids':sorted(val),'test_system_ids':sorted(test),
                     'val_compositions':[list(x) for x in sorted(val_comps)],
                     'test_compositions':[list(x) for x in sorted(test_comps)]}

def v56source():
    s=os.environ.get('V510_V56','')
    if s:
        root=Path(s).resolve()
        if not (root/'responses_public_dev_round_02.jsonl').is_file():raise RuntimeError('V510_V56 missing R2 responses')
        return root
    base=P/'artifacts/role_direct_within_v4/nonadaptive_v56'
    rs=sorted((r for r in base.glob('mas5_*') if (r/'responses_public_dev_round_02.jsonl').is_file()),reverse=True)
    if not rs:raise RuntimeError('No completed V56 source; set V510_V56')
    return rs[0]

def obtain_sources():
    roles,descs=v57.roles_and_descriptions()
    historical,dev=v57.historical_v137()
    historical=[r for r in historical if r['role'] in roles]
    dev=[r for r in dev if r['role'] in roles]
    val,test,split=partition(dev,roles)
    devcomps={tuple(sorted(sys[1])) for sys in {**val,**test}.values()}
    # Drop any same-composition historical TRAIN systems, even if different system_id.
    traincomplete,traininvalid=complete_systems(historical,roles)
    same_comp_systems={sid for sid,(_,c) in traincomplete.items() if tuple(sorted(c)) in devcomps}
    kept=[r for r in historical if r['system_id'] not in same_comp_systems]
    # Remove identical role/query/answer observations reused across train and DEV.
    seen_dev={tkey(r) for group in (val,test) for selected,_ in group.values() for r in selected.values()}
    exact_overlap=[r for r in kept if tkey(r) in seen_dev]
    kept=[r for r in kept if tkey(r) not in seen_dev]
    if len(kept)<300:raise RuntimeError(f'Only {len(kept)} V137 TRAIN rows after composition/overlap filtering; do not launch')
    vv,config=v57.v56_records(v56source(),True)
    vv=[r for r in vv if tuple(sorted(config[r['system_id']])) not in devcomps and tkey(r) not in seen_dev]
    # Verifiable within-V137 TRAIN same-query, same-role comparisons across
    # different compositions. DO NOT use V44/V54 by default because their full
    # role-compositions are not verifiable by this script and might overlap DEV.
    grouped=collections.defaultdict(lambda:{0:[],1:[]})
    for r in kept:
        grouped[(r['role'],' '.join(r['query'].lower().split()))][r['label']].append(r)
    matched=[]
    for group in grouped.values():
        for pos in group[1]:
            for neg in group[0]:
                if pos['system_id']!=neg['system_id']:
                    matched.append((pos,neg))
    mroot=None
    # Within-MAS cross-role discrimination, in V137 TRAIN (different topics), and V56.
    def within(rows):
        groups=collections.defaultdict(lambda:{0:[],1:[]})
        for r in rows:groups[(r['system_id'],r.get('kind','single'),r.get('round',0))][r['label']].append(r)
        result=[]
        for g in groups.values():
            for pos in g[1]:
                for neg in g[0]:result.append((pos,neg))
        return result
    histwithin=within(kept)
    maswithin=within(vv)
    # Hard negatives: exact same target Role across DIFFERENT historical compositions.
    # This controls for role-name and role-description shortcuts.
    completekept,invalid=complete_systems(kept,roles)
    by_role_pos=collections.defaultdict(list);by_role_neg=collections.defaultdict(list)
    for sid,(rows,c) in completekept.items():
        for role,r in rows.items():
            (by_role_pos if r['label'] else by_role_neg)[role].append((r,tuple(sorted(c))))
    cross=[];rng=random.Random(SEED)
    for role in roles:
        positives=by_role_pos[role];negatives=by_role_neg[role]
        for pos,cp in positives:
            allowed=[neg for neg,cn in negatives if cp!=cn and neg['system_id']!=pos['system_id']]
            rng.shuffle(allowed)
            for neg in allowed[:24]:cross.append((pos,neg))
    if len(cross)<100:raise RuntimeError(f'Only {len(cross)} historical cross-composition same-role pairs; abort')
    pools={'hist_cross_same_role':cross,'hist_within_system':histwithin,
           'matched_same_query':matched,'mas5_within_system':maswithin}
    stats={'v137_train_rows_original':len(historical),'v137_train_rows_accepted':len(kept),
           'v137_train_complete_systems_original':len(traincomplete),
           'v137_train_complete_systems_accepted':len(completekept),
           'historical_systems_excluded_matching_DEV_composition':sorted(same_comp_systems),
           'train_rows_excluded_exact_DEV_response_overlap':len(exact_overlap),
           'v56_train_rows_included':len(vv),'matched_source':'V137_TRAIN_same_query_same_role' if matched else None,
           'unverified_V44_V54_external_matched_included':False,
           'pair_counts':{k:len(v) for k,v in pools.items()},**split}
    return roles,descs,val,test,pools,stats

def rootcheck(out):
    out=Path(out).resolve()
    if out.parent!=BASE.resolve() or not out.name.startswith('generalize_'):
        raise RuntimeError('Run output must be a new generalize_* directory below sensor_v510')
    return out

def preflight(out):
    out=rootcheck(out);roles,descs,val,test,pools,stats=obtain_sources()
    report={'status':'READY','purpose':'improve unseen-role-COMPOSITIONS, NOT old MAS5 train AUC',
            'roles':len(roles),'stats':stats,'original_sensor_adapter':str(OLD),
            'checkpoint_selected_on':'V137_DEV validation system compositions ONLY',
            'heldout_test_used_during_training':False,
            'heldout_caveat':'V137 DEV aggregate metrics were already examined in prior V57 work; exploratory split, not pristine external test',
            'no_mas_or_paid_api_calls':True,'no_v57_or_v58_jobs_touched':True}
    out.mkdir(parents=True,exist_ok=False)
    dump(out/'preflight.json',report)
    dump(out/'split_manifest.json',stats)
    print('V510_PREFLIGHT',json.dumps({'val_systems':len(val),'test_systems':len(test),'train_rows':stats['v137_train_rows_accepted'],
          'pair_counts':stats['pair_counts'],'excluded_same_compositions':len(stats['historical_systems_excluded_matching_DEV_composition']),
          'no_api_calls':True},ensure_ascii=False),flush=True)
    return report

def dev_metrics(scores,rows):
    vals={};truth={}
    for sid,(observations,positives) in rows.items():
        vals[sid]={role:float(scores[(sid,role)]) for role in observations}
        truth[sid]=positives
    return v57.auc20(vals,truth)

def load_model(adapter,train=False):
    import torch
    from peft import PeftConfig,PeftModel,prepare_model_for_kbit_training
    from transformers import AutoTokenizer,AutoModelForSequenceClassification,BitsAndBytesConfig
    cfg=PeftConfig.from_pretrained(str(adapter),local_files_only=True)
    if cfg.base_model_name_or_path!='Qwen/Qwen2.5-3B-Instruct':raise RuntimeError('Unexpected base model')
    tok=AutoTokenizer.from_pretrained(str(adapter),local_files_only=True)
    if tok.pad_token_id is None:tok.pad_token=tok.eos_token
    tok.padding_side='right';tok.truncation_side='right'
    quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16)
    base=AutoModelForSequenceClassification.from_pretrained(cfg.base_model_name_or_path,num_labels=1,
          quantization_config=quant,device_map={'':0},dtype=torch.bfloat16,attn_implementation='eager',local_files_only=True)
    base.config.pad_token_id=tok.pad_token_id;base.config.use_cache=False
    if train:base=prepare_model_for_kbit_training(base,use_gradient_checkpointing=True,
                                    gradient_checkpointing_kwargs={'use_reentrant':False})
    model=PeftModel.from_pretrained(base,str(adapter),is_trainable=train,local_files_only=True)
    return model,tok

def encode_rows(allrows,tok,descs,max_tokens):
    from analysis.role_direct_within_v4.v57_sensor_night import sensor_text
    cache={};truncated=collections.Counter()
    for r in allrows:
        k=tkey(r)
        if k in cache:continue
        text=sensor_text(r,tok,descs)
        ids=tok(text,add_special_tokens=True,max_length=max_tokens,truncation=True)['input_ids']
        # Faster and sufficient for reporting truncation: text generated with its own
        # clip_text; count actual post-clip tokenized length only at pretokenization.
        cache[k]=ids
        if len(tok(text,add_special_tokens=True,truncation=False)['input_ids'])>max_tokens:
            truncated[r.get('source','unknown')]+=1
    return cache,dict(truncated)

def tensorbatch(rs,cache,tok):
    import torch
    ids=[cache[tkey(r)] for r in rs];m=max(map(len,ids))
    return {'input_ids':torch.tensor([x+[tok.pad_token_id]*(m-len(x)) for x in ids],device='cuda'),
            'attention_mask':torch.tensor([[1]*len(x)+[0]*(m-len(x)) for x in ids],device='cuda')}

def eval_one(model,rows,cache,tok):
    import torch
    entries=[(sid,role,r) for sid,(obs,_) in rows.items() for role,r in obs.items()]
    model.eval();scores={}
    with torch.no_grad():
        for i in range(0,len(entries),3):
            batch=entries[i:i+3]
            with torch.autocast(device_type='cuda',dtype=torch.bfloat16):
                logits=model(**tensorbatch([r for _,_,r in batch],cache,tok)).logits.view(-1).float()
            for (sid,role,_),sc in zip(batch,logits.detach().cpu().tolist()):scores[(sid,role)]=sc
    return dev_metrics(scores,rows)

def train(out,hours,max_steps,eval_every,patience,seq_len):
    import torch
    import torch.nn.functional as F
    out=rootcheck(out)
    if not (out/'preflight.json').exists():raise RuntimeError('Run preflight first')
    if (out/'training_report.json').exists():raise RuntimeError('Refuse to overwrite completed training')
    if not 0<hours<=9:raise RuntimeError('Hours must be within (0,9]')
    if not 900<=seq_len<=1536:raise RuntimeError('Tokens must be 900..1536')
    v57.get_device_status();torch.manual_seed(SEED)
    roles,descs,val,test,pools,stats=obtain_sources()
    # No heldout TEST rows are tokenized or scored during training.
    active=[k for k in ('hist_cross_same_role','hist_within_system','matched_same_query','mas5_within_system') if pools[k]]
    weights={'hist_cross_same_role':.45,'hist_within_system':.25,'matched_same_query':.25,'mas5_within_system':.05}
    model,tok=load_model(OLD,train=True)
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=float(os.environ.get('V510_LR','3e-6')),weight_decay=.03)
    trainrows=[x for p in pools.values() for a,b in p for x in (a,b)]
    valrows=[r for obs,_ in val.values() for r in obs.values()]
    cache,trunc=encode_rows(trainrows+valrows,tok,descs,seq_len)
    baseline=eval_one(model,val,cache,tok)
    dump(out/'baseline_validation.json',baseline)
    print('V510_BASELINE_VAL',json.dumps({'auc':baseline['mean_within_mas_auc'],'exact_top3':baseline['exact_top3_count'],'systems':len(val)}),flush=True)
    def score(m):return .75*m['mean_within_mas_auc']+.25*(m['exact_top3_count']/len(val))
    best=score(baseline);beststep=0;best_adapter=str(OLD)
    randomgen=random.Random(SEED+1);start=time.monotonic();deadline=start+hours*3600
    bad=0;history=[];losses=[];stopped='max_steps';current_step=0
    dump(out/'current_status.json',{'state':'TRAINING','best_step':0,'best_val_auc':baseline['mean_within_mas_auc'],'best_adapter':best_adapter})
    for step in range(1,max_steps+1):
        current_step=step
        if time.monotonic()>deadline-8*60:stopped='time_budget';break
        model.train();opt.zero_grad(set_to_none=True)
        for _ in range(4):
            source=randomgen.choices(active,weights=[weights[k] for k in active],k=1)[0]
            pos,neg=randomgen.choice(pools[source])
            with torch.autocast(device_type='cuda',dtype=torch.bfloat16):
                logits=model(**tensorbatch([pos,neg],cache,tok)).logits.reshape(-1).float()
                # Same-role cross-composition contrasts constrain role-conditioned
                # shortcut, matched same-query controls lexical question confounds.
                loss=F.softplus(.5-(logits[0]-logits[1]))+.15*(F.softplus(-logits[0])+F.softplus(logits[1]))
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
            (loss/4).backward();losses.append(float(loss.detach().cpu()))
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],.8)
        opt.step()
        if step%25==0:
            print('V510_STEP',step,'train_loss',round(statistics.mean(losses[-100:]),4),'elapsed_min',round((time.monotonic()-start)/60,1),flush=True)
        if step%eval_every==0 or step==max_steps or time.monotonic()>deadline-15*60:
            metrics=eval_one(model,val,cache,tok);current=score(metrics)
            row={'step':step,'elapsed_hours':round((time.monotonic()-start)/3600,3),
                 'train_loss':round(statistics.mean(losses[-100:]),5),
                 'validation_auc':metrics['mean_within_mas_auc'],
                 'validation_exact_top3':metrics['exact_top3_count'],'validation_systems':len(val),
                 'selection_score':round(current,6),'checkpoint_selection_uses_test':False}
            history.append(row)
            with (out/'metrics_history.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
            print('V510_VALIDATION',json.dumps(row),flush=True)
            if current>best+0.001:
                best=current;beststep=step;bad=0
                dest=out/f'checkpoints/sensor_step_{step:05d}'
                if dest.exists():raise RuntimeError('Existing checkpoint; refusing to overwrite')
                dest.mkdir(parents=True)
                model.save_pretrained(str(dest));tok.save_pretrained(str(dest))
                best_adapter=str(dest)
                dump(out/'best.json',{'step':step,'selected_adapter':best_adapter,'validation':metrics,'selection_score':best})
                print('V510_BEST',step,round(best,5),flush=True)
            else:bad+=1
            dump(out/'current_status.json',{'state':'TRAINING','step':step,'bad_evals':bad,'best_step':beststep,
                                            'best_val_auc':metrics['mean_within_mas_auc'] if beststep==step else None,
                                            'best_adapter':best_adapter,'validation':row})
            if bad>=patience:stopped='validation_plateau';break
    report={'status':'FINISHED','stop_reason':stopped,'steps':current_step,'elapsed_hours':round((time.monotonic()-start)/3600,3),
            'best_step':beststep,'selected_adapter':best_adapter,'baseline_validation':baseline,
            'best_selection_score':round(best,6),'data_summary':stats,'token_truncations_train_and_val':trunc,
            'train_source_probabilities':{k:weights[k]/sum(weights[a] for a in active) for k in active},
            'dev_test_scored_during_training':False,'no_mas_api_calls':True,
            'old_checkpoint_untouched':str(OLD),'validation_is_different_compositions_from_direct_training_observations':True,
            'prior_V4_adapter_training_lineage_not_independently_audited':True,
            'note':'Selected by V137 dev VAL composition-grouped AUC/Top3, NOT V56 in-sample AUC.  V137 is exploratory, formerly viewed as aggregate.'}
    dump(out/'training_report.json',report)
    dump(out/'current_status.json',{'state':'FINISHED','best_adapter':best_adapter,'report_file':str(out/'training_report.json')})
    print('V510_TRAIN_FINISHED',json.dumps({'steps':current_step,'best_step':beststep,'selected':best_adapter,'stop':stopped}),flush=True)

def final_evaluate(out,seq_len):
    import torch,gc
    out=rootcheck(out)
    if not (out/'training_report.json').is_file():raise RuntimeError('Missing training result')
    roles,descs,val,test,pools,stats=obtain_sources()
    status=v57.get_device_status();selected=v57.load(out/'training_report.json')['selected_adapter']
    previous=[]
    for p in sorted((P/'artifacts/role_direct_within_v4/sensor_v57').glob('night_*/training_report.json'),reverse=True):
        try:
            z=v57.load(p)
            if z.get('selected_adapter') and (Path(z['selected_adapter'])/'adapter_config.json').exists():
                previous=[('V57_overfit_parent',z['selected_adapter'])];break
        except Exception:continue
    adapters=[('V4_frozen',str(OLD)),('V510_validation_selected',selected)]+previous
    # Fully independent comparison on exactly same test response rows.
    rows=[r for obs,_ in test.values() for r in obs.values()]
    report={'test_systems':len(test),'test_system_ids':sorted(test),'test_compositions':[sorted(list(t)) for t in {tuple(sorted(g[1])) for g in test.values()}],
            'V137_DEV_prior_aggregate_was_seen_in_V57':True,
            'validation_selected_without_test_scores':True,
            'no_mas_calls':True,'no_train_updates_during_comparison':True,'frozen_models':{}}
    for name,path in adapters:
        if not (Path(path)/'adapter_config.json').exists():
            print('V510_SKIP_MISSING',name,path,flush=True);continue
        model,tok=load_model(path,train=False)
        cache,trunc=encode_rows(rows,tok,descs,seq_len)
        metrics=eval_one(model,test,cache,tok)
        report['frozen_models'][name]={'adapter':str(path),'metrics':metrics,'truncated_inputs':trunc}
        print('V510_TEST',name,'auc',metrics['mean_within_mas_auc'],'exact3',metrics['exact_top3_count'],'/',len(test),flush=True)
        del model,tok,cache;gc.collect();torch.cuda.empty_cache()
    dump(out/'generalization_comparison.json',report)
    with (out/'generalization_comparison.tsv').open('w',encoding='utf8') as f:
        f.write('model\ttest_auc\ttest_exact_top3\ttest_systems\ttest_f1_at3\n')
        for name,entry in report['frozen_models'].items():
            m=entry['metrics'];f.write(f"{name}\t{m['mean_within_mas_auc']}\t{m['exact_top3_count']}\t{len(test)}\t{m['mean_role_f1_at3']}\n")
    print('V510_FINAL',str(out/'generalization_comparison.tsv'),flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['selftest','prepare','train','compare'])
    ap.add_argument('--run',type=Path);ap.add_argument('--hours',type=float,default=8.75)
    ap.add_argument('--max-steps',type=int,default=6000);ap.add_argument('--eval-every',type=int,default=150)
    ap.add_argument('--patience',type=int,default=12);ap.add_argument('--max-tokens',type=int,default=1280)
    a=ap.parse_args()
    if a.stage=='selftest':
        roles=list('abcdefghijklmnopqrst');systems=[]
        for i in range(12):
            pos={roles[(3*i+j)%20] for j in range(3)}
            systems.extend({'system_id':f's{i}','role':r,'label':int(r in pos),'query':'Q','answer':'A'+str(i)+r} for r in roles)
        val,test,split=partition(systems,roles)
        assert set(val).isdisjoint(test)
        assert {tuple(sorted(g[1])) for g in val.values()}.isdisjoint({tuple(sorted(g[1])) for g in test.values()})
        assert len(val)+len(test)==12
        print('V510_SELFTEST_OK (split by 3-role compositions, mock values only)')
    elif a.stage=='prepare':preflight(a.run)
    elif a.stage=='train':train(a.run,a.hours,a.max_steps,a.eval_every,a.patience,a.max_tokens)
    elif a.stage=='compare':final_evaluate(a.run,a.max_tokens)
if __name__=='__main__':main()
