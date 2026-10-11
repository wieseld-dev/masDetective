#!/usr/bin/env python3
"""MAS5 V5.5: Diagnostic frozen Sensor inference on the EXISTING V5.4 Round-2 pilot.

Only 18 locally saved public query/final-answer observations. No MAS/API calls,
training, selector reranking, or claimed MAS5 AUC. TARGET_PRESENT labels are used
exclusively AFTER logits have been computed to tabulate paired diagnostics.
"""
from __future__ import annotations
import argparse
import collections
import csv
import json
import math
import os
from pathlib import Path
import statistics
import sys

PROJECT = Path('/home/wieseld/mas_rl_detective')
DEFAULT_SENSOR = PROJECT / 'artifacts/role_direct_within_v4/runs/v4_20261009_185359_1475154/checkpoints/sensor_epoch02'
ARMS = ('adaptive_a', 'adaptive_b', 'fixed')
CONDITIONS = ('TARGET_PRESENT', 'TARGET_ABSENT')
ROLES = ('research', 'database_design', 'customer_support_triage')


def load_jsonl(path):
    p=Path(path)
    with p.open(encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def save_json(path, data):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists(): raise RuntimeError('Won\'t overwrite: '+str(p))
    p.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def save_jsonl(path, records):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists(): raise RuntimeError('Won\'t overwrite: '+str(p))
    with p.open('w',encoding='utf-8') as f:
        for row in records:f.write(json.dumps(row,ensure_ascii=False)+'\n')


def save_tsv(path, rows, fields):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists(): raise RuntimeError('Won\'t overwrite: '+str(p))
    with p.open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,delimiter='\t',fieldnames=fields,extrasaction='ignore')
        writer.writeheader();writer.writerows(rows)


def validate(root):
    """Returns only already-observed public inputs plus labels kept outside scoring."""
    root=Path(root).resolve()
    path=root/'responses_train_round_02.jsonl'
    if not path.is_file():raise RuntimeError('V5.4 responses not found: '+str(path))
    rows=load_jsonl(path)
    if len(rows)!=18: raise RuntimeError(f'Expected exactly 18 V5.4 responses, got {len(rows)}')
    candidates=load_jsonl(root/'candidates_train_round_02.jsonl')
    if len(candidates)!=18:raise RuntimeError('Expected exactly 18 candidates')
    expected={(str(c['session_id']),int(c['candidate_index'])):c for c in candidates}
    if len(expected)!=18:raise RuntimeError('Duplicate candidate (session,index)')
    seen=set(); grouped=collections.defaultdict(dict)
    for row in rows:
        sid=str(row['session_id']);slot=int(row['candidate_index'])
        key=(sid,slot)
        if key in seen or key not in expected:raise RuntimeError('Unexpected/duplicate response key '+str(key))
        seen.add(key)
        c=expected[key]
        for fld in ('query','target_role','pair_id','arm','condition'):
            if row.get(fld)!=c.get(fld):raise RuntimeError('Candidate/response mismatch '+fld+' '+str(key))
        if row.get('execution_status')!='ok':raise RuntimeError('Non-ok execution '+str(key))
        if row.get('target_role') not in ROLES:raise RuntimeError('Unknown role '+str(row.get('target_role')))
        if row.get('arm') not in ARMS or row.get('condition') not in CONDITIONS:
            raise RuntimeError('Unexpected arm/condition '+str(key))
        q=str(row.get('query','')); a=str(row.get('final_output',''))
        if not q.strip() or not a.strip():raise RuntimeError('Missing public query/response '+str(key))
        groupkey=(row['target_role'],row['pair_id'],row['arm'])
        if row['condition'] in grouped[groupkey]:raise RuntimeError('Double condition '+str(groupkey))
        grouped[groupkey][row['condition']]=row
    if len(seen)!=18 or len(grouped)!=9:
        raise RuntimeError(f'Unexpected matches {len(seen)} / groups {len(grouped)}')
    for g, pair in grouped.items():
        if set(pair)!=set(CONDITIONS):raise RuntimeError('Missing POS/NEG for '+str(g))
        if pair['TARGET_PRESENT']['query']!=pair['TARGET_ABSENT']['query']:
            raise RuntimeError('Query-text confound for '+str(g))
        if len({pair[x]['pair_id'] for x in CONDITIONS})!=1:
            raise RuntimeError('Mixed pair IDs')
    if len({(role,pid) for role,pid,arm in grouped})!=3:
        raise RuntimeError('Expected three distinct paired MAS configurations')
    return rows,grouped


def sensor_score(public_rows, adapter, max_length=640):
    import torch
    from peft import PeftConfig, PeftModel
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from analysis.role_direct_within_v4.pipeline import catalog, make_sensor_text
    if not torch.cuda.is_available():raise RuntimeError('CUDA GPU required')
    card=torch.cuda.get_device_name(0)
    if '4090' not in card:raise RuntimeError('Expected RTX 4090, got '+str(card))
    adapter=Path(adapter).resolve()
    if not (adapter/'adapter_config.json').is_file():
        raise RuntimeError('Frozen trained Sensor adapter unavailable: '+str(adapter))
    cfg=PeftConfig.from_pretrained(str(adapter),local_files_only=True)
    tok=AutoTokenizer.from_pretrained(str(adapter),local_files_only=True)
    if tok.pad_token_id is None:tok.pad_token=tok.eos_token
    model=AutoModelForSequenceClassification.from_pretrained(
        cfg.base_model_name_or_path,num_labels=1,device_map={'':0},
        dtype=torch.bfloat16,attn_implementation='eager',local_files_only=True)
    model.config.pad_token_id=tok.pad_token_id
    model=PeftModel.from_pretrained(model,str(adapter),is_trainable=False,local_files_only=True).eval()
    desc=catalog()['DESCRIPTIONS']
    out=[]
    with torch.inference_mode():
        for i,r in enumerate(public_rows,1):
            # ALLOWED public inputs ONLY: target_role, query and final answer.
            text=make_sensor_text(str(r['target_role']),str(r['query']),str(r['final_output']),desc)
            ids=tok(text,add_special_tokens=True,truncation=False)['input_ids']
            truncated=len(ids)>max_length
            enc=tok([text],truncation=True,max_length=max_length,return_tensors='pt',padding=True).to('cuda')
            with torch.autocast(device_type='cuda',dtype=torch.bfloat16):
                logit=float(model(**enc).logits.reshape(-1)[0].float().item())
            if not math.isfinite(logit):raise RuntimeError('Nonfinite sensor logit')
            # No label, pair_id or condition has been passed to Sensor.
            out.append({'session_id':str(r['session_id']),
                        'candidate_index':int(r['candidate_index']),
                        'sensor_logit':round(logit,7),
                        'input_tokens_full':len(ids),
                        'truncated_at_640':truncated})
            print('MAS5_V55_SCORE',f'{i}/{len(public_rows)}',
                  'role='+str(r['target_role']),'arm='+str(r['arm']),
                  'tokens='+str(len(ids)),'logit='+f'{logit:.5f}',flush=True)
    return out,card


def diagnostic(rows, scores):
    bykey={(r['session_id'],int(r['candidate_index'])):r for r in rows}
    scorekey={(r['session_id'],int(r['candidate_index'])):r for r in scores}
    if len(scorekey)!=18 or set(scorekey)!=set(bykey):raise RuntimeError('Missing/duplicate scores')
    groups=collections.defaultdict(dict)
    for key,x in bykey.items():
        group=(x['target_role'],x['pair_id'],x['arm'])
        groups[group][x['condition']]=(x,scorekey[key])
    arms=[]
    for (role,pid,arm),conditions in sorted(groups.items()):
        (pos,sp)=conditions['TARGET_PRESENT'];(neg,sn)=conditions['TARGET_ABSENT']
        margin=sp['sensor_logit']-sn['sensor_logit']
        active=role in pos.get('selected_roles_training_only',[])
        arms.append({'target_role':role,'pair_id':pid,'arm':arm,
                     'present_logit':sp['sensor_logit'], 'absent_logit':sn['sensor_logit'],
                     'present_minus_absent_logit':round(margin,7),
                     'present_above_absent':bool(margin>0),
                     'tie':bool(margin==0),
                     'present_target_activated_training_only':bool(active),
                     'present_input_tokens':sp['input_tokens_full'],
                     'absent_input_tokens':sn['input_tokens_full'],
                     'present_truncated':sp['truncated_at_640'],
                     'absent_truncated':sn['truncated_at_640']})
    comparison=[]
    for role,pid in sorted({(a['target_role'],a['pair_id']) for a in arms}):
        m={a['arm']:a for a in arms if a['target_role']==role and a['pair_id']==pid}
        if set(m)!=set(ARMS):raise RuntimeError('Not three arms for '+role)
        adaptive=(m['adaptive_a']['present_minus_absent_logit']+
                  m['adaptive_b']['present_minus_absent_logit'])/2
        fixed=m['fixed']['present_minus_absent_logit']
        comparison.append({'target_role':role,'pair_id':pid,
             'adaptive_a_margin':m['adaptive_a']['present_minus_absent_logit'],
             'adaptive_b_margin':m['adaptive_b']['present_minus_absent_logit'],
             'mean_adaptive_margin':round(adaptive,7),
             'fixed_margin':round(fixed,7),
             'adaptive_minus_fixed_margin':round(adaptive-fixed,7),
             'adaptive_outperformed_fixed':bool(adaptive>fixed)})
    adap=[a for a in arms if a['arm']!='fixed']
    fixed=[a for a in arms if a['arm']=='fixed']
    s={
       'status':'FROZEN_SENSOR_DIAGNOSTIC_ONLY',
       'executions':len(rows),'paired_arms':len(arms),'matched_pairs':len(comparison),
       'adaptive_arms':len(adap),'fixed_arms':len(fixed),
       'adaptive_pairwise_correct_fraction':sum(a['present_above_absent']+.5*a['tie'] for a in adap)/len(adap),
       'fixed_pairwise_correct_fraction':sum(a['present_above_absent']+.5*a['tie'] for a in fixed)/len(fixed),
       'mean_adaptive_present_minus_absent_logit':round(statistics.mean(a['present_minus_absent_logit'] for a in adap),7),
       'mean_fixed_present_minus_absent_logit':round(statistics.mean(a['present_minus_absent_logit'] for a in fixed),7),
       'mean_adaptive_minus_fixed_margin_over_3_background_pairs':round(statistics.mean(x['adaptive_minus_fixed_margin'] for x in comparison),7),
       'adaptive_better_background_pairs':sum(x['adaptive_outperformed_fixed'] for x in comparison),
       'input_count_truncated_at_640':sum(bool(x['truncated_at_640']) for x in scores),
       'roles':list(ROLES),
       'sensor_scores_are_uncalibrated_logits':True,
       'this_is_not_within_mas_auc':True,
       'this_is_not_an_eig_or_dpo_reward':True,
       'responses_are_from_manually_rewritten_probes':True,
       'stateless_mas_no_history_in_r2_input':True,
       'substitution_of_other_roles_is_a_confound':True,
       'n_is_too_small_for_conclusive_comparison':True,
       'pilot_reused_v49_training_pairs_not_a_heldout_test':True,
       'no_api_calls':True,
       'no_training':True,
       'notes':'Scores computed only from target role name, public query and final response. Labels were used after scoring. Positive margin means the frozen sensor assigned higher target-role evidence to the PRESENT response than to the ABSENT response for the exact same query. No Selector, no full 20-role ranking.'
    }
    return arms,comparison,s


def check(args):
    rows,grouped=validate(args.root)
    print('MAS5_V55_PREFLIGHT',json.dumps({'status':'READY_FOR_FROZEN_SENSOR_OFFLINE',
       'responses':len(rows),'query_matched_arms':len(grouped),
       'roles':list(ROLES),'no_api_calls':True,'no_training':True},ensure_ascii=False),flush=True)


def score(args):
    root=Path(args.root).resolve();out=Path(args.out).resolve()
    rows,_=validate(root)
    if out.exists():
        # Submission writes source_root.txt and job_id.txt before scoring starts.
        forbidden=['sensor_scores.jsonl','paired_arm_scores.jsonl','paired_arm_scores.tsv',
                   'pair_comparison.jsonl','pair_comparison.tsv','sensor_diagnostic_report.json']
        if any((out/name).exists() for name in forbidden):
            raise RuntimeError('Refusing to overwrite previous Sensor results in '+str(out))
    else:out.mkdir(parents=True)
    adapter=Path(os.getenv('V55_SENSOR_ADAPTER',str(DEFAULT_SENSOR)))
    scored,device=sensor_score(rows,adapter,max_length=640)
    arms,comparison,report=diagnostic(rows,scored)
    report['sensor_adapter']=str(adapter)
    report['gpu']=device
    report['source_v54']=str(root)
    # Save scores *after* GPU inference; condition never enters Sensor input.
    save_jsonl(out/'sensor_scores.jsonl',scored)
    save_jsonl(out/'paired_arm_scores.jsonl',arms)
    save_tsv(out/'paired_arm_scores.tsv',arms,list(arms[0].keys()))
    save_jsonl(out/'pair_comparison.jsonl',comparison)
    save_tsv(out/'pair_comparison.tsv',comparison,list(comparison[0].keys()))
    save_json(out/'sensor_diagnostic_report.json',report)
    print('MAS5_V55_REPORT',json.dumps(report,ensure_ascii=False),flush=True)
    print('RESULTS',out,flush=True)


def test(args):
    rows,_=validate(args.root)
    # SYNTHETIC score values FOR PROGRAM UNIT TEST ONLY; never written as findings.
    mocks=[]
    for r in rows:
        mocks.append({'session_id':r['session_id'],
            'candidate_index':r['candidate_index'],
            'sensor_logit':1. if r['condition']=='TARGET_PRESENT' else 0.,
            'input_tokens_full':10,'truncated_at_640':False})
    arms,pairs,rep=diagnostic(rows,mocks)
    assert len(arms)==9 and len(pairs)==3
    assert rep['adaptive_pairwise_correct_fraction']==1
    assert rep['fixed_pairwise_correct_fraction']==1
    assert rep['mean_adaptive_minus_fixed_margin_over_3_background_pairs']==0
    print('MAS5_V55_UNIT_TEST_PASS (mock logits only; NO scientific score computed)',flush=True)


def main():
    parser=argparse.ArgumentParser()
    p=parser.add_subparsers(dest='mode',required=True)
    for mode in ('check','score','test'):
        sub=p.add_parser(mode)
        sub.add_argument('--root',type=Path,required=True)
        if mode=='score':sub.add_argument('--out',type=Path,required=True)
    a=parser.parse_args()
    return {'check':check,'score':score,'test':test}[a.mode](a)

if __name__=='__main__':main()
