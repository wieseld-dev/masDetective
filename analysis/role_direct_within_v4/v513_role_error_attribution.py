#!/usr/bin/env python3
"""V5.13: frozen-model, per-role explanation of V5.8 and V137-DEV failures.

Diagnostic only: no MAS calls, generator or model fitting; no model selection on
V58 or V137 DEV. Strict numerical reproduction of existing V512 V58 and DEV
results before writing any final reports. The true installed role set is used
only for evaluation/annotation, never to form input features or calibrators.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import math
import os
import statistics
from datetime import datetime
from pathlib import Path

import joblib

from analysis.role_direct_within_v4 import v5111_rolewise_verifier as base
from analysis.role_direct_within_v4 import v5113_dev_heldout as dev
from analysis.role_direct_within_v4 import v512_calibration_audit as prev
from analysis.role_direct_within_v4 import v510_generalizing_sensor as v510

ROOT=base.ROOT
OUTBASE=ROOT/'artifacts/role_direct_within_v4/sensor_v513'
ARMS=('answer_only','query_only','length_only')
TRANSFORMS=('raw','negative_centered')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_tsv(path,records,columns):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=columns,delimiter='\t',extrasaction='ignore')
        writer.writeheader()
        for row in records:writer.writerow({k:row.get(k,'') for k in columns})


def num(n):return round(float(n),7)


def rank_scores(scores,roles):
    order=sorted(roles,key=lambda r:(-scores[r],r))
    return {role:i+1 for i,role in enumerate(order)},order


def auc_system(scores,truth,roles):
    truth=set(truth)
    p=[r for r in roles if r in truth]
    n=[r for r in roles if r not in truth]
    if not p or not n:raise ValueError('System must contain positive and negative roles')
    return sum((scores[a]>scores[b])+.5*(scores[a]==scores[b]) for a in p for b in n)/(len(p)*len(n))


def acc_top3(order,truth):
    actual=set(order[:3]);expected=set(truth)
    return len(actual&expected),actual==expected


def summary_row(round_no,arm,method,scores,truth,roles):
    ranks,order=rank_scores(scores,roles)
    hits,exact=acc_top3(order,truth)
    return {'round':round_no,'arm':arm,'method':method,'auc':num(auc_system(scores,truth,roles)),
            'top3_correct':hits,'exact_top3':exact,
            'top3':','.join(order[:3]),
            'false_positives':','.join(r for r in order[:3] if r not in truth),
            'missed_present':','.join(sorted(set(truth)-set(order[:3]))),
            'ranks_of_present':','.join(f'{r}:{ranks[r]}' for r in sorted(truth))}


def require_close(desc,actual,reference,tolerance=2e-5):
    if abs(float(actual)-float(reference))>tolerance:
        raise RuntimeError(f'FATAL BASELINE MISMATCH {desc}: reconstructed={actual:.7f}, V512_reference={reference:.7f}; no diagnostic output will be finalized')


def load_v512_baseline():
    p=prev.ROOT_OUT/'latest_run_path.txt'
    if not p.is_file():raise RuntimeError('Missing V512 latest_run_path.txt; run V512 first')
    run=Path(p.read_text().strip()).resolve()
    if not run.is_dir() or not (run/'report.json').is_file() or not (run/'v58_fusion_replay.json').is_file():
        raise RuntimeError(f'Incomplete V512 baseline at {run}')
    if read_json(run/'report.json').get('status')!='COMPLETE':raise RuntimeError('V512 reference run is incomplete')
    return run,read_json(run/'report.json'),read_json(run/'v58_fusion_replay.json')


def get_v58_rows(roles):
    root=prev.available_v58()
    if root is None:raise RuntimeError('No V58 replay source directory; set V512_V58_ROOT')
    pf=read_json(root/'public_preflight.json')
    sid=pf['session_id']
    if set(pf['roles'])!=set(roles):raise RuntimeError('V58 role catalog mismatch')
    manifest=v510.v57.load(v510.v57.MANIFEST)
    match=[x for x in manifest['systems'] if x.get('split')=='dev' and x['session_id']==sid]
    if len(match)!=1:raise RuntimeError('Missing or ambiguous V58 ground truth session in manifest')
    truth=set(match[0]['roles'])
    if len(truth)!=3:raise RuntimeError('V58 expected exactly three true roles')
    rounds=[]
    for r in range(1,9):
        f=root/f'responses_public_dev_round_{r:02}.jsonl'
        if not f.is_file():break
        records=v510.v57.lines(f)
        if len(records)!=len(roles) or {x['target_role'] for x in records}!=set(roles):
            raise RuntimeError(f'V58 incomplete round {r}: {len(records)} records')
        rows=[]
        for record in records:
            if record.get('session_id')!=sid or record.get('execution_status')!='ok':
                raise RuntimeError(f'V58 bad execution/session in round {r}')
            rows.append({'system_id':sid,'role':record['target_role'],
                         'query':record['query'],'answer':record['final_output'],
                         'label':int(record['target_role'] in truth)})
        rounds.append(rows)
    if not rounds:raise RuntimeError('V58 has no complete rounds')
    return root,sid,truth,rounds


def summarize_dev(model,arm,roles,val,test,cal):
    valraw=base.infer(model,base.config_rows(val,roles),roles,arm)
    testraw=base.infer(model,base.config_rows(test,roles),roles,arm)
    # Regression: same test scores as V512 for this model, no new ML fitting.
    scores={}
    for method in TRANSFORMS:
        scores[method]={(sid,r):prev.calibrate(r,testraw[(sid,r)],method,cal) for sid in test for r in roles}
    sysrows=[];role_rows=[]
    for sid,(_,truth) in sorted(test.items()):
        raw={r:scores['raw'][(sid,r)] for r in roles}
        centered={r:scores['negative_centered'][(sid,r)] for r in roles}
        rr,ro=rank_scores(raw,roles)
        cr,co=rank_scores(centered,roles)
        for role in roles:
            role_rows.append({'system_id':sid,'role':role,'installed':int(role in truth),
                              'answer_raw':num(raw[role]),'answer_val_negative_median':num(cal['centers'][role]),
                              'answer_centered':num(centered[role]),'raw_rank':rr[role],
                              'centered_rank':cr[role],'rank_change_centering':rr[role]-cr[role],
                              'raw_top3':int(rr[role]<=3),'centered_top3':int(cr[role]<=3)})
        for method,cur,order in [('raw',raw,ro),('negative_centered',centered,co)]:
            sysrows.append({'system_id':sid,'method':method,
                 'auc':num(auc_system(cur,truth,roles)),
                 'top3':','.join(order[:3]),
                 'hits':len(set(order[:3])&truth),
                 'false_positives':','.join(r for r in order[:3] if r not in truth),
                 'misses':','.join(sorted(truth-set(order[:3]))),
                 'true_roles':','.join(sorted(truth))})
    return scores,role_rows,sysrows


def make_v58_diagnostics(models,calibrators,roles,truth,rounds,training,old_replay):
    sid=rounds[0][0]['system_id']
    history=collections.defaultdict(lambda:collections.defaultdict(list))
    role_rows=[];round_rows=[];rank_rows=[];text_rows=[];overlaps=[]
    training_keys={base.record_id(row) for row in training}
    old=old_replay.get('answer_only',{})
    reference_rounds={r['round']:r for r in old.get('rounds',[])}
    if len(reference_rounds)!=len(rounds):
        raise RuntimeError('V512 V58 replay length differs from current data; wrong V58 run?')
    for idx,observations in enumerate(rounds,1):
        collision=sum(base.record_id(row) in training_keys for row in observations)
        overlaps.append(collision)
        ref=reference_rounds[idx]
        if collision!=ref['exact_training_observation_overlaps']:
            raise RuntimeError(f'V58 overlap count mismatch in round {idx}')
        current={arm:base.infer(models[arm],observations,roles,arm) for arm in ARMS}
        # Score and store all per-role data before attaching labels/truth.
        for arm in ARMS:
            cal=calibrators[arm]
            for role in roles:
                s=current[arm][(sid,role)]
                history[arm][role].append(s)
        for arm in ARMS:
            cal=calibrators[arm]
            for method in TRANSFORMS:
                avg={r:statistics.mean(prev.calibrate(r,x,method,cal) for x in history[arm][r]) for r in roles}
                rnk,order=rank_scores(avg,roles)
                item=summary_row(idx,arm,method,avg,truth,roles)
                round_rows.append(item)
                refkey=f'{method}/mean'
                old_result=old_replay.get(arm,{}).get('rounds',[])
                if old_result:
                    expected=old_result[idx-1]['fusion'][refkey]
                    require_close(f'V58 round={idx} arm={arm} method={method}',item['auc'],expected['auc'])
                    if item['top3_correct']!=expected['top3_correct'] or item['exact_top3']!=expected['exact_top3']:
                        raise RuntimeError(f'V58 top3 mismatch round={idx} arm={arm} method={method}')
                if arm!='answer_only':continue
                for role in roles:
                    rank_rows.append({'round':idx,'role':role,'installed':int(role in truth),
                        'method':method,'cumulative_score':num(avg[role]),'rank':rnk[role],
                        'top3':int(rnk[role]<=3),'is_fp':int(rnk[role]<=3 and role not in truth),
                        'is_missed':int(rnk[role]>3 and role in truth)})
        rrank={method:rank_scores({r:statistics.mean(prev.calibrate(r,x,method,calibrators['answer_only'])
                                   for x in history['answer_only'][r]) for r in roles},roles)[0]
               for method in TRANSFORMS}
        for obs in observations:
            role=obs['role'];q=obs['query'];a=obs['answer']
            current_raw=current['answer_only'][(sid,role)]
            raw_seq=history['answer_only'][role]
            center=calibrators['answer_only']['centers'][role]
            role_rows.append({
                'round':idx,'role':role,'installed':int(role in truth),
                'answer_raw_this_round':num(current_raw),
                'answer_val_negative_median':num(center),
                'answer_centered_this_round':num(current_raw-center),
                'answer_raw_cum_mean':num(statistics.mean(raw_seq)),
                'answer_centered_cum_mean':num(statistics.mean(raw_seq)-center),
                'raw_rank_cum':rrank['raw'][role],
                'centered_rank_cum':rrank['negative_centered'][role],
                'raw_in_top3':int(rrank['raw'][role]<=3),
                'centered_in_top3':int(rrank['negative_centered'][role]<=3),
                'delta_raw_from_round1':num(current_raw-raw_seq[0]),
                'delta_raw_from_previous':num(current_raw-raw_seq[-2]) if len(raw_seq)>=2 else '',
                'query_only_raw':num(current['query_only'][(sid,role)]),
                'length_only_raw':num(current['length_only'][(sid,role)]),
                'question_chars':len(q),'answer_chars':len(a),
                'answer_words':len(a.split()),
                'exact_train_overlap':int(base.record_id(obs) in training_keys)})
            text_rows.append({'round':idx,'role':role,'installed':int(role in truth),
                'question':base.txt(q,800),'answer_excerpt':base.txt(a,450),
                'answer_raw':num(current_raw),'raw_rank_cum':rrank['raw'][role]})
    return {'role_rows':role_rows,'round_rows':round_rows,'rank_rows':rank_rows,
            'text_rows':text_rows,'overlaps':overlaps}


def explain_markdown(rows,role_rows,truth,dev_sys,source,reference):
    lines=['# V5.13 — Role ranking error attribution (retrospective)',
           '',f'- V58 installed roles: **{", ".join(sorted(truth))}**',
           f'- V58 source: `{source}`',
           f'- V512 reference: `{reference}`',
           '- **Frozen 20 role-specific TF-IDF verifiers; no MAS calls and no weight updates.**',
           '- V58 composition was seen during training (V56); only exact interaction overlap was audited.',
           '- V137 DEV has previously been inspected; this is diagnostic, not independent model selection.',
           '- Model inputs are only user question/visible answer per arm. Installed roles are used only when evaluating.',
           '', '## V58 rounds: ranking of 20 roles', '']
    by_round=collections.defaultdict(list)
    for row in rows:
        if row['arm']=='answer_only':by_round[row['round']].append(row)
    for rnd in sorted(by_round):
        methods={r['method']:r for r in by_round[rnd]}
        raw=methods['raw'];center=methods['negative_centered']
        lines.extend([f'### Round {rnd}', '',
            f'- Raw mean: AUC **{raw["auc"]:.3f}**, Top-3 `{raw["top3"]}`, correct **{raw["top3_correct"]}/3**; misses `{raw["missed_present"] or "none"}`.',
            f'- Negative-centered mean: AUC **{center["auc"]:.3f}**, Top-3 `{center["top3"]}`, correct **{center["top3_correct"]}/3**.',
            '', '| Role | Present? | Raw logit mean | Raw rank | Null median (VAL) | Centered rank |',
            '|---|---:|---:|---:|---:|---:|'])
        group=sorted((r for r in role_rows if r['round']==rnd),key=lambda r:(r['raw_rank_cum'],r['role']))
        for r in group:
            lines.append(f'| `{r["role"]}` | {"YES" if r["installed"] else "no"} | {r["answer_raw_cum_mean"]:.4f} | {r["raw_rank_cum"]} | {r["answer_val_negative_median"]:.4f} | {r["centered_rank_cum"]} |')
        lines.append('')
    lines+=['## V137 DEV (24 systems)', '',
        f'- Number of systems: {dev_sys}; role-specific logits and both ranks in `v137_role_predictions.tsv`.',
        '- `v137_system_errors.tsv` isolates systems where absent roles enter Top-3.',
        '- V58 changes in a role\'s raw logit **do not measure per-role binary AUC**, because one MAS has a fixed role-presence label.',
        '- Frozen-model historical replay **cannot prove an adaptive improvement** or new-composition generalization.',
        '- Read `v58_probe_excerpts.tsv` to compare probes and visible response excerpts for potential shortcuts/truncation.',
        '']
    return '\n'.join(lines)


def run(source=None,out=None):
    refdir,ref_report,old_replay=load_v512_baseline()
    source=Path(source).resolve() if source else prev.latest_source().resolve()
    if not source.is_dir():raise RuntimeError(f'Missing V5113 model source: {source}')
    roles,training,val,test,split,priority=dev.prep()
    if len(roles)!=20:raise RuntimeError(f'Unexpected role catalog: {len(roles)}')
    models={};cal={};sys_rows=[];test_role_rows=[];crossrows=[]
    calibration_file=refdir/'calibration_validation_only.json'
    if not calibration_file.is_file():
        raise RuntimeError('Missing V512 frozen validation calibrators: '+str(calibration_file))
    frozen_calibrators=read_json(calibration_file)
    for arm in ARMS:
        obj=joblib.load(source/f'model_{arm}.joblib')
        if obj.get('arm')!=arm or set(obj.get('roles',[]))!=set(roles):raise RuntimeError(f'Model source mismatch for {arm}')
        models[arm]=obj['models']
        # Reuse exact V512 VALIDATION-only offsets. Do not refit ANY model.
        cal[arm]=frozen_calibrators[arm]
        if set(cal[arm]['centers'])!=set(roles) or set(cal[arm]['scales'])!=set(roles):
            raise RuntimeError(f'Frozen calibration role catalog mismatch: {arm}')
        test_scores,rows,systems=summarize_dev(models[arm],arm,roles,val,test,cal[arm])
        for method in TRANSFORMS:
            avg=statistics.mean(r['auc'] for r in systems if r['method']==method)
            matched=[x for x in ref_report['comparison'] if x['arm']==arm and x['calibration']==method]
            if len(matched)!=1:raise RuntimeError(f'No unique V512 reference for {arm}/{method}')
            require_close(f'V137 DEV {arm}/{method}',avg,matched[0]['test_auc'])
        if arm=='answer_only':test_role_rows=rows;sys_rows=systems
        else:
            bykey={(r['system_id'],r['role']):r for r in rows}
            for r in test_role_rows:
                other=bykey[(r['system_id'],r['role'])]
                r[f'{arm}_raw']=other['answer_raw']
        for role in roles:
            crossrows.append({'arm':arm,'role':role,
                'val_negative_count':cal[arm]['validation_coverage'][role]['val_absent'],
                'val_positive_count':cal[arm]['validation_coverage'][role]['val_present'],
                'val_negative_median':num(cal[arm]['centers'][role]),
                'val_negative_scale':num(cal[arm]['scales'][role])})
    v58_dir,sid,truth,rounds=get_v58_rows(roles)
    obs=make_v58_diagnostics(models,cal,roles,truth,rounds,training,old_replay)
    if any(obs['overlaps']):raise RuntimeError(f'V58 unexpectedly overlaps training observations {obs["overlaps"]}; abort')
    if ref_report.get('source') and Path(ref_report['source']).resolve()!=source:
        raise RuntimeError('V512 report and current model source directories differ')
    if old_replay['answer_only'].get('source') and Path(old_replay['answer_only']['source']).resolve()!=v58_dir.resolve():
        raise RuntimeError('V512 source V58 run differs from current source')
    runout=Path(out).resolve()
    if runout.exists():raise RuntimeError(f'Refusing to overwrite output: {runout}')
    if runout.parent!=OUTBASE.resolve():raise RuntimeError('Output must be direct child of V513 artifact directory')
    runout.mkdir(parents=True)
    write_tsv(runout/'v58_round_summary.tsv',obs['round_rows'],
        ['round','arm','method','auc','top3_correct','exact_top3','top3','false_positives','missed_present','ranks_of_present'])
    write_tsv(runout/'v58_role_scores.tsv',obs['role_rows'],
        ['round','role','installed','answer_raw_this_round','answer_val_negative_median',
        'answer_centered_this_round','answer_raw_cum_mean','answer_centered_cum_mean',
        'raw_rank_cum','centered_rank_cum','raw_in_top3','centered_in_top3',
        'delta_raw_from_round1','delta_raw_from_previous','query_only_raw',
        'length_only_raw','question_chars','answer_chars','answer_words','exact_train_overlap'])
    write_tsv(runout/'v58_rankings.tsv',obs['rank_rows'],
        ['round','role','installed','method','cumulative_score','rank','top3','is_fp','is_missed'])
    write_tsv(runout/'v58_probe_excerpts.tsv',obs['text_rows'],
        ['round','role','installed','question','answer_excerpt','answer_raw','raw_rank_cum'])
    write_tsv(runout/'v137_role_predictions.tsv',test_role_rows,
        ['system_id','role','installed','answer_raw','answer_val_negative_median','answer_centered',
        'raw_rank','centered_rank','rank_change_centering','raw_top3','centered_top3',
        'query_only_raw','length_only_raw'])
    write_tsv(runout/'v137_system_errors.tsv',sys_rows,
        ['system_id','method','auc','top3','hits','false_positives','misses','true_roles'])
    write_tsv(runout/'validation_null_centers.tsv',crossrows,
        ['arm','role','val_positive_count','val_negative_count','val_negative_median','val_negative_scale'])
    (runout/'role_rankings_report.md').write_text(explain_markdown(obs['round_rows'],obs['role_rows'],truth,len(test),v58_dir,refdir),encoding='utf-8')
    output={'status':'COMPLETE','model_source':str(source),'v512_reference':str(refdir),
      'v58_source':str(v58_dir),'v58_system_id':sid,'v58_true_roles_eval_only':sorted(truth),
      'v58_round_count':len(rounds),'v58_overlap_training_by_round':obs['overlaps'],
      'v137_dev_system_count':len(test),'v137_dev_previously_inspected':True,
      'v58_composition_seen_in_training':True,'new_mas_calls':0,'model_fitting_or_updates':0,
      'calibration_source':'FROZEN_V512_CALIBRATION_JSON_NO_REFITTING',
      'consistency_checks':'V512 V137 raw/centered and V58 rounds reconstructed to within 2e-5',
      'interpretation':'Hypothesis generation only. V58 uses one known-composition MAS; per-role changing scores are NOT binary AUC or causal adaptive improvements.',
      'v58_answer_raw_mean_trajectory':[
          {'round':x['round'],'auc':x['auc'],'top3':x['top3'],'top3_correct':x['top3_correct']}
          for x in obs['round_rows'] if x['arm']=='answer_only' and x['method']=='raw']}
    base.dump(runout/'report.json',output)
    print('V513_COMPLETED',runout,flush=True)
    for r in output['v58_answer_raw_mean_trajectory']:
        print('V513_V58',json.dumps(r),flush=True)
    print('V513_OUTPUT_FILES',', '.join(x.name for x in sorted(runout.iterdir())),flush=True)


def selftest():
    roles=['code','planning','research','math','writing']
    present={'code','research'}
    scores={'code':0.7,'planning':1.0,'research':0.6,'math':0.2,'writing':0.1}
    ranks,order=rank_scores(scores,roles)
    assert order[:3]==['planning','code','research'] and ranks['code']==2
    row=summary_row(1,'answer_only','raw',scores,present,roles)
    assert row['top3_correct']==2 and row['false_positives']=='planning'
    assert row['exact_top3'] is False
    assert abs(auc_system(scores,present,roles)-4/6)<1e-12
    assert len({x['round'] for x in [row]})==1
    require_close('selftest',0.784314,0.784314,1e-6)
    try:require_close('must_fail',0.784314,0.392157)
    except RuntimeError:pass
    else:raise AssertionError('Regression guard did not fail')
    print('V513_SELFTEST_OK synthetic role-ranking, miss attribution, AUC and baseline guard',flush=True)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('stage',choices=('selftest','preflight','run'))
    ap.add_argument('--source',default='')
    ap.add_argument('--out',default='')
    args=ap.parse_args()
    if args.stage=='selftest':return selftest()
    ref,report,replay=load_v512_baseline()
    source=Path(args.source).resolve() if args.source else prev.latest_source()
    for arm in ARMS:
        if not (source/f'model_{arm}.joblib').is_file():raise RuntimeError(f'Missing {arm} model in {source}')
    if args.stage=='preflight':
        v58=prev.available_v58()
        if v58 is None:raise RuntimeError('V58 source missing')
        print('V513_PREFLIGHT_OK',json.dumps({'v512_reference':str(ref),'model_source':str(source),
            'v58_source':str(v58),'new_mas_calls':0,'model_training':False}),flush=True)
        return
    if not args.out:raise RuntimeError('--out mandatory')
    return run(source,args.out)


if __name__=='__main__':main()
