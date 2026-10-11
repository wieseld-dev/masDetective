#!/usr/bin/env python3
"""V5.11: train 20 *role-wise* PRESENT/ABSENT response verifiers.

This is an offline classical-ML SENSOR baseline, not Qwen finetuning.
Training observations: already paid V137 TRAIN + V56; DEV is split BY ROLE
COMPOSITION into validation and untouched test via V510. No API calls.
Models NEVER receive labels, installed role lists, system IDs, or traces as inputs.

Report three controls: query-only, answer-only, query+answer. Query-only performing
similarly to answer-only is evidence of probe/topic shortcuts, not reliable detection.
"""
from __future__ import annotations
import argparse, collections, csv, hashlib, json, math, os, random, statistics, time
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import joblib

from analysis.role_direct_within_v4 import v510_generalizing_sensor as v510

ROOT = v510.P
BASE = ROOT/'artifacts/role_direct_within_v4/sensor_v511'
SEED=5112026
ARMS=('query_only','answer_only','answer_only_matched','query_answer','length_only')

def dump(path, data):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(data,ensure_ascii=False,indent=2,default=str)+'\n',encoding='utf-8')

def txt(s, n=4400):
    s=' '.join(str(s or '').split())
    if len(s)<=n:return s
    return s[:int(n*.65)]+' [...middle omitted...] '+s[-int(n*.35):]

def text(row,arm):
    q=txt(row['query'],800);a=txt(row['answer'])
    if arm=='query_only':return q
    if arm in ('answer_only','answer_only_matched'):return a
    if arm=='query_answer':return 'USER QUERY: '+q+'\nVISIBLE MAS RESPONSE: '+a
    raise ValueError(arm)

def length_features(rows):
    return np.asarray([[math.log1p(len(str(x['query']))), math.log1p(len(str(x['answer']))),
                       math.log1p(len(str(x['answer']).split()))] for x in rows],dtype=float)

def record_id(row):
    return v510.tkey(row)

def source_rows(pools):
    seen={}; conflicts=[]
    for name,pairs in pools.items():
        for positive,negative in pairs:
            for r in (positive,negative):
                key=record_id(r)
                if key in seen and seen[key]['label']!=r['label']:
                    conflicts.append(key)
                else:
                    seen.setdefault(key,r)
    if conflicts:raise RuntimeError(f'Conflicting labels on {len(conflicts)} identical role/query/answer observations')
    return list(seen.values())

def config_rows(systems, roles):
    ret=[]
    for sid,(data, truth) in sorted(systems.items()):
        for role in roles:
            if role not in data:raise RuntimeError(f'Missing {role} for {sid}')
            ret.append(data[role])
    return ret

def group_counts(rows,roles):
    result={}
    for role in roles:
        sub=[r for r in rows if r['role']==role]
        result[role]={'positive':sum(r['label']==1 for r in sub), 'negative':sum(r['label']==0 for r in sub),
                      'distinct_systems':len({r['system_id'] for r in sub}),
                      'distinct_queries':len({str(r['query']).strip().lower() for r in sub})}
    return result

def prepare_sources():
    roles,desc,val,test,pools,stats=v510.obtain_sources()
    train=source_rows(pools)
    validation=config_rows(val,roles)
    testing=config_rows(test,roles)
    trk={record_id(r) for r in train}; vk={record_id(r) for r in validation}; tk={record_id(r) for r in testing}
    if trk&vk or trk&tk or vk&tk:raise RuntimeError('Identical input observation in training and validation/test')
    tr_ids={r['system_id'] for r in train}; v_ids=set(val); t_ids=set(test)
    if tr_ids&v_ids or tr_ids&t_ids or v_ids&t_ids:raise RuntimeError('SYSTEM ID SPLIT LEAK')
    def comp(groups):return {tuple(sorted(x[1])) for x in groups.values()}
    if comp(val)&comp(test):raise RuntimeError('VAL/TEST ROLE COMPOSITION LEAK')
    priority={record_id(r) for pos,neg in pools.get('matched_same_query',[]) for r in (pos,neg)}
    stat={'train_rows':len(train),'val_systems':len(val),'test_systems':len(test),
          'source_rows':dict(collections.Counter(r.get('source','unknown') for r in train)),
          'matched_priority_unique_rows':len(priority),
          'training_by_role':group_counts(train,roles),
          'val_by_role':group_counts(validation,roles),
          'test_by_role':group_counts(testing,roles),
          'note':'V137 DEV aggregate previously inspected; split is exploratory, not pristine external test',
          'v510_split_stats':stats}
    return roles,train,val,test,stat,priority

def fit_one(role,rows,arm,priority=None):
    sub=[r for r in rows if r['role']==role]
    y=np.asarray([r['label'] for r in sub],dtype=int)
    if min(int((y==0).sum()),int((y==1).sum()))<3:
        raise RuntimeError(f'{role}: <3 examples of either class; cannot train independent role verifier')
    if arm=='length_only':
        model=make_pipeline(StandardScaler(),LogisticRegression(C=.3,class_weight='balanced',solver='liblinear',random_state=SEED))
        X=length_features(sub)
    else:
        model=make_pipeline(TfidfVectorizer(analyzer='char',ngram_range=(3,5),min_df=2,
                       max_features=16000,sublinear_tf=True,dtype=np.float32),
                       LogisticRegression(C=1.0,class_weight='balanced',solver='liblinear',max_iter=600,random_state=SEED))
        X=[text(r,arm) for r in sub]
    # Deduplicated role-specific observations, all labels used only in optimizer.
    if arm=='answer_only_matched':
        # Same exact query across PRESENT/ABSENT background systems receives extra
        # weight. Labels and system identifiers are never included in model text.
        weights=np.asarray([3.0 if record_id(r) in (priority or set()) else 1.0 for r in sub],dtype=float)
        model.fit(X,y,logisticregression__sample_weight=weights)
    else:
        model.fit(X,y)
    return model

def predict(model,rows,arm):
    X=length_features(rows) if arm=='length_only' else [text(r,arm) for r in rows]
    return model.decision_function(X).astype(float).tolist()

def infer(models,rows,roles,arm):
    scores={}
    for role in roles:
        sub=[r for r in rows if r['role']==role]
        ss=predict(models[role],sub,arm)
        for r,s in zip(sub,ss):scores[(r['system_id'],role)]=float(s)
    return scores

def calc_metrics(scores,systems,roles,role_bias=None):
    vals={};truth={}
    role_bias=role_bias or {}
    for sid,(obs,pos) in systems.items():
        vals[sid]={r:scores[(sid,r)]-role_bias.get(r,0) for r in roles}
        truth[sid]=pos
    return v510.v57.auc20(vals,truth)

def within_role_auc(scores,systems,roles):
    details={}
    for role in roles:
        present=[scores[(sid,role)] for sid,(_,pos) in systems.items() if role in pos]
        absent=[scores[(sid,role)] for sid,(_,pos) in systems.items() if role not in pos]
        details[role]={'positive_systems':len(present),'negative_systems':len(absent),
            'binary_auc':round(sum((a>b)+.5*(a==b) for a in present for b in absent)/(len(present)*len(absent)),6)
                     if present and absent else None}
    valid=[x['binary_auc'] for x in details.values() if x['binary_auc'] is not None]
    return {'macro_role_presence_auc':round(statistics.mean(valid),6) if valid else None,
            'evaluated_roles':len(valid),'by_role':details}

def negative_centers(raw,val,roles):
    # Calibration uses only validation labels and only negative responses.
    # It does not train on test and does not imply calibrated probability.
    biases={}
    for role in roles:
        neg=[raw[(sid,role)] for sid,(_,pos) in val.items() if role not in pos]
        biases[role]=float(statistics.median(neg)) if neg else 0.0
    return biases

def eval_arm(models,arm,val,test,roles):
    rv=config_rows(val,roles);rt=config_rows(test,roles)
    vscore=infer(models,rv,roles,arm)
    centers=negative_centers(vscore,val,roles)
    # Choose between raw logits and negative-centered logits *on validation only*.
    val_raw=calc_metrics(vscore,val,roles)
    val_center=calc_metrics(vscore,val,roles,centers)
    def select(m):return .75*m['mean_within_mas_auc']+.25*(m['exact_top3_count']/len(val))
    calibrated=select(val_center)>select(val_raw)
    test_scores=infer(models,rt,roles,arm)
    test_result=calc_metrics(test_scores,test,roles,centers if calibrated else {})
    return {'validation_raw':val_raw,'validation_negative_centered':val_center,
            'selected_negative_centering':calibrated,'test':test_result,
            'test_per_role':within_role_auc(test_scores,test,roles),
            'biases_val_only':centers if calibrated else {r:0 for r in roles}},test_scores

def replay_existing_v58(models_by_arm, arm_results, roles, training, out):
    """Offline replay on already-paid V58 fresh probes; MAS composition seen in V56.

    Replay responses are never used to train or select the role-wise models.
    Exact re-use of an input observation is reported; no new MAS/API calls.
    """
    source=os.environ.get('V511_V58_ROOT','')
    if source:
        roots=[Path(source).resolve()]
    else:
        base=ROOT/'artifacts/role_direct_within_v4/adaptive_v58'
        roots=sorted(base.glob('mas5_*'),reverse=True) if base.exists() else []
    roots=[r for r in roots if (r/'public_preflight.json').is_file()]
    if not roots:return {'status':'NO_EXISTING_V58_DATA'}
    root=roots[0]
    pf=json.loads((root/'public_preflight.json').read_text())
    sid=pf['session_id']
    if set(pf['roles'])!=set(roles):return {'status':'ROLE_CATALOG_MISMATCH','root':str(root)}
    manifest=v510.v57.load(v510.v57.MANIFEST)
    found=[x for x in manifest['systems'] if x.get('split')=='dev' and x['session_id']==sid]
    if len(found)!=1:return {'status':'MISSING_EVALUATION_SESSION','session_id':sid}
    truth=set(found[0]['roles'])
    trained={record_id(x) for x in training}
    saved=[];rounds=[];role_history={arm:collections.defaultdict(list) for arm in models_by_arm}
    for rd in range(1,9):
        p=root/f'responses_public_dev_round_{rd:02}.jsonl'
        if not p.exists():break
        old=v510.v57.lines(p)
        if len(old)!=20 or {x['target_role'] for x in old}!=set(roles):
            return {'status':'INCOMPLETE_ROUND','round':rd,'root':str(root)}
        obs=[]
        for x in old:
            if x['session_id']!=sid or x.get('execution_status')!='ok':
                raise RuntimeError('Bad V58 replay response')
            obs.append({'system_id':sid,'role':x['target_role'],'query':x['query'],
                        'answer':x['final_output'],'label':int(x['target_role'] in truth),
                        'source':'v58_replay_not_trained'})
        duplicates=sum(record_id(x) in trained for x in obs)
        r={'round':rd,'exact_train_observation_overlaps':duplicates,'models':{}}
        for arm,models in models_by_arm.items():
            score=infer(models,obs,roles,arm)
            bias=arm_results[arm]['biases_val_only']
            for role in roles:role_history[arm][role].append(score[(sid,role)]-bias[role])
            vals={role:statistics.mean(role_history[arm][role]) for role in roles}
            pos=[vals[role] for role in roles if role in truth];neg=[vals[role] for role in roles if role not in truth]
            auc=sum((p>n)+.5*(p==n) for p in pos for n in neg)/51
            top3=sorted(roles,key=lambda t:(-vals[t],t))[:3]
            r['models'][arm]={'within_mas_auc':round(auc,6),
                    'top3':top3,'top3_correct':len(set(top3)&truth),
                    'exact_top3':set(top3)==truth}
        rounds.append(r)
        print('V511_REPLAY_V58',rd,'/',8,
            'answer_only_auc',r['models']['answer_only']['within_mas_auc'],
            'query_only_auc',r['models']['query_only']['within_mas_auc'],flush=True)
    result={'status':'REPLAY_COMPLETE' if rounds else 'NO_COMPLETED_ROUNDS',
        'source_v58':str(root),'session_id':sid,'rounds':rounds,
        'truth_eval_only':sorted(truth),'no_MAS_calls':True,
        'V58_MAS_COMPOSITION_WAS_ALREADY_SEEN_IN_V56_TRAINING':True,
        'only_new_questions_and_answers_are_replay_evidence':True,
        'not_a_cross_MAS_generalization_test':True}
    dump(out/'v58_offline_replay.json',result)
    return {'status':result['status'],'rounds':len(rounds),'source':str(root)}

def run(out):
    out=Path(out).resolve();expected=BASE.resolve()
    if out.parent!=expected or not out.name.startswith('rolewise_'):
        raise RuntimeError('Output must be a new rolewise_* directory directly under sensor_v511')
    if out.exists():raise RuntimeError(f'Refusing to overwrite {out}')
    roles,train,val,test,stat,priority=prepare_sources()
    out.mkdir(parents=True)
    dump(out/'preflight.json',{'status':'SPLIT_VERIFIED','dataset':stat,'no_mas_calls':True,
        'no_gpu':True,'test_not_used_for_model_or_arm_selection':True,
        'validation_compositions_disjoint_from_test':True,
        'label_only_in_optimizer_and_evaluation':True})
    print('V511_DATA',json.dumps({'train':len(train),'val':len(val),'test':len(test),'roles':len(roles)}),flush=True)
    results={};allmodels={}
    for arm in ARMS:
        print('V511_ARM_START',arm,flush=True)
        models={}
        for idx,role in enumerate(roles,1):
            models[role]=fit_one(role,train,arm,priority)
            print('V511_TRAINED',arm,idx,'/',len(roles),role,flush=True)
        result,test_scores=eval_arm(models,arm,val,test,roles)
        dump(out/f'metrics_{arm}.json',result)
        joblib.dump({'arm':arm,'models':models,'roles':roles},out/f'model_{arm}.joblib',compress=3)
        results[arm]=result
        allmodels[arm]=models
        print('V511_EVAL',arm,json.dumps({'val_raw_auc':result['validation_raw']['mean_within_mas_auc'],
            'test_auc':result['test']['mean_within_mas_auc'],
            'test_exact_top3':result['test']['exact_top3_count'],
            'mean_binary_auc':result['test_per_role']['macro_role_presence_auc']}),flush=True)
    # Predeclared primary hypothesis: response-only rolewise detector vs query-only shortcut.
    main=results['answer_only']['test'];control=results['query_only']['test']
    # Select *among response-based variants* using validation only. Test is then
    # evaluated exactly once for each arm; its labels never drive selection.
    response_arms=('answer_only','answer_only_matched','query_answer')
    def val_selection(a):
        r=results[a];m=r['validation_negative_centered'] if r['selected_negative_centering'] else r['validation_raw']
        return (.75*m['mean_within_mas_auc']+.25*m['exact_top3_count']/len(val),-response_arms.index(a))
    selected=max(response_arms,key=val_selection)
    conclusion={'answer_only_test_auc':main['mean_within_mas_auc'],
        'query_only_test_auc':control['mean_within_mas_auc'],
        'answer_minus_query_only_auc':round(main['mean_within_mas_auc']-control['mean_within_mas_auc'],6),
        'answer_only_exact_top3':main['exact_top3_count'], 'query_only_exact_top3':control['exact_top3_count'],
        'selected_response_arm_by_validation_only':selected,
        'selected_arm_test_auc':results[selected]['test']['mean_within_mas_auc'],
        'selected_arm_test_exact_top3':results[selected]['test']['exact_top3_count'],
        'strong_signal_not_established_if_query_only_matches_answer':True,
        'not_evidence_that_generator_has_99_percent_presence_discrimination':True,
        'test_is_composition_disjoint_V137_DEV_but_prior_DEV_aggregate_was_inspected':True,
        'per_role_probability_calibrated':False,
        'V57_training_checkpoint_not_changed':True,'no_mas_calls':True}
    # Optional, existing data only: score previously paid adaptive V58 responses.
    try:
        replay=replay_existing_v58(allmodels,results,roles,train,out)
    except Exception as exc:
        replay={'status':'REPLAY_SKIPPED_ERROR','error':str(exc)}
    dump(out/'report.json',{'status':'FINISHED','primary':conclusion,'arms':results,'dataset':stat,'v58_replay':replay})
    with (out/'comparison.tsv').open('w',newline='',encoding='utf8') as f:
        w=csv.writer(f,delimiter='\t');w.writerow(['arm','validation_auc_raw','test_auc','test_exact_top3','test_systems','test_f1_at3','test_macro_per_role_binary_auc','negative_centered'])
        for arm,x in results.items():
            t=x['test'];w.writerow([arm,x['validation_raw']['mean_within_mas_auc'],t['mean_within_mas_auc'],t['exact_top3_count'],len(test),t['mean_role_f1_at3'],x['test_per_role']['macro_role_presence_auc'],x['selected_negative_centering']])
    print('V511_FINISHED',json.dumps(conclusion),flush=True)
    print('REPORT',out/'comparison.tsv',flush=True)

def selftest():
    roles=['r1','r2'];raw=[]
    for role in roles:
        for i in range(12):
            lab=int(i%3==0)
            raw.append({'role':role,'system_id':f's{i}','query':'Help me investigate this problem',
                        'answer':('Use role-specific expertise and detailed protocol' if lab else 'Generic general advice without specialist depth')+f' observation {i%2}',
                        'label':lab})
    models={r:fit_one(r,raw,'answer_only') for r in roles}
    scored=infer(models,raw,roles,'answer_only')
    assert len(scored)==24 and all(np.isfinite(x) for x in scored.values())
    sysdata={f's{i}':({r:next(x for x in raw if x['system_id']==f's{i}' and x['role']==r) for r in roles},
                       {'r1'} if i%2 else {'r2'}) for i in range(12)}
    ac=within_role_auc(scored,sysdata,roles)
    assert ac['evaluated_roles']==2
    # Check negative calibration cannot read held-out test rows.
    assert len(negative_centers(scored,dict(list(sysdata.items())[:6]),roles))==2
    assert text(raw[0],'query_only')!='' and text(raw[0],'answer_only')!=''
    print('V511_SELFTEST_OK (synthetic only)')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['selftest','run']);ap.add_argument('--out',type=Path)
    x=ap.parse_args()
    if x.stage=='selftest':selftest()
    else:
        if x.out is None:raise RuntimeError('--out is required')
        run(x.out)
if __name__=='__main__':main()
