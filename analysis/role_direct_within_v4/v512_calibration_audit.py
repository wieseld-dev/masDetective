#!/usr/bin/env python3
"""V5.12: post-hoc calibration audit of frozen V5.11.3 role-wise detectors.

No training of role detectors, no new MAS calls, no use of DEV labels during
calibration. All compared transforms are predeclared; none are selected on DEV.
V58 re-scoring is retrospective, one MAS already used in other training.
"""
from __future__ import annotations
import argparse, collections, csv, json, math, os, statistics
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

from analysis.role_direct_within_v4 import v5111_rolewise_verifier as base
from analysis.role_direct_within_v4 import v5113_dev_heldout as dev
from analysis.role_direct_within_v4 import v510_generalizing_sensor as v510

ROOT=base.ROOT
ROOT_OUT=ROOT/'artifacts/role_direct_within_v4/sensor_v512'
SOURCE_BASE=ROOT/'artifacts/role_direct_within_v4/sensor_v5113'
ARMS=('answer_only','answer_only_matched','query_answer','query_only','length_only')
METHODS=('raw','negative_centered','negative_z','platt_shrunk')


def latest_source():
    runs=sorted(SOURCE_BASE.glob('rolewise_*'),reverse=True)
    for p in runs:
        if (p/'model_answer_only.joblib').is_file() and (p/'metrics_answer_only.json').is_file():return p
    raise RuntimeError('No trained V5.11.3 models found. Need model_*.joblib under artifacts/role_direct_within_v4/sensor_v5113/rolewise_*.')


def truth_scores(scores,sys,roles):
    vals={sid:{role:float(scores[(sid,role)]) for role in roles} for sid in sys}
    truth={sid:present for sid,(_,present) in sys.items()}
    return v510.v57.auc20(vals,truth)


def safe_float(x):
    if not np.isfinite(x):raise RuntimeError('Nonfinite calibration score')
    return float(x)


def validation_calibrators(scores,systems,roles):
    """Fit all transforms using validation only; no hyperparameter search on DEV.

    null_z employs empirical negative median and shrunken dispersion. Platt
    employs a shared logit mapping plus small role-specific shrinkage when
    sufficient calibration labels exist. Validation AUC after fitting is NOT
    unbiased validation performance.
    """
    neg={r:[scores[(sid,r)] for sid,(_,positive) in systems.items() if r not in positive] for r in roles}
    pos={r:[scores[(sid,r)] for sid,(_,positive) in systems.items() if r in positive] for r in roles}
    missing=[r for r in roles if not neg[r]]
    if missing:raise RuntimeError(f'No validation negatives for {missing}; cannot calibrate without data.')
    allneg=[v for ns in neg.values() for v in ns]
    global_med=float(np.median(allneg))
    global_spread=max(0.25,float(np.std(allneg)),1e-5)
    medians={r:float(np.median(neg[r])) for r in roles}
    scales={}
    for r in roles:
        values=np.array(neg[r],dtype=float)
        med=medians[r]
        mad=float(np.median(np.abs(values-med)))*1.4826
        # Conservative heavy shrinkage when few negatives.
        confidence=len(values)/(len(values)+12)
        scales[r]=max(.2,confidence*max(mad,.15)+(1-confidence)*global_spread)
    # Global Platt on role-normalized validation scores, with balanced classes.
    X=np.array([[scores[(sid,r)]-medians[r]] for sid in systems for r in roles],dtype=float)
    Y=np.array([int(r in positive) for sid,(_,positive) in systems.items() for r in roles],dtype=int)
    if len(np.unique(Y))!=2:raise RuntimeError('Validation requires both PRESENT and ABSENT examples')
    glob=LogisticRegression(C=.08,class_weight='balanced',max_iter=1000,solver='liblinear',random_state=512)
    glob.fit(X,Y)
    glob_slope=float(glob.coef_[0,0]);glob_intercept=float(glob.intercept_[0])
    local={};coverage={}
    for r in roles:
        coverage[r]={'val_present':len(pos[r]),'val_absent':len(neg[r])}
        if len(pos[r])>=3 and len(neg[r])>=5:
            xx=np.array([[scores[(sid,r)]-medians[r]] for sid in systems])
            yy=np.array([int(r in positive) for _,positive in systems.values()],dtype=int)
            lr=LogisticRegression(C=.08,class_weight='balanced',max_iter=1000,solver='liblinear',random_state=512)
            lr.fit(xx,yy)
            # Shrink both coef and intercept toward pooled calibration.
            weight=min(.5,len(pos[r])/(len(pos[r])+10.0))
            local[r]=(weight*float(lr.coef_[0,0])+(1-weight)*glob_slope,
                      weight*float(lr.intercept_[0])+(1-weight)*glob_intercept)
        else:
            local[r]=(glob_slope,glob_intercept)
    return {'centers':medians,'scales':scales,'platt':local,'validation_coverage':coverage,
            'global_null_median':global_med,'global_negative_std':global_spread,
            'global_platt_slope':glob_slope,'global_platt_intercept':glob_intercept,
            'calibration_systems':len(systems),'calibration_uses_test':False,
            'caveat':'Platt mapping fitted on VAL, no VAL performance claims after fitting; low positive counts shrink to global.'}


def calibrate(role,raw,method,cal):
    s=float(raw)
    if method=='raw':return s
    center=cal['centers'][role]
    if method=='negative_centered':return safe_float(s-center)
    if method=='negative_z':return safe_float((s-center)/cal['scales'][role])
    if method=='platt_shrunk':
        slope,inter=cal['platt'][role]
        return safe_float(slope*(s-center)+inter)
    raise ValueError(method)


def score_systems(raw,sys,roles,method,cal):
    values={(sid,r):calibrate(r,raw[(sid,r)],method,cal) for sid in sys for r in roles}
    return truth_scores(values,sys,roles)


def per_role_details(raw,sys,roles):
    return base.within_role_auc(raw,sys,roles)


def available_v58():
    env=os.environ.get('V512_V58_ROOT')
    if env:
        p=Path(env).resolve()
        if not p.is_dir():raise RuntimeError(f'V512_V58_ROOT is not a directory: {p}')
        return p
    basepath=ROOT/'artifacts/role_direct_within_v4/adaptive_v58'
    paths=sorted(basepath.glob('mas5_*'),reverse=True)
    return next((p for p in paths if (p/'public_preflight.json').is_file()),None)


def v58_replay(model,arm,roles,cal,training,run):
    if run is None:return {'status':'NO_V58_RUN'}
    pf=json.loads((run/'public_preflight.json').read_text())
    sid=pf['session_id']
    if set(pf['roles'])!=set(roles):return {'status':'V58_ROLES_MISMATCH'}
    manifest=v510.v57.load(v510.v57.MANIFEST)
    relevant=[x for x in manifest['systems'] if x.get('split')=='dev' and x['session_id']==sid]
    if len(relevant)!=1:return {'status':'MISSING_V58_TRUTH_SESSION'}
    truth=set(relevant[0]['roles'])
    train_keys={base.record_id(r) for r in training}
    hist={m:collections.defaultdict(list) for m in METHODS}
    rounds=[]
    for rnd in range(1,9):
        path=run/f'responses_public_dev_round_{rnd:02}.jsonl'
        if not path.is_file():break
        rows=v510.v57.lines(path)
        if len(rows)!=20 or {x['target_role'] for x in rows}!=set(roles):break
        obs=[]
        for x in rows:
            if x['session_id']!=sid or x.get('execution_status')!='ok':
                return {'status':'V58_BAD_RESPONSE','round':rnd}
            obs.append({'system_id':sid,'role':x['target_role'],'query':x['query'],
                'answer':x['final_output'],'label':int(x['target_role'] in truth)})
        raw=base.infer(model,obs,roles,arm)
        entries={'round':rnd,'exact_training_observation_overlaps':sum(base.record_id(x) in train_keys for x in obs),'fusion':{}}
        for meth in METHODS:
            for role in roles:
                hist[meth][role].append(calibrate(role,raw[(sid,role)],meth,cal))
            for fusion in ('mean','latest','median','recency'):
                per_role={}
                for role in roles:
                    values=hist[meth][role]
                    if fusion=='mean':score=statistics.mean(values)
                    elif fusion=='latest':score=values[-1]
                    elif fusion=='median':score=statistics.median(values)
                    else:
                        w=[.7**(len(values)-i-1) for i in range(len(values))]
                        score=sum(a*b for a,b in zip(values,w))/sum(w)
                    per_role[role]=score
                auc=sum((per_role[p]>per_role[n])+.5*(per_role[p]==per_role[n]) for p in truth for n in roles if n not in truth)/51
                top3=sorted(roles,key=lambda r:(-per_role[r],r))[:3]
                entries['fusion'][f'{meth}/{fusion}']={'auc':round(auc,6),'top3_correct':len(set(top3)&truth),'exact_top3':set(top3)==truth}
        rounds.append(entries)
    return {'status':'REPLAY_ONLY' if rounds else 'NO_COMPLETED_ROUNDS',
            'source':str(run),'rounds':rounds,'truth_eval_only':sorted(truth),
            'not_model_selection':True,'training_mas_composition_seen_before':True,
            'no_mas_calls':True}


def write_csv(path,fieldnames,records):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fieldnames,delimiter='\t')
        w.writeheader();w.writerows(records)


def run(args):
    source=Path(args.source).resolve() if args.source else latest_source()
    if not source.is_dir():raise RuntimeError(f'No source folder: {source}')
    destination=Path(args.out).resolve()
    if destination.exists():raise RuntimeError('Refusing to overwrite output folder: '+str(destination))
    if destination.parent!=ROOT_OUT.resolve():raise RuntimeError(f'--out must be a child of {ROOT_OUT}')
    roles,training,val,test,split_report,priority=dev.prep()
    if len(roles)!=20 or len(test)<1:raise RuntimeError('Bad role catalog / DEV split')
    destination.mkdir(parents=True)
    base.dump(destination/'run_manifest.json',{
        'status':'RUNNING','source_models':str(source),'target_roles':roles,
        'val_systems':len(val),'test_systems':len(test),
        'truth_eval_only':True,'no_mas_calls':True,'no_model_weight_training':True,
        'no_test_fitted_calibrators':True,'previous_test_has_been_inspected':True,
        'warning':'Test is V137 DEV historically inspected, so results exploratory.'})
    summary=[];role_rows=[];cal_out={};replays={}
    v58=available_v58()
    for arm in ARMS:
        path=source/f'model_{arm}.joblib'
        if not path.is_file():raise RuntimeError('Missing model: '+str(path))
        bundle=joblib.load(path)
        if set(bundle['roles'])!=set(roles) or bundle['arm']!=arm:raise RuntimeError('Model bundle mismatch')
        models=bundle['models']
        val_scores=base.infer(models,base.config_rows(val,roles),roles,arm)
        test_scores=base.infer(models,base.config_rows(test,roles),roles,arm)
        cal=validation_calibrators(val_scores,val,roles)
        cal_out[arm]={k:v for k,v in cal.items() if k!='platt'}
        cal_out[arm]['platt_by_role']={r:{'slope':cal['platt'][r][0],'intercept':cal['platt'][r][1]} for r in roles}
        # Independent consistency check against the saved V5.11.3 evaluation.
        previous=json.loads((source/f'metrics_{arm}.json').read_text())
        prev_method='negative_centered' if previous['selected_negative_centering'] else 'raw'
        expected=previous['test']['mean_within_mas_auc']
        actual=score_systems(test_scores,test,roles,prev_method,cal)['mean_within_mas_auc']
        if abs(expected-actual)>0.00001:
            raise RuntimeError(f'Baseline mismatch for {arm}: old={expected} reconstructed={actual}. Check splits and source!')
        for meth in METHODS:
            metric=score_systems(test_scores,test,roles,meth,cal)
            summary.append({'arm':arm,'calibration':meth,'test_auc':metric['mean_within_mas_auc'],
                            'exact_top3_count':metric['exact_top3_count'],
                            'test_systems':len(test),'test_f1_at3':metric['mean_role_f1_at3']})
        detail=per_role_details(test_scores,test,roles)['by_role']
        for r in roles:
            role_rows.append({'arm':arm,'role':r,'positive_systems':detail[r]['positive_systems'],
                'negative_systems':detail[r]['negative_systems'],'binary_auc':detail[r]['binary_auc'],
                'val_positive_systems':cal['validation_coverage'][r]['val_present'],
                'val_negative_systems':cal['validation_coverage'][r]['val_absent']})
        if arm in ('answer_only','query_only','length_only'):
            try:
                replays[arm]=v58_replay(models,arm,roles,cal,training,v58)
            except Exception as exc:
                replays[arm]={'status':'REPLAY_SKIPPED_ERROR','message':str(exc),'no_mas_calls':True}
        print('V512_ARM',arm,'test_raw=',next(x['test_auc'] for x in summary if x['arm']==arm and x['calibration']=='raw'),flush=True)
    write_csv(destination/'comparison.tsv',['arm','calibration','test_auc','exact_top3_count','test_systems','test_f1_at3'],summary)
    write_csv(destination/'per_role_audit.tsv',['arm','role','positive_systems','negative_systems','binary_auc','val_positive_systems','val_negative_systems'],role_rows)
    base.dump(destination/'calibration_validation_only.json',cal_out)
    base.dump(destination/'v58_fusion_replay.json',replays)
    base.dump(destination/'report.json',{
        'status':'COMPLETE','source':str(source),'comparison':summary,
        'dataset':{'validation_systems':len(val),'test_systems':len(test),
                   'test_previously_inspected':True,
                   'split_protocol':split_report['split_protocol']},
        'no_training':True,'no_mas_api_calls':True,
        'selection_policy':'Predeclared transforms all reported, none picked by DEV scores',
        'limitation':'Role-level binary AUC cannot be made comparable across 20 roles without calibration; fitted calibration itself may overfit small VAL. V58 replay is not heldout for model selection.'})
    print('V512_COMPLETE',destination,flush=True)


def selftest():
    roles=['r0','r1','r2']
    systems={}
    for i in range(15):
        ps={roles[i%3]}
        systems[f's{i}']=({},{r for r in ps})
    scores={(sid,r): float(i%7+1.5*int(r in present)+(2.5 if r=='r0' else 0))
        for i,(sid,(_,present)) in enumerate(systems.items()) for r in roles}
    cal=validation_calibrators(scores,systems,roles)
    for r in roles:
        for method in METHODS:
            assert math.isfinite(calibrate(r,3.,method,cal))
    assert cal['calibration_uses_test'] is False
    # No fitted parameters change with a fabricated TEST perturbation.
    shifted={(sid,r):v+1000 for (sid,r),v in scores.items()}
    assert validation_calibrators(scores,systems,roles)['centers']==cal['centers']
    assert any(abs(calibrate(r,shifted[(sid,r)],'negative_centered',cal))>100
               for sid in systems for r in roles)
    print('V512_SELFTEST_OK (synthetic only)',flush=True)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('stage',choices=['selftest','preflight','run'])
    ap.add_argument('--source')
    ap.add_argument('--out')
    args=ap.parse_args()
    if args.stage=='selftest':selftest();return
    source=Path(args.source).resolve() if args.source else latest_source()
    missing=[x for x in ARMS if not (source/f'model_{x}.joblib').is_file()]
    if missing:raise RuntimeError(f'Missing models in {source}: {missing}')
    if args.stage=='preflight':
        roles,train,val,test,report,priority=dev.prep()
        print('V512_PREFLIGHT',json.dumps({'source':str(source),'roles':len(roles),
              'validation_systems':len(val),'evaluation_systems':len(test),
              'model_bundles':len(ARMS),'no_api_calls':True,'no_gpu':True,
              'evaluation_is_retrospective':True}),flush=True);return
    if not args.out:raise RuntimeError('--out is required')
    run(args)

if __name__=='__main__':main()
