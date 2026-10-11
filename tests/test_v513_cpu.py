#!/usr/bin/env python3
"""Offline synthetic E2E smoke: validates TSV generation, old-run parity and leakage guard.

Run with existing V510-V512 Python modules importable (local synthetic only).
"""
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import joblib

from analysis.role_direct_within_v4 import v513_role_error_attribution as m


def synthetic():
    roles=[f'role_{i:02d}' for i in range(20)]
    def row(sid,role,rnd,truth):
        present=role in truth
        return {'system_id':sid,'role':role,
                'query':f'Question about {role} in {sid}, round {rnd}',
                'answer':f'Observed signal {"SPECIALIST" if present else "GENERAL"}, sample {sid} round {rnd}',
                'label':int(present)}
    def system(i):
        sid=f'sys_{i:02d}'
        truth={roles[(i+j*5)%20] for j in range(3)}
        return sid,({r:row(sid,r,0,truth) for r in roles},truth)
    val=dict(system(i) for i in range(0,14))
    test=dict(system(i) for i in range(14,25))
    sid='v58_sample';truth={roles[i] for i in (1,5,13)}
    rounds=[[row(sid,r,rnd,truth) for r in roles] for rnd in range(1,4)]
    train=[row('train_unique',r,100,set()) for r in roles]
    def infer(models,observations,roles,arm):
        ret={}
        for x in observations:
            r=x['role'];ri=int(r.split('_')[1]);base=ri*.055
            if arm=='answer_only': base+=(.6 if 'SPECIALIST' in x['answer'] else 0)+len(x['answer'])*.001
            elif arm=='query_only':base+=(len(x['query'])%7)*.015
            else:base+=len(x['answer'])*.004
            ret[(x['system_id'],r)]=base
        return ret
    return roles,train,val,test,sid,truth,rounds,infer


def main():
    m.selftest()
    roles,train,val,test,sid,truth,rounds,infer=synthetic()
    with tempfile.TemporaryDirectory(prefix='v513_e2e_') as temp:
        path=Path(temp);source=path/'models';source.mkdir()
        refdir=path/'v512';refdir.mkdir()
        v58source=path/'v58';v58source.mkdir()
        report={'comparison':[],'source':str(source),'status':'COMPLETE'}
        old={};frozen_cal={}
        for arm in m.ARMS:
            joblib.dump({'arm':arm,'roles':roles,'models':{r:None for r in roles}},source/f'model_{arm}.joblib')
            raw_val=infer(None,m.base.config_rows(val,roles),roles,arm)
            raw_test=infer(None,m.base.config_rows(test,roles),roles,arm)
            cal=m.prev.validation_calibrators(raw_val,val,roles)
            frozen_cal[arm]={k:v for k,v in cal.items() if k!='platt'}
            for meth in m.TRANSFORMS:
                avg=m.prev.score_systems(raw_test,test,roles,meth,cal)['mean_within_mas_auc']
                report['comparison'].append({'arm':arm,'calibration':meth,'test_auc':avg})
            history={r:[] for r in roles}
            oldrounds=[]
            for idx,entries in enumerate(rounds,1):
                scores=infer(None,entries,roles,arm)
                for r in roles:history[r].append(scores[(sid,r)])
                fused={}
                for method in m.TRANSFORMS:
                    per={r:sum(m.prev.calibrate(r,s,method,cal) for s in history[r])/len(history[r]) for r in roles}
                    item=m.summary_row(idx,arm,method,per,truth,roles)
                    fused[method+'/mean']={'auc':item['auc'],'top3_correct':item['top3_correct'],
                                          'exact_top3':item['exact_top3']}
                oldrounds.append({'round':idx,'exact_training_observation_overlaps':0,'fusion':fused})
            old[arm]={'rounds':oldrounds,'source':str(v58source)}
        (refdir/'calibration_validation_only.json').write_text(json.dumps(frozen_cal))
        outputroot=path/'v513';outputroot.mkdir()
        out=outputroot/'audit_1'
        with patch.object(m.dev,'prep',lambda:(roles,train,val,test,{},None)), \
             patch.object(m.base,'infer',infer), \
             patch.object(m,'load_v512_baseline',lambda:(refdir,report,old)), \
             patch.object(m,'get_v58_rows',lambda roles:(v58source,sid,truth,rounds)), \
             patch.object(m,'OUTBASE',outputroot):
            m.run(source,out)
        assert json.loads((out/'report.json').read_text())['status']=='COMPLETE'
        assert len((out/'v58_role_scores.tsv').read_text().splitlines())==1+3*20
        assert len((out/'v58_rankings.tsv').read_text().splitlines())==1+3*20*2
        assert len((out/'v137_role_predictions.tsv').read_text().splitlines())==1+len(test)*20
        assert 'role_01' in (out/'role_rankings_report.md').read_text()
        print('V513_SYNTHETIC_E2E_PASS 20 roles, 3 rounds, 11 test systems, reproduced baseline',flush=True)


if __name__=='__main__':main()
