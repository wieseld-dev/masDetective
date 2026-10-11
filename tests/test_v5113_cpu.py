#!/usr/bin/env python3
"""CPU-only synthetic fixture; does not read paid data or contact any API."""
from __future__ import annotations
import collections
from analysis.role_direct_within_v4 import v5113_dev_heldout as impl
from analysis.role_direct_within_v4 import v5111_rolewise_verifier as prev


def fixture():
    roles=[f'r{k:02}' for k in range(20)]
    train_systems={}
    for i in range(50):
        sid=f'historical_{i:03}'
        positives={roles[(7*i+j)%20] for j in range(3)}
        train_systems[sid]=({role:{'system_id':sid,'role':role,'query':f'Historical probe {i} for {role}?',
                     'answer':f'Historical answer {i} {role}', 'label':int(role in positives), 'source':'historical'}
                    for role in roles},positives)
    dev={}
    for i in range(16):
        sid=f'dev_{i:03}'
        positives={roles[(i+2)%20],roles[(i+10)%20],roles[(i+16)%20]}
        observations={role:{'system_id':sid,'role':role,'query':f'Dev question {i} for {role}?',
                       'answer':f'Dev answer {i} {role}', 'label':int(role in positives),'source':'dev'}
                      for role in roles}
        # All DEV systems connected through same role/query/answer -- original
        # V5112 component partition cannot separate them. Our new split needn't.
        observations['r00']['query']='SHARED DEV QUESTION'
        observations['r00']['answer']='SHARED DEV ANSWER'
        dev[sid]=(observations,positives)
    # Contaminate historical examples with an actual DEV observation key.
    train_systems['historical_000'][0]['r19']['query']=dev['dev_000'][0]['r19']['query']
    train_systems['historical_000'][0]['r19']['answer']=dev['dev_000'][0]['r19']['answer']
    p=[]
    for obs,_ in train_systems.values():
        positive=[r for r in obs.values() if r['label']]
        negative=[r for r in obs.values() if not r['label']]
        p.extend((pos,neg) for pos in positive for neg in negative)
    return roles,train_systems,dev,{'hist_within_system':p,'hist_cross_same_role':[],
                             'matched_same_query':[], 'mas5_within_system':[]}


def main():
    roles,train_sys,dev,pools=fixture()
    oldval=dict(list(dev.items())[:8]);oldtest=dict(list(dev.items())[8:])
    training,conf=prev.source_rows(pools,return_audit=True)
    # Existing V5112 must fail: DEV all connected (we're not running it here).
    test=dict(dev)
    hist,invalid=impl.data_for_validation(pools,roles)
    assert len(hist)==50 and not invalid
    tr, val, selection=impl.choose_train_validation(roles,training,hist,test)
    assert len(val)>=5 and len(tr)>500
    assert not {prev.record_id(r) for r in tr} & impl.unique_keys(val)
    # This helper only separates TRAIN from VAL; prep() separately quarantines DEV.
    override=(roles,{},oldval,oldtest,pools,{'fixture':True})
    rs,tr,val,test,report,priority=impl.prep(override)
    assert len(test)==16 and len(val)>=5 and len(tr)>500
    assert not any(report['train_val_dev_exact_overlap'].values())
    assert report['test_macro_per_role_possible']>=10
    assert report['status']=='SPLIT_VERIFIED'
    print('V5113_SELFTEST_OK: intact connected DEV evaluation, independent TRAIN validation, no duplicate leakage')

if __name__=='__main__':main()
