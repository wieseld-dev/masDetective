#!/usr/bin/env python3
"""End-to-end V511 dry run using synthetic full 20-role MAS data (no real data)."""
import json, tempfile
from pathlib import Path
from analysis.role_direct_within_v4 import v511_rolewise_verifier as v

def fixture():
    roles=[f'role_{i:02d}' for i in range(20)]
    systems=[]
    for i in range(70):
        pos={roles[(i*3+j)%20] for j in range(3)}
        rows={}
        for r in roles:
            # Equal user queries across labels for a role, to test query-only control.
            rows[r]={'role':r,'system_id':f'mas_{i:03d}','query':f'Please analyze a realistic {r} task.',
                'answer':f'{r} specialist tools workflow evidence procedure traceable observations '+ ('advanced validation checks and rigorous analysis' if r in pos else 'general assistance and tentative suggestions')+f' variant{i%4}',
                'label':int(r in pos),'source':'synthetic_mock'}
        systems.append((rows,pos))
    train=[r for obs,_ in systems[:46] for r in obs.values()]
    val={f'mas_{i:03d}':systems[i] for i in range(46,58)}
    test={f'mas_{i:03d}':systems[i] for i in range(58,70)}
    priority={v.record_id(r) for r in train if r['label']==1}
    return roles,train,val,test,{'train_rows':len(train),'training_by_role':v.group_counts(train,roles)},priority

def main():
    with tempfile.TemporaryDirectory() as t:
        old=(v.BASE,v.prepare_sources)
        try:
            v.BASE=Path(t)
            v.prepare_sources=fixture
            v.run(Path(t)/'rolewise_mock_test')
            report=json.loads((Path(t)/'rolewise_mock_test/report.json').read_text())
            assert report['status']=='FINISHED'
            assert len(report['arms'])==5
            assert report['arms']['answer_only']['test']['mean_within_mas_auc']>=0
            assert (Path(t)/'rolewise_mock_test/model_answer_only.joblib').is_file()
            print('V511_END_TO_END_OK (20 roles, 12 validation MAS, 12 test MAS; all synthetic)')
        finally:
            v.BASE,v.prepare_sources=old

if __name__=='__main__':main()
