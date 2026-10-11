"""No-data CPU tests: splits, leakage fail-closed, and real v5111 training code integration."""
from pathlib import Path
import tempfile, json
from analysis.role_direct_within_v4 import v5112_overlap_safe as v
from analysis.role_direct_within_v4 import v5111_rolewise_verifier as old
def fixture():
    roles=[f'role_{i:02d}' for i in range(20)]
    systems=[]
    for i in range(70):
        pos={roles[(i*3+j)%20] for j in range(3)}
        rows={}
        for r in roles:
            rows[r]={'role':r,'system_id':f'mas_{i:03d}','query':f'Please analyze {r} task scenario{i}',
                     'answer':f'{r} specialist workflow observations '+('detailed technical validation' if r in pos else 'general advice')+f' example{i%4}',
                     'label':int(r in pos),'source':'synthetic_mock'}
        systems.append((rows,pos))
    train=[r for obs,_ in systems[:46] for r in obs.values()]
    val={f'mas_{i:03d}':systems[i] for i in range(46,58)}
    test={f'mas_{i:03d}':systems[i] for i in range(58,70)}
    priority={old.record_id(r) for r in train if r['label']==1}
    return roles,train,val,test,{'train_rows':len(train),'training_by_role':old.group_counts(train,roles)},priority

def main():
    v.selftest()
    roles,train,val,test,stat,priority=fixture()
    assert not v.overlap_stats(train,val,test)['train_test_unique']
    with tempfile.TemporaryDirectory() as temp:
        old_base,old_prep=old.BASE,old.prepare_sources
        try:
            old.BASE=Path(temp)
            old.prepare_sources=lambda:(roles,train,val,test,stat,priority)
            old.run(Path(temp)/'rolewise_v5112_synthetic')
            result=json.loads((Path(temp)/'rolewise_v5112_synthetic/report.json').read_text())
            assert result['status']=='FINISHED'
            assert len(result['arms'])==5
        finally:
            old.BASE,old.prepare_sources=old_base,old_prep
    print('V5112_END_TO_END_OK: 20 roles, 12 VAL, 12 TEST, CPU mock')
if __name__=='__main__':main()
