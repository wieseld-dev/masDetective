import json, tempfile, sys
from pathlib import Path
from analysis.role_direct_within_v4 import v56_nonadaptive as v56
from analysis.role_direct_within_v4 import v561_live_auc as v561

work=Path(tempfile.mkdtemp(prefix='mas5_v561_mock_'))
v56.PROJECT=work
v56.SIMPLE=work/'artifacts/role_direct_routing_pilot_v1/manifest_privileged.json'
v56.SIMPLE.parent.mkdir(parents=True)
roles=[f'role{i:02d}' for i in range(20)]
systems=[{'session_id':f'mas{i}','split':'dev','roles':[roles[(3*i+j)%20] for j in range(3)]} for i in range(5)]
v56.SIMPLE.write_text(json.dumps({'systems':systems}),encoding='utf-8')
run=work/'artifacts/role_direct_within_v4/nonadaptive_v56/mas5_mock'
run.mkdir(parents=True)
v56.safej(run/'public_preflight.json',{'roles':roles,'session_ids':[s['session_id'] for s in systems]})
for rd in (1,2):
    for length in (640,1536):
        rows=[]
        for s in systems:
            for i,role in enumerate(roles):
                ispositive=role in s['roles']
                # Round1 all ties => 0.5 AUC, round2 perfect => 1.0 AUC
                val=0.0 if rd==1 else (1.0 if ispositive else -1.0)
                rows.append({'session_id':s['session_id'],'target_role':role,
                    'round':rd,'logit':val,'full_tokens':100,'truncated':False})
        v56.safel(run/f'sensor_public_len{length}_round_{rd:02}.jsonl',rows)
    v561.live(run,rd)
    rpt=v56.j(run/f'live_auc_round_{rd:02}.json')
    assert rpt['within_mas_auc']==(0.5 if rd==1 else 1.0),rpt
    assert rpt['round_completed']==rd
assert len(v56.jl(run/'live_auc_history.jsonl'))==2
assert v56.j(run/'live_auc_latest.json')['round_completed']==2
assert v56.j(run/'live_auc_round_02.json')['change_auc_fused_r2_vs_r1']==0.5
v561.final(run)
final=v56.j(run/'final_MAS5_report.json')
assert final['precommitted_primary_result']['mean_within_mas_auc']==1.0
assert final['methods']['len640_r1']['mean_within_mas_auc']==0.5
assert final['methods']['len1536_r2']['mean_within_mas_auc']==1.0
assert final['methods']['len640_mean']['exact_top3_count']==5
try:
    v561.live(run,2)
except RuntimeError as e:
    assert 'already exists' in str(e)
else: raise AssertionError('Rerun should be blocked')
print('MAS5_V561_MOCK_TEST_PASSED',run)
