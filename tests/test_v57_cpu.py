"""Offline synthetic fixture. This DOES NOT assert actual Sensor quality."""
from __future__ import annotations
import importlib.util,json,tempfile
from pathlib import Path

p=Path(__file__).resolve().parents[1]/'analysis/role_direct_within_v4/v57_sensor_night.py'
spec=importlib.util.spec_from_file_location('v57',p)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
roles=[f'role_{i:02}' for i in range(20)]
with tempfile.TemporaryDirectory() as tmp:
    project=Path(tmp)/'mas_rl_detective'
    (project/'artifacts/role_direct_routing_pilot_v1').mkdir(parents=True)
    v56=project/'artifacts/role_direct_within_v4/nonadaptive_v56/mas5_test'
    v56.mkdir(parents=True)
    m.P=project
    m.MANIFEST=project/'artifacts/role_direct_routing_pilot_v1/manifest_privileged.json'
    m.V137=project/'artifacts/v137_presence_v1/dataset_v1'
    m.roles_and_descriptions=lambda:(roles,{r:'Synthetic capability for '+r for r in roles})
    systems=[{'split':'dev','session_id':f'system_{i}','roles':roles[i:i+3]} for i in range(5)]
    m.MANIFEST.write_text(json.dumps({'systems':systems}))
    (v56/'public_preflight.json').write_text(json.dumps({'roles':roles,'session_ids':[s['session_id'] for s in systems]}))
    for rd in (1,2):
        c=[];r=[]
        for sid in [s['session_id'] for s in systems]:
            for idx,role in enumerate(roles):
                q=f'What information would you need from me to do task {idx} in round {rd}?'
                c.append({'session_id':sid,'target_role':role,'query':q})
                r.append({'session_id':sid,'target_role':role,'round':rd,'query':q,
                           'final_output':'Synthetic answer '+str(rd),'execution_status':'ok'})
        for name,rows in [(f'candidates_dev_round_{rd:02}.jsonl',c),(f'responses_public_dev_round_{rd:02}.jsonl',r)]:
            (v56/name).write_text(''.join(json.dumps(x)+'\n' for x in rows))
    m.V137.mkdir(parents=True)
    h=[]
    for i in range(12):
        for role in roles:
            h.append({'system_id':f'h{i}','target_role':role,'query':'How can I help you with this task?',
                      'final_output':'A response for '+role,'label':int(role in roles[2:5])})
    (m.V137/'train_v137.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in h))
    (m.V137/'dev_v137.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in h[:40]))
    vr,truth=m.v56_records(v56)
    assert len(vr)==300 and len(truth)==5
    assert len([x for x in vr if x['kind']=='dual'])==100
    assert len([x for x in vr if x['label']==1])==45
    rows,history,dev,pools,stats=m.build_groups(v56)
    assert len(rows)==300 and len(pools['v56'])==765,stats
    assert len(pools['historical'])>0
    assert len(m.make_v56_eval_rows(rows))==300
    vals={sid:{r:float(r in truth[sid]) for r in roles} for sid in truth}
    auc=m.auc20(vals,truth)
    assert auc['mean_within_mas_auc']==1 and auc['exact_top3_count']==5
    print('V57_CPU_FIXTURE_PASS 5 MAS x 20 roles x 2 rounds, 765 within-MAS pairs, 100 duals, no APIs')
