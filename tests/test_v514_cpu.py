#!/usr/bin/env python3
"""API-free end-to-end prepare/report test with synthetic CF manifest and synthetic MAS outputs."""
import hashlib, json, os, sys, tempfile, types
from pathlib import Path

with tempfile.TemporaryDirectory(prefix='v514_test_') as temp:
    project=Path(temp)/'mas_rl_detective';project.mkdir()
    os.environ['V514_PROJECT_ROOT']=str(project)
    mod=types.ModuleType('adaptive_v11_active')
    io=types.ModuleType('adaptive_v11_active.io')
    io.sha256_json=lambda value:hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()
    sys.modules['adaptive_v11_active']=mod
    sys.modules['adaptive_v11_active.io']=io
    from analysis.role_direct_within_v4 import v514_matched_discrimination as v
    v.selftest()
    base=v.CF.parent;base.mkdir(parents=True)
    data=[]
    backgrounds=[('data','operations'),('writing','accessibility_review'),('qa_testing','experiment_design'),('math','fact_checking')]
    for role in v.ROLES:
        for j,(a,b) in enumerate(backgrounds):
            other='documentation' if role!='planning' else 'translation'
            for condition,r in [('TARGET_PRESENT',role),('TARGET_ABSENT',other)]:
                data.append({'session_id':f'orig_{role}_{j}_{condition}', 'target_role':role,'pair_id':f'orig_{role}_{j}',
                             'condition':condition,'roles':[a,b,r],'mode':'hierarchical','style':'default','split':'train'})
    v.CF.write_text(json.dumps({'systems':data}))
    v.MAS5.parent.mkdir(parents=True)
    v.MAS5.write_text(json.dumps({'systems':[{'split':'dev','roles':['code','writing','research']}]}))
    run=v.BASE/'pilot_fixture'
    v.prepare(run)
    assert len(v.getl(run/'candidates_train_round_01.jsonl'))==30
    candidates=v.getl(run/'candidates_train_round_01.jsonl')
    assert set(c['original_cf_pair'] for c in candidates)=={f'orig_{r}_3' for r in v.ROLES}
    responses=[]
    for c in candidates:
        ispos=c['condition']=='TARGET_PRESENT'
        # Synthetic ONLY: explicitly differentiated answers for positive/negative.
        text=(f"{c['target_role']} detailed role-specific output with tests and evidence. "*9 if ispos
              else "General advice without tools or detailed deliverables. "*3)
        responses.append({**c,'execution_status':'ok','final_output':text,'selected_roles_training_only':[],'cache_hit':False})
    v.putl(run/'responses_train_round_01.jsonl',responses)
    # Stub only the already-trained predictor interface to exercise the frozen-model report path.
    import joblib
    from analysis.role_direct_within_v4 import v514_matched_discrimination
    stub=types.ModuleType('analysis.role_direct_within_v4.v5111_rolewise_verifier')
    def fake_predict(model, rows, arm):
        return [float(len(r['answer']) if arm=='answer_only' else len(r['answer'])//100) for r in rows]
    stub.predict=fake_predict
    sys.modules[stub.__name__]=stub
    modeldir=project/'artifacts/role_direct_within_v4/sensor_v5113/rolewise_fixture'
    modeldir.mkdir(parents=True)
    for arm in ('answer_only','length_only'):
        joblib.dump({'arm':arm,'models':{role:role for role in v.ROLES}},modeldir/f'model_{arm}.joblib')
    result=v.report(run)
    assert all(x['answer_only_paired_win_rate']==1.0 for x in result['matched_role_comparison'])
    assert result['status']=='COMPLETE_DIAGNOSTIC_NOT_HELDOUT_GENERALIZATION'
    assert len(v.getj(run/'manifest_privileged.json')['systems'])==6
    assert len(v.getl(run/'responses_train_round_01.jsonl'))==30
    assert len((run/'matched_pairs.tsv').read_text().splitlines())==16
    assert len((run/'comparison.tsv').read_text().splitlines())==4
    assert (run/'report.md').is_file()
    print('V514_SYNTHETIC_E2E_OK',run)
