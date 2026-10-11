#!/usr/bin/env python3
"""No API. Synthetic integration: select different backgrounds, prepare, fake responses, report."""
import csv, hashlib, json, os, sys, tempfile, types
from pathlib import Path
with tempfile.TemporaryDirectory(prefix='v515_test_') as temp:
    root=Path(temp)/'mas_rl_detective';root.mkdir()
    os.environ['V514_PROJECT_ROOT']=str(root)
    os.environ['V515_PROJECT_ROOT']=str(root)
    mod=types.ModuleType('adaptive_v11_active')
    io=types.ModuleType('adaptive_v11_active.io')
    io.sha256_json=lambda x:hashlib.sha256(json.dumps(x,sort_keys=True).encode()).hexdigest()
    sys.modules['adaptive_v11_active']=mod
    sys.modules['adaptive_v11_active.io']=io
    from analysis.role_direct_within_v4 import v514_matched_discrimination as old
    from analysis.role_direct_within_v4 import v515_multi_background as v
    v.selftest()
    old.CF.parent.mkdir(parents=True)
    backgrounds=[('data','operations'),('writing','design'),('math','fact_checking'),('qa_testing','documentation')]
    systems=[]
    for role in v.ROLES:
        for i,(a,b) in enumerate(backgrounds):
            for cond,worker in [('TARGET_PRESENT',role),('TARGET_ABSENT','translation')]:
                systems.append({'session_id':f'orig_{role}_{i}_{cond}',
                   'target_role':role,'pair_id':f'orig_{role}_{i}',
                   'condition':cond,'roles':[a,b,worker],
                   'mode':'hierarchical','style':'standard','split':'train'})
    old.CF.write_text(json.dumps({'systems':systems}))
    old.MAS5.parent.mkdir(parents=True)
    old.MAS5.write_text(json.dumps({'systems':[{'split':'dev','roles':['code','planning','research']}]}))
    previous=v.V514_BASE/'pilot_earlier'
    previous.mkdir(parents=True)
    (previous/'preflight.json').write_text(json.dumps({'pairs':{r:{'original_cf_pair':f'orig_{r}_3'} for r in v.ROLES}}))
    v.V514_BASE.joinpath('latest_run_path.txt').write_text(str(previous)+'\n')
    run=v.BASE/'pilot_fixture'
    v.prepare(run)
    assert len(v.l(run/'candidates_train_round_01.jsonl'))==60
    candidates=v.l(run/'candidates_train_round_01.jsonl')
    assert not any(c['original_cf_pair'].endswith('_3') for c in candidates)
    assert len(set(c['session_id'] for c in candidates))==12
    assert len(set((c['target_role'],c['background'],c['candidate_index'],c['query']) for c in candidates))==30
    pairs={}
    for c in candidates:
        k=(c['target_role'],c['background'],c['candidate_index'])
        pairs.setdefault(k,set()).add(c['query'])
    assert all(len(q)==1 for q in pairs.values())
    responses=[]
    for c in candidates:
        present=c['condition']=='TARGET_PRESENT'
        answer=('Verified structured evidence and reproducible tests, '*15 if present
                else 'General response without role fingerprint, '*4)
        responses.append({**c,'execution_status':'ok','final_output':answer,
          'selected_roles_training_only':[c['target_role']] if present else []})
    v.wl(run/'responses_train_round_01.jsonl',responses)
    # Mimic frozen model scores in test ONLY; no trained model needed for integrity check.
    v.frozen_scores=lambda bykey: ({(arm,role,bg,slot,cond):(0.9 if cond=='TARGET_PRESENT' else 0.1)
      for arm in ('answer_only','length_only') for role in v.ROLES for bg in (1,2) for slot in range(5)
      for cond in ('TARGET_PRESENT','TARGET_ABSENT')},'synthetic_stub')
    result=v.report(run)
    assert len(result['by_background'])==6 and len(result['by_role'])==3
    assert all(x['answer_only_paired_win_rate']==1 for x in result['by_background'])
    assert all(x['answer_only_pooled_auc']==1 for x in result['by_role'])
    assert len((run/'matched_pairs.tsv').read_text().splitlines())==31
    assert len((run/'probe_consistency.tsv').read_text().splitlines())==16
    assert len((run/'by_background.tsv').read_text().splitlines())==7
    assert len((run/'comparison.tsv').read_text().splitlines())==4
    assert result['n_mas_executions']==60
    print('V515_SYNTHETIC_E2E_OK (NOT real MAS accuracy)')
