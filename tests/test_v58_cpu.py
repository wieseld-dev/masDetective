"""Run on login node: python tests/test_v58_cpu.py (no models, no API)."""
from __future__ import annotations
import importlib.util, json, math, os, tempfile
from pathlib import Path
from unittest import mock

SRC=Path(__file__).resolve().parents[1]/'analysis/role_direct_within_v4/v58_adaptive_eight.py'
spec=importlib.util.spec_from_file_location('v58',SRC)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
roles=[f'role_{i:02}' for i in range(20)]

b=m.initial_belief(roles)
assert b['round_completed']==0 and len(b['roles'])==20
assert all(abs(v['heuristic_marginal']-0.15)<1e-9 for v in b['roles'].values())

scorehist=[]
for rd in range(1,9):
    for i,role in enumerate(roles):
        scorehist.append({'round':rd,'target_role':role,'logit':(2 if i in (0,1,2) else -2)+0.01*rd})
    b=m.fuse_belief(roles,scorehist,rd)
    assert b['round_completed']==rd
    assert sum(v['heuristic_marginal'] for v in b['roles'].values())==3 or abs(sum(v['heuristic_marginal'] for v in b['roles'].values())-3)<1e-4
    assert all(v['observations']==rd for v in b['roles'].values())
    assert b['roles']['role_00']['rank']<=3
    for forbidden in ('truth_eval_only','installed_roles','selected_roles_training_only','routing_trace'):
        assert forbidden not in json.dumps(b)

# Selector changes preference by changing belief state; no truth input.
cat={'DESCRIPTIONS':{r:'database records schema indices constraints performance metrics design' for r in roles},
     'CONFUSABLES':{r:[roles[(i+1)%20]] for i,r in enumerate(roles)}}
cands=[]
for slot,q in enumerate([
  'For a database migration with slow indexed queries, which query plans should I share so you can pinpoint missing indexes?',
  'For a database migration with ambiguous schema requirements, what constraints and examples should I provide for normalization?',
  'What could I send you to help analyze schema metrics for my database migration performance problem?',
  'What test cases should I provide to verify that a database schema will support our expected workloads?',
  'I need to decide whether to denormalize several tables. What workload evidence should I provide to make that decision?',
  'What data anomalies should I collect so you can evaluate my current indexing and schema constraints?',
  'For our database performance regression, what counterexample should I share to disprove a suspected lock contention issue?',
  'What logs or sample queries would help you distinguish a database schema issue from an application concurrency bottleneck?'
]):cands.append({'candidate_index':slot,'intent':m.INTENTS[slot],'query':q})
win,scored=m.selector_heuristic(cands,roles[0],b,[],cat)
assert len(scored)==8 and sum(c['quality_valid'] for c in scored)>=6
assert win['candidate_index'] in range(8)
assert all('truth_eval_only' not in q for q in scored)
assert m.validate_query('You mentioned query latency. What logs should I send you?')[1]=='stateless_not_self_contained'
assert m.validate_query(cands[0]['query'])[0]

messages=m.make_messages(roles[0],'VERIFY',b,[{'round':1,'query':cands[0]['query'],'response':'Index scans were slow. Gather query plans first.'}],cat,1)
assert 'CURRENT PUBLIC BELIEF OVER ALL 20 ROLES' in messages[1]['content']
assert 'Index scans were slow' in messages[1]['content']
assert 'NO MEMORY' in messages[0]['content']
assert 'truth_eval_only' not in messages[1]['content']

# Eight distinct per-round checkpoint files; evaluate does not leak truth into public belief.
with tempfile.TemporaryDirectory() as td:
    base=Path(td);m.AREA=base/'artifacts'/'role_direct_within_v4'/'adaptive_v58';m.AREA.mkdir(parents=True)
    root=m.AREA/'mas5_unit';root.mkdir()
    mf=base/'manifest.json';mf.write_text(json.dumps({'systems':[{'split':'dev','session_id':'sys1','roles':roles[:3]}]}))
    m.MANIFEST=mf
    m.atomic_json(root/'public_preflight.json',{'roles':roles,'session_id':'sys1'})
    m.atomic_json(root/'public_belief_round_01.json',b)
    m.evaluate(root,1)
    output=m.read_json(root/'live_auc_latest.json')
    assert output['top3_correct']==3 and output['exact_top3'] is True and output['within_mas_auc']==1.0
    assert m.read_json(root/'public_belief_round_01.json').get('truth_eval_only') is None
    assert m.read_json(root/'live_auc_round_01.json')['total_executed_mas_queries']==20
    try:m.once_json(root/'live_auc_round_01.json',{})
    except FileExistsError:pass
    else:raise AssertionError('No-overwrite protection broken')
print('V58_CPU_TESTS_PASS belief x8, selector, message condition, no overwrite, live Within-MAS AUC, no leakage')
