#!/usr/bin/env python3
"""Offline CPU-only synthetic-data tests. No MAS calls, no CUDA."""
from __future__ import annotations
import random, itertools
from pathlib import Path

from analysis.role_direct_within_v4 import v510_generalizing_sensor as mod

roles=list('abcdefghijklmnopqrst')
comps=random.Random(5102026).sample(list(itertools.combinations(roles,3)),50)
assert len(comps)>=12

def make(sid,compsel,kind='single'):
    return [dict(system_id=sid,role=r,query='What evidence should I send?',
                 answer=sid+'-'+r,kind=kind,label=int(r in compsel),source='v137_train') for r in roles]

dev=[]
for i,c in enumerate(comps[:10]):dev+=make('dev'+str(i),c)
val,test,split=mod.partition(dev,roles)
assert len(val)==5 and len(test)==5
assert {tuple(sorted(x[1])) for x in val.values()}.isdisjoint({tuple(sorted(x[1])) for x in test.values()})
assert sum(len(x[0]) for x in val.values())==100
assert sum(len(x[0]) for x in test.values())==100
assert mod.tkey(dev[0]) == mod.tkey(dict(dev[0]))
assert mod.tkey(dev[0]) != mod.tkey(dict(dev[0],answer='changed'))
# Ensure same-composition sessions are not split across test and validation.
clone=make('dev_clone',comps[0]);val2,test2,_=mod.partition(dev+clone,roles)
assert ('dev0' in val2)==('dev_clone' in val2)
assert ('dev0' in test2)==('dev_clone' in test2)
try:
    mod.partition(make('x',comps[0]),roles)
except RuntimeError:
    pass
else:
    raise AssertionError('Tiny dataset should stop before jobs')

# Source-building logic mock: substantial hist training and V56, real test excluded.
orig={key:getattr(mod.v57,key) for key in ['historical_v137','roles_and_descriptions','v56_records','locate_matched','v54_matched']}
original_source=mod.v56source
try:
    hist=[]
    for i,c in enumerate(comps[10:]):hist+=make('train'+str(i),c)
    # One replayed TRAIN composition MUST be excluded from training.
    hist+=make('duplicate_comp',comps[1])
    mod.v57.historical_v137=lambda:(hist,dev)
    mod.v57.roles_and_descriptions=lambda:(roles,{r:r for r in roles})
    vv=[];config={}
    for i,c in enumerate(comps[10:15]):
        sid='v56_'+str(i);config[sid]=set(c)
        for rnd in (1,2):
            for r in roles:vv.append(dict(system_id=sid,role=r,query='q',answer='a'+str(rnd)+r,
                         round=rnd,label=int(r in c),kind='single',source='v56'))
    # dual responses for each v56 system/role
    for sid,c in config.items():
        for r in roles:vv.append(dict(system_id=sid,role=r,query='q',answer='a1'+r,
                 query2='q2',answer2='a2'+r,round=0,label=int(r in c),kind='dual',source='v56_dual'))
    mod.v57.v56_records=lambda *args,**kwargs:(vv,config)
    mod.v57.locate_matched=lambda:None
    mod.v57.v54_matched=lambda:[]
    mod.v56source=lambda:Path('/unused')
    rr,dd,v,t,pools,stats=mod.obtain_sources()
    assert len(rr)==20 and len(v)==5 and len(t)==5
    assert 'duplicate_comp' in stats['historical_systems_excluded_matching_DEV_composition']
    assert len(pools['hist_cross_same_role'])>100
    assert len(pools['hist_within_system'])>100
    assert len(pools['mas5_within_system'])>100
    assert len(pools['matched_same_query'])>0
    assert all(a['role']==b['role'] and a['query']==b['query'] and a['system_id']!=b['system_id'] for a,b in pools['matched_same_query'])
    assert stats['v137_train_rows_accepted']>=300
finally:
    for key,obj in orig.items():setattr(mod.v57,key,obj)
    mod.v56source=original_source

print('V510_CPU_TEST_PASS; mocked compositions, no training/API calls')
