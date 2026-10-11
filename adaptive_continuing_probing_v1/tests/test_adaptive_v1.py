from __future__ import annotations
import random
from adaptive_v1.schema import Action,EvidenceObservation
from adaptive_v1.belief import BankSystem,PrefixBeliefModel
from adaptive_v1.outcome import EmpiricalOutcomeModel
from adaptive_v1.controllers import MaxUncertaintyController,EmpiricalVOIController
from adaptive_v1.replay import ReplayEnv

ROLES=('a','b','c','d')

def make_system(i,truth):
    obs=[]
    for j,target in enumerate(ROLES):
        e={r:(2.5 if r in truth else -2.5) for r in ROLES}
        # targeted role is especially informative but all-role vector still carries cross evidence
        e[target]+=(1.5 if target in truth else -1.5)
        obs.append(EvidenceObservation(Action(target),f'q-{target}',e,system_id=f's{i}',response_id=f'{i}-{j}'))
    return BankSystem(f's{i}',tuple(truth),obs,'train' if i<8 else 'dev')

def test_prefix_belief_and_voi():
    systems=[]
    combos=[('a','b'),('a','c'),('a','d'),('b','c'),('b','d'),('c','d'),('a','b'),('c','d'),('a','c'),('b','d')]
    for i,t in enumerate(combos):systems.append(make_system(i,t))
    model=PrefixBeliefModel(ROLES,C=.5);model.fit(systems[:8],permutations_per_system=2,max_prefix=4,seed=1)
    out=EmpiricalOutcomeModel(ROLES);out.fit(systems[:8])
    env=ReplayEnv(systems[8],model,budget=4,seed=1);st=env.state();assert set(st.belief)==set(ROLES);assert all(0.0 < st.belief[r] < 1.0 for r in ROLES)
    ctrl=EmpiricalVOIController(model,out,query_cost=0.0,mc_samples=8,min_queries=0,seed=1);a=ctrl.choose(st,env.available_actions());assert a!='STOP'
    env.step(a);b=env.state().belief;assert set(b)==set(ROLES)
