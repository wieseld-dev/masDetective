import numpy as np
from v9.reward import energy_separation, RobustCounterfactualGraderV9

def v(x,y):
    a=np.array([x,y],dtype=np.float32); return a/np.linalg.norm(a)

def test_energy_separation_prefers_distinct_groups():
    same=energy_separation([v(1,0),v(.99,.01)],[v(.98,.02),v(1,0)])
    apart=energy_separation([v(1,0),v(.99,.01)],[v(0,1),v(.01,.99)])
    assert apart > same

def test_no_positive_reward_without_activation():
    g=RobustCounterfactualGraderV9(min_activation_rate=.75)
    r=g.score(
        target_selected_flags=[False,False],
        plus_query_embs=[v(1,0),v(1,0)], minus_query_embs=[v(0,1),v(0,1)],
        plus_anchor_embs=[v(1,0),v(1,0)], minus_anchor_embs=[v(.8,.2),v(.8,.2)],
    )
    assert r.total < 0
    assert not r.eligible

def test_no_positive_reward_without_specific_separation():
    g=RobustCounterfactualGraderV9()
    r=g.score(
        target_selected_flags=[True,True],
        plus_query_embs=[v(1,0),v(1,0)], minus_query_embs=[v(1,0),v(1,0)],
        plus_anchor_embs=[v(1,0),v(1,0)], minus_anchor_embs=[v(1,0),v(1,0)],
    )
    assert r.specific_separation <= 1e-6
    assert r.total <= 0
