from __future__ import annotations

import random
from typing import Sequence

from .controllers import RandomController,MaxUncertaintyController,EmpiricalVOIController,STOP
from .metrics import role_metrics
from .oracle import one_step_oracle
from .replay import ReplayEnv


def run_controller(system,belief_model,controller,*,budget,seed=0,oracle=False,oracle_cost=.01):
    env=ReplayEnv(system,belief_model,budget=budget,seed=seed); stopped=False
    while len(env.history)<budget:
        state=env.state(); avail=env.available_actions()
        if not avail:break
        action=one_step_oracle(env,avail,query_cost=oracle_cost).action if oracle else controller.choose(state,avail)
        if action==STOP:stopped=True;break
        env.step(action)
    st=env.state();m=role_metrics(st.belief,system.truth_roles,st.roles);m.update({"queries":len(env.history),"stopped_early":bool(stopped and len(env.history)<budget),"system_id":system.system_id})
    return m
