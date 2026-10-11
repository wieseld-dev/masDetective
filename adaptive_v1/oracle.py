from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .features import total_entropy
from .metrics import role_metrics
from .replay import ReplayEnv
from .schema import Action


@dataclass
class OracleDecision:
    action: Action | str
    utility: float
    brier_gain: float
    query_cost: float


def one_step_oracle(env:ReplayEnv,available:Sequence[Action],*,query_cost:float=.01)->OracleDecision:
    """Privileged one-step upper bound used only for labels/evaluation.

    It peeks at cached outcomes and true roles. Never deploy this policy.
    """
    state=env.state(); m0=role_metrics(state.belief,env.system.truth_roles,state.roles); b0=float(m0["brier"])
    best=OracleDecision("STOP",0.0,0.0,float(query_cost))
    for a in available:
        o=env.peek(a)
        if o is None: continue
        b1=env.belief_model.predict(list(state.history)+[o]); m1=role_metrics(b1,env.system.truth_roles,state.roles); gain=b0-float(m1["brier"]); u=gain-float(query_cost)
        if u>best.utility: best=OracleDecision(a,u,gain,float(query_cost))
    return best
