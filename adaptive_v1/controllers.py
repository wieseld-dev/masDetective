from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from .belief import PrefixBeliefModel
from .features import binary_entropy, total_entropy
from .outcome import EmpiricalOutcomeModel
from .schema import Action, EpisodeState

STOP = "STOP"


def _uncertainty(p: float) -> float:
    return binary_entropy(float(p))


class RandomController:
    name="random"
    def __init__(self,seed:int=0): self.rng=random.Random(seed)
    def choose(self,state:EpisodeState,available:Sequence[Action]):
        return self.rng.choice(list(available)) if available else STOP


class MaxUncertaintyController:
    name="max_uncertainty"
    def choose(self,state:EpisodeState,available:Sequence[Action]):
        if not available: return STOP
        b=state.belief or {r:.5 for r in state.roles}
        return max(available,key=lambda a:(_uncertainty(b.get(a.target,.5)),-sum(1 for x in state.history if x.action.target==a.target),a.key))


@dataclass
class VOIDecision:
    action: Action | str
    score: float
    expected_information_gain: float
    query_cost: float
    details: dict


class EmpiricalVOIController:
    """One-step Monte-Carlo value-of-information controller with explicit STOP.

    V13.5 evidence is never interpreted as a likelihood ratio. Hypothetical
    observations are re-aggregated through the learned prefix belief model.

    For efficiency the default VOI scope is the action target/competitor plus the
    currently most uncertain roles. This still values cross-role information while
    avoiding hundreds of full 20-role classifier passes per decision. Set
    `uncertain_scope=20` for full-global VOI.
    """
    name="empirical_voi"
    def __init__(
        self,
        belief_model:PrefixBeliefModel,
        outcome_model:EmpiricalOutcomeModel,
        *,query_cost:float=.04,mc_samples:int=8,min_queries:int=2,uncertain_scope:int=4,max_candidate_actions:int=6,seed:int=0,
    ):
        self.belief_model=belief_model; self.outcome_model=outcome_model; self.query_cost=float(query_cost); self.mc_samples=int(mc_samples); self.min_queries=int(min_queries); self.uncertain_scope=int(uncertain_scope); self.max_candidate_actions=int(max_candidate_actions); self.seed=int(seed)

    def _scope(self,state:EpisodeState,action:Action):
        b=state.belief or self.belief_model.predict(state.history)
        ranked=sorted(state.roles,key=lambda r:binary_entropy(float(b.get(r,.5))),reverse=True)
        scope=[]
        for r in [action.target,action.competitor,*ranked[:max(0,self.uncertain_scope)]]:
            if r in state.roles and r not in scope:scope.append(r)
        return tuple(scope)

    def value(self,state:EpisodeState,action:Action)->VOIDecision:
        b=state.belief or self.belief_model.predict(state.history);scope=self._scope(state,action)
        h0=sum(binary_entropy(float(b.get(r,.5))) for r in scope)
        samples=self.outcome_model.weighted_samples(action,b,n=self.mc_samples,seed=self.seed+state.step*1009+abs(hash(action.key))%100000)
        if not samples:return VOIDecision(action,-1e9,0.0,self.query_cost,{"samples":0,"scope":scope})
        histories=[list(state.history)+[o] for o in samples]
        b1s=self.belief_model.predict_many_subset(histories,scope)
        hs=[sum(binary_entropy(float(bb.get(r,.5))) for r in scope) for bb in b1s]
        eig=max(0.0,h0-float(np.mean(hs)));score=eig-self.query_cost
        return VOIDecision(action,score,eig,self.query_cost,{"samples":len(samples),"scope":scope,"entropy_before":h0,"entropy_after_mean":float(np.mean(hs))})

    def choose_with_details(self,state:EpisodeState,available:Sequence[Action])->VOIDecision:
        if not available:return VOIDecision(STOP,0.0,0.0,self.query_cost,{"reason":"no_actions"})
        b=state.belief or self.belief_model.predict(state.history)
        ranked=sorted(available,key=lambda a:(binary_entropy(float(b.get(a.target,.5))),-sum(1 for x in state.history if x.action.target==a.target)),reverse=True)
        shortlist=ranked[:max(1,self.max_candidate_actions)]
        vals=[self.value(state,a) for a in shortlist];best=max(vals,key=lambda x:x.score)
        if state.step>=self.min_queries and best.score<=0.0:
            return VOIDecision(STOP,0.0,0.0,self.query_cost,{"reason":"voi_below_cost","best_action":best.action.key if isinstance(best.action,Action) else best.action,"best_voi":best.expected_information_gain})
        return best

    def choose(self,state:EpisodeState,available:Sequence[Action]):return self.choose_with_details(state,available).action

