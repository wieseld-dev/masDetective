from __future__ import annotations

from dataclasses import dataclass
from typing import Callable,Mapping,Sequence

from .schema import Action,EvidenceObservation,EpisodeState


@dataclass
class LiveStep:
    step:int; action:Action; query:str; answer:str; evidence:dict[str,float]; belief_after:dict[str,float]


class AdaptiveProbeSession:
    """Dependency-injected deployment loop.

    No privileged truth is accepted by this class. Final wiring after V11.2/V13.5:
      probe_fn(Action, history) -> benign query
      mas_query_fn(query) -> visible text response
      judge_fn(query, response) -> all-role V13.5 evidence vector
    """
    def __init__(self,roles,belief_model,controller,probe_fn:Callable,mas_query_fn:Callable,judge_fn:Callable,*,budget:int=20):
        self.roles=tuple(roles); self.belief_model=belief_model; self.controller=controller; self.probe_fn=probe_fn; self.mas_query_fn=mas_query_fn; self.judge_fn=judge_fn; self.budget=int(budget); self.history=[]; self.steps=[]
    def state(self):
        return EpisodeState(self.roles,list(self.history),self.budget-len(self.history),self.belief_model.predict(self.history))
    def run(self,action_provider:Callable[[EpisodeState],Sequence[Action]]):
        while len(self.history)<self.budget:
            state=self.state(); available=list(action_provider(state)); action=self.controller.choose(state,available)
            if action=="STOP":break
            query=str(self.probe_fn(action,list(self.history))); answer=str(self.mas_query_fn(query)); evidence=dict(self.judge_fn(query,answer))
            if set(evidence)!=set(self.roles): raise RuntimeError("judge_fn must return exactly one evidence score per role")
            obs=EvidenceObservation(action,query,evidence,response_id=f"live-{len(self.history)+1}")
            self.history.append(obs); b=self.belief_model.predict(self.history); self.steps.append(LiveStep(len(self.history),action,query,answer,evidence,b))
        return self.state()
