from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Mapping, Sequence

from .belief import BankSystem, PrefixBeliefModel
from .schema import Action, EpisodeState, EvidenceObservation


class ReplayEnv:
    """Offline black-box episode over a cached system.

    Truth is retained internally for evaluation/oracle labels only. `state()` never
    exposes it. An action consumes one cached observation for that action key.
    """
    def __init__(self,system:BankSystem,belief_model:PrefixBeliefModel,*,budget:int=20,seed:int=0):
        self.system=system; self.belief_model=belief_model; self.budget=int(budget); self.rng=random.Random(seed)
        self.by_action=defaultdict(list)
        for o in system.observations:self.by_action[o.action.key].append(o)
        for xs in self.by_action.values(): self.rng.shuffle(xs)
        self.history=[]; self.used=defaultdict(int)

    @property
    def roles(self): return self.belief_model.roles

    def state(self)->EpisodeState:
        b=self.belief_model.predict(self.history)
        return EpisodeState(tuple(self.roles),list(self.history),max(self.budget-len(self.history),0),b)

    def available_actions(self)->list[Action]:
        if len(self.history)>=self.budget:return []
        out=[]
        for key,xs in self.by_action.items():
            i=self.used[key]
            if i<len(xs): out.append(xs[i].action)
        return sorted(out,key=lambda a:a.key)

    def peek(self,action:Action)->EvidenceObservation|None:
        xs=self.by_action.get(action.key,()); i=self.used[action.key]
        return xs[i] if i<len(xs) else None

    def step(self,action:Action)->EpisodeState:
        o=self.peek(action)
        if o is None: raise KeyError(f"no replay observation for {action.key}")
        self.history.append(o); self.used[action.key]+=1
        return self.state()

    @property
    def truth_roles_eval_only(self)->set[str]: return set(self.system.truth_roles)
