from __future__ import annotations

import random
from collections import defaultdict
from typing import Mapping, Sequence

from .belief import BankSystem
from .schema import Action, EvidenceObservation


class EmpiricalOutcomeModel:
    """Empirical observation model used only for *action valuation*.

    Pools are conditioned on action target/competitor and the hidden present/absent
    states in the training bank. At deployment the hidden state is marginalized
    under the current learned belief; truth is never read by the controller.
    """

    def __init__(self, roles: Sequence[str]):
        self.roles=tuple(roles)
        self.exact=defaultdict(list)
        self.target_only=defaultdict(list)
        self.global_by_y=defaultdict(list)

    def fit(self, systems: Sequence[BankSystem]) -> dict:
        n=0
        for s in systems:
            truth=set(s.truth_roles)
            for o in s.observations:
                a=o.action; yt=int(a.target in truth); yc=None if a.competitor is None else int(a.competitor in truth)
                self.exact[(a.target,a.competitor,yt,yc)].append(o)
                self.target_only[(a.target,yt)].append(o)
                self.global_by_y[yt].append(o); n+=1
        return {"n_observations":n,"exact_pools":len(self.exact),"target_pools":len(self.target_only)}

    def _pool(self, action: Action, yt: int, yc: int | None):
        p=self.exact.get((action.target,action.competitor,int(yt),yc),())
        if p: return p
        p=self.target_only.get((action.target,int(yt)),())
        if p: return p
        return self.global_by_y.get(int(yt),())

    def weighted_samples(self, action: Action, belief: Mapping[str,float], *, n: int=16, seed: int=0):
        rng=random.Random(seed)
        pt=float(belief.get(action.target,.5))
        if action.competitor is None:
            states=[(1,None,pt),(0,None,1-pt)]
        else:
            pc=float(belief.get(action.competitor,.5))
            # Independence here is only a proposal distribution for Monte Carlo VOI;
            # the learned history aggregator handles correlated observed evidence.
            states=[(1,1,pt*pc),(1,0,pt*(1-pc)),(0,1,(1-pt)*pc),(0,0,(1-pt)*(1-pc))]
        avail=[]
        for yt,yc,w in states:
            pool=list(self._pool(action,yt,yc))
            if pool and w>0: avail.append((pool,float(w)))
        z=sum(w for _,w in avail)
        if z<=0: return []
        avail=[(p,w/z) for p,w in avail]
        out=[]
        for i in range(max(1,int(n))):
            u=rng.random(); acc=0.0; pool=avail[-1][0]
            for pp,w in avail:
                acc+=w
                if u<=acc: pool=pp; break
            out.append(pool[rng.randrange(len(pool))])
        return out
