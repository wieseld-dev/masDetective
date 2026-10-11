from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import BASE_FEATURE_NAMES, role_history_features
from .schema import EvidenceObservation


@dataclass
class BankSystem:
    system_id: str
    truth_roles: tuple[str, ...]
    observations: list[EvidenceObservation]
    split: str = "train"
    metadata: dict | None = None


class PrefixBeliefModel:
    """Learned multi-observation aggregator.

    The Judge's raw evidence scores are deliberately *not* accumulated as logits.
    Instead, a grouped supervised prefix model learns how histories of correlated
    all-role evidence map to role-presence probabilities. Truth is used only while
    fitting this offline model; inference receives evidence/history only.
    """

    VERSION = "adaptive_prefix_belief_v1"

    def __init__(self, roles: Sequence[str], *, C: float = 0.30):
        self.roles = tuple(map(str, roles))
        self.C = float(C)
        self.model = Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(C=self.C, class_weight="balanced", max_iter=3000, solver="lbfgs")),
        ])
        self._fitted = False

    def _row(self, history: Sequence[EvidenceObservation], role: str) -> np.ndarray:
        base = role_history_features(history, role, self.roles)
        onehot = np.zeros(len(self.roles), dtype=np.float32)
        onehot[self.roles.index(role)] = 1.0
        return np.concatenate([base, onehot], axis=0)

    @property
    def feature_names(self) -> tuple[str, ...]:
        return tuple(BASE_FEATURE_NAMES) + tuple(f"role={r}" for r in self.roles)

    def predict_subset(self, history: Sequence[EvidenceObservation], subset: Sequence[str]) -> dict[str, float]:
        subset=tuple(subset)
        if not self._fitted:
            return {r:0.5 for r in subset}
        X=np.vstack([self._row(history,r) for r in subset])
        p=self.model.predict_proba(X)[:,1]
        return {r:float(np.clip(v,1e-4,1-1e-4)) for r,v in zip(subset,p)}

    def predict_many_subset(self, histories: Sequence[Sequence[EvidenceObservation]], subset: Sequence[str]) -> list[dict[str,float]]:
        subset=tuple(subset)
        if not histories:return []
        if not self._fitted:return [{r:.5 for r in subset} for _ in histories]
        X=np.vstack([self._row(h,r) for h in histories for r in subset])
        p=self.model.predict_proba(X)[:,1]
        out=[];k=len(subset)
        for i in range(len(histories)):
            vals=p[i*k:(i+1)*k];out.append({r:float(np.clip(v,1e-4,1-1e-4)) for r,v in zip(subset,vals)})
        return out

    def predict(self, history: Sequence[EvidenceObservation]) -> dict[str, float]:
        return self.predict_subset(history,self.roles)

    def fit(
        self,
        systems: Sequence[BankSystem],
        *,
        permutations_per_system: int = 4,
        max_prefix: int = 20,
        seed: int = 1414,
    ) -> dict:
        X=[]; y=[]; groups=[]; prefix_sizes=[]
        rng = random.Random(seed)
        for sys in systems:
            obs = list(sys.observations)
            if not obs:
                continue
            truth = set(sys.truth_roles)
            perms=[]
            perms.append(list(range(len(obs))))
            for _ in range(max(0, int(permutations_per_system)-1)):
                idx=list(range(len(obs))); rng.shuffle(idx); perms.append(idx)
            seen=set()
            for pi, idxs in enumerate(perms):
                cap=min(len(idxs), int(max_prefix))
                # dense early prefixes + log-ish later coverage
                sizes=sorted(set([1,2,3,4,5,6,8,10,12,16,20,cap]))
                sizes=[s for s in sizes if 1 <= s <= cap]
                for k in sizes:
                    key=(pi,tuple(idxs[:k]))
                    if key in seen: continue
                    seen.add(key)
                    hist=[obs[i] for i in idxs[:k]]
                    for role in self.roles:
                        X.append(self._row(hist, role)); y.append(1 if role in truth else 0); groups.append(sys.system_id); prefix_sizes.append(k)
        if not X:
            raise RuntimeError("no prefix training rows")
        X=np.vstack(X).astype(np.float32); y=np.asarray(y,dtype=np.int64)
        if len(set(y.tolist())) < 2:
            raise RuntimeError("belief fit needs both present and absent labels")
        self.model.fit(X,y); self._fitted=True
        pred=self.model.predict_proba(X)[:,1]
        eps=1e-8
        brier=float(np.mean((pred-y)**2))
        return {
            "version":self.VERSION,"n_rows":int(len(y)),"n_systems":int(len(set(groups))),"positive_rate":float(np.mean(y)),
            "train_brier":brier,"prefix_min":int(min(prefix_sizes)),"prefix_max":int(max(prefix_sizes)),
            "feature_dim":int(X.shape[1]),"permutations_per_system":int(permutations_per_system),
            "warning":"training metrics are diagnostic only; evaluate on group-held-out systems",
        }

    def save(self, path: str | Path) -> None:
        p=Path(path); p.parent.mkdir(parents=True,exist_ok=True)
        joblib.dump({"version":self.VERSION,"roles":self.roles,"C":self.C,"model":self.model,"fitted":self._fitted},p)

    @classmethod
    def load(cls, path: str | Path) -> "PrefixBeliefModel":
        d=joblib.load(path); obj=cls(d["roles"],C=float(d.get("C",.3))); obj.model=d["model"]; obj._fitted=bool(d.get("fitted",True)); return obj
