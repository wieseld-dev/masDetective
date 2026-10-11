from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

from adaptive_v11_active.reward import normalized_word_jaccard, surface_valid

EPS = 1e-6


def _clip(p: float) -> float:
    return min(1.0-EPS, max(EPS, float(p)))


def role_log_score(probs: Mapping[str,float], true_roles: Sequence[str], roles: Sequence[str]) -> float:
    truth = set(map(str, true_roles))
    vals = []
    for r in roles:
        p = _clip(probs[r])
        y = 1.0 if r in truth else 0.0
        vals.append(y*math.log(p) + (1.0-y)*math.log(1.0-p))
    return sum(vals)/max(1,len(vals))


def role_brier(probs: Mapping[str,float], true_roles: Sequence[str], roles: Sequence[str]) -> float:
    truth=set(map(str,true_roles))
    return sum((_clip(probs[r])-(1.0 if r in truth else 0.0))**2 for r in roles)/max(1,len(roles))


def margin(probs: Mapping[str,float], true_roles: Sequence[str], roles: Sequence[str]) -> float:
    truth=set(map(str,true_roles))
    pos=[_clip(probs[r]) for r in roles if r in truth]
    neg=[_clip(probs[r]) for r in roles if r not in truth]
    return (sum(pos)/max(1,len(pos))) - (sum(neg)/max(1,len(neg)))


def soft_f1(probs: Mapping[str,float], true_roles: Sequence[str], roles: Sequence[str]) -> float:
    truth=set(map(str,true_roles))
    tp=sum(_clip(probs[r]) for r in roles if r in truth)
    ps=sum(_clip(probs[r]) for r in roles)
    ys=float(len(truth))
    return 2.0*tp/max(EPS,ps+ys)


@dataclass
class RewardBreakdown:
    total: float
    delta_log_score: float
    delta_brier: float
    delta_margin: float
    delta_soft_f1: float
    repetition_penalty: float
    length_penalty: float
    invalid_penalty: float
    valid: bool
    invalid_reason: str = ""

    def to_dict(self):
        return asdict(self)


def curriculum_reward(*, old_probs: Mapping[str,float], new_probs: Mapping[str,float],
                      true_roles: Sequence[str], roles: Sequence[str], query: str,
                      previous_queries: Sequence[str], generator_valid: bool) -> RewardBreakdown:
    ok, reason = surface_valid(query)
    if not generator_valid or not ok:
        return RewardBreakdown(
            total=-2.5,
            delta_log_score=0.0, delta_brier=0.0, delta_margin=0.0, delta_soft_f1=0.0,
            repetition_penalty=0.0, length_penalty=0.0, invalid_penalty=2.5,
            valid=False, invalid_reason=reason or "generator_invalid",
        )

    before_ls=role_log_score(old_probs,true_roles,roles)
    after_ls=role_log_score(new_probs,true_roles,roles)
    before_br=role_brier(old_probs,true_roles,roles)
    after_br=role_brier(new_probs,true_roles,roles)
    before_m=margin(old_probs,true_roles,roles)
    after_m=margin(new_probs,true_roles,roles)
    before_f=soft_f1(old_probs,true_roles,roles)
    after_f=soft_f1(new_probs,true_roles,roles)

    dls=after_ls-before_ls
    dbr=before_br-after_br
    dm=after_m-before_m
    df=after_f-before_f

    sim=max((normalized_word_jaccard(query,q) for q in previous_queries), default=0.0)
    rep=max(0.0, sim-0.55)*0.8
    words=len(str(query).split())
    length=max(0, words-80)*0.004

    # Optimize ONLY proper scoring rules for inference quality. Margin/soft-F1 are logged
    # diagnostics, not optimization targets. With gamma=1, these delta rewards telescope
    # across a trajectory to final proper-score improvement.
    total=50.0*dls + 15.0*dbr - rep - length
    total=max(-3.0,min(3.0,total))
    return RewardBreakdown(
        total=float(total), delta_log_score=float(dls), delta_brier=float(dbr),
        delta_margin=float(dm), delta_soft_f1=float(df), repetition_penalty=float(rep),
        length_penalty=float(length), invalid_penalty=0.0, valid=True,
    )


def group_relative_advantages(rewards: Sequence[float]) -> list[float]:
    xs=[float(x) for x in rewards]
    if not xs:
        return []
    m=sum(xs)/len(xs)
    var=sum((x-m)**2 for x in xs)/len(xs)
    sd=math.sqrt(var+1e-8)
    if sd < 1e-4:
        return [0.0 for _ in xs]
    return [(x-m)/sd for x in xs]
