from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Mapping, Sequence

from .reward import normalized_word_jaccard, surface_valid

EPS = 1e-5


def _clip01(p: float) -> float:
    return min(1.0 - EPS, max(EPS, float(p)))


def role_log_score(role_probs: Mapping[str, float], true_roles: Sequence[str]) -> float:
    truth = set(map(str, true_roles))
    roles = tuple(sorted(role_probs))
    if not roles:
        return 0.0
    total = 0.0
    for r in roles:
        p = _clip01(role_probs[r])
        total += math.log(p) if r in truth else math.log1p(-p)
    return total / len(roles)


@dataclass(frozen=True)
class RoleRewardBreakdown:
    total: float
    role_log_score_before: float
    role_log_score_after: float
    role_log_score_improvement: float
    repetition_penalty: float
    length_penalty: float
    invalid_penalty: float
    valid: bool
    invalid_reason: str = ""

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def adaptive_role_reward(
    *,
    old_role_probs: Mapping[str, float],
    new_role_probs: Mapping[str, float],
    true_roles: Sequence[str],
    query: str,
    previous_queries: Sequence[str] = (),
    generator_valid: bool = True,
    role_scale: float = 20.0,
    reward_clip: float = 3.0,
) -> RoleRewardBreakdown:
    ok, reason = surface_valid(query)
    if not generator_valid:
        ok = False
        reason = reason or "generator_invalid"
    if not ok:
        return RoleRewardBreakdown(
            total=-1.5,
            role_log_score_before=role_log_score(old_role_probs, true_roles),
            role_log_score_after=role_log_score(old_role_probs, true_roles),
            role_log_score_improvement=0.0,
            repetition_penalty=0.0,
            length_penalty=0.0,
            invalid_penalty=1.5,
            valid=False,
            invalid_reason=reason,
        )

    before = role_log_score(old_role_probs, true_roles)
    after = role_log_score(new_role_probs, true_roles)
    improvement = after - before

    sim = max((normalized_word_jaccard(query, q) for q in previous_queries), default=0.0)
    repetition_penalty = 0.0 if sim <= 0.70 else min(0.12, (sim - 0.70) * 0.40)
    words = len(re.findall(r"\b\w+\b", query))
    length_penalty = max(0, words - 80) * 0.001

    total = float(role_scale) * improvement - repetition_penalty - length_penalty
    total = max(-reward_clip, min(reward_clip, total))
    return RoleRewardBreakdown(
        total=float(total),
        role_log_score_before=float(before),
        role_log_score_after=float(after),
        role_log_score_improvement=float(improvement),
        repetition_penalty=float(repetition_penalty),
        length_penalty=float(length_penalty),
        invalid_penalty=0.0,
        valid=True,
        invalid_reason="",
    )
