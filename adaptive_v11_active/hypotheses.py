from __future__ import annotations

import heapq
import itertools
import math
from dataclasses import dataclass
from typing import Mapping, Sequence


EPS = 1e-6


@dataclass(frozen=True)
class RoleSetHypothesis:
    k: int
    roles: tuple[str, ...]
    log_score: float
    relative_prob: float = 0.0


def _clip01(p: float) -> float:
    return min(1.0 - EPS, max(EPS, float(p)))


def top_role_set_hypotheses(
    role_probs: Mapping[str, float],
    k_probs: Mapping[int, float],
    *,
    top_n: int = 8,
    allowed_ks: Sequence[int] | None = None,
) -> list[RoleSetHypothesis]:
    """Exact top-N independent-marginal role-set hypotheses for small role universes.

    With 20 roles and K=3..6 this enumerates 60,249 subsets, which is cheap and avoids
    heuristic hypothesis construction. The weights are *relative within the returned
    top-N*, not the posterior mass captured by those hypotheses.
    """
    roles = tuple(sorted(map(str, role_probs)))
    ks = tuple(sorted(map(int, allowed_ks if allowed_ks is not None else k_probs.keys())))
    p = {r: _clip01(role_probs[r]) for r in roles}
    base = sum(math.log1p(-p[r]) for r in roles)
    logits = {r: math.log(p[r]) - math.log1p(-p[r]) for r in roles}

    heap: list[tuple[float, int, tuple[str, ...]]] = []
    for k in ks:
        pk = _clip01(k_probs.get(k, EPS))
        if k < 0 or k > len(roles):
            continue
        k_base = base + math.log(pk)
        for combo in itertools.combinations(roles, k):
            score = k_base + sum(logits[r] for r in combo)
            item = (score, k, combo)
            if len(heap) < top_n:
                heapq.heappush(heap, item)
            elif score > heap[0][0]:
                heapq.heapreplace(heap, item)

    raw = sorted(heap, reverse=True)
    if not raw:
        return []
    mx = raw[0][0]
    ws = [math.exp(max(-80.0, s - mx)) for s, _, _ in raw]
    z = sum(ws) or 1.0
    return [
        RoleSetHypothesis(k=k, roles=tuple(combo), log_score=float(score), relative_prob=float(w / z))
        for (score, k, combo), w in zip(raw, ws)
    ]
