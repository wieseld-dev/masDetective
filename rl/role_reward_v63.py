from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Set


@dataclass(frozen=True)
class RoleRewardConfigV63:
    # Dense action-specific signal + global reconstruction signal.
    focus_brier_weight: float = 1.00
    global_brier_weight: float = 1.50
    k_quality_weight: float = 0.15
    query_cost: float = 0.01
    focus_delta_clip: float = 0.50
    global_delta_clip: float = 0.25
    k_delta_clip: float = 0.25


class RoleWhiteBoxRewardV63:
    """Training-only reward. White-box truth is converted to a scalar and never enters policy state."""

    def __init__(self, candidate_roles, config: RoleRewardConfigV63 | None = None):
        self.candidate_roles = tuple(candidate_roles)
        self.config = config or RoleRewardConfigV63()

    def reset(self) -> None:
        pass

    def prime(self, trace) -> None:
        pass

    def _brier(self, belief: Mapping[str, float], true_roles: Set[str]) -> float:
        return sum(
            (float(belief.get(role, 0.5)) - (1.0 if role in true_roles else 0.0)) ** 2
            for role in self.candidate_roles
        ) / max(len(self.candidate_roles), 1)

    def step(
        self,
        trace,
        *,
        focus_role: str,
        prev_role_belief: Mapping[str, float],
        new_role_belief: Mapping[str, float],
        query: str,
        output: str,
    ) -> tuple[float, Dict[str, float]]:
        c = self.config
        true_roles = set(trace.true_roles)
        k = len(true_roles)

        b0 = self._brier(prev_role_belief, true_roles)
        b1 = self._brier(new_role_belief, true_roles)
        global_raw = b0 - b1
        global_delta = max(-c.global_delta_clip, min(c.global_delta_clip, global_raw))

        y = 1.0 if focus_role in true_roles else 0.0
        p0 = float(prev_role_belief.get(focus_role, 0.5))
        p1 = float(new_role_belief.get(focus_role, 0.5))
        focus_raw = (p0 - y) ** 2 - (p1 - y) ** 2
        focus_delta = max(-c.focus_delta_clip, min(c.focus_delta_clip, focus_raw))

        ek0 = sum(float(prev_role_belief.get(r, 0.0)) for r in self.candidate_roles)
        ek1 = sum(float(new_role_belief.get(r, 0.0)) for r in self.candidate_roles)
        denom = max(len(self.candidate_roles), 1)
        k_raw = (abs(ek0 - k) - abs(ek1 - k)) / denom
        k_delta = max(-c.k_delta_clip, min(c.k_delta_clip, k_raw))

        reward = (
            c.focus_brier_weight * focus_delta
            + c.global_brier_weight * global_delta
            + c.k_quality_weight * k_delta
            - c.query_cost
        )

        selected_roles = tuple(getattr(trace, "selected_roles", ()))
        return float(reward), {
            "r_focus_raw": float(focus_raw),
            "r_focus": float(focus_delta),
            "r_global_raw": float(global_raw),
            "r_global": float(global_delta),
            "r_k_raw": float(k_raw),
            "r_k": float(k_delta),
            "brier_before": float(b0),
            "brier_after": float(b1),
            "focus_p_before": float(p0),
            "focus_p_after": float(p1),
            "focus_true": float(y),
            "expected_k_before": float(ek0),
            "expected_k_after": float(ek1),
            "activated_role_count": float(len(selected_roles)),
            "focus_activated": float(focus_role in selected_roles),
        }
