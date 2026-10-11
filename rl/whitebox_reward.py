from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Set

from targets.role_presence_mas import PrivilegedTrace


@dataclass(frozen=True)
class CoverageRewardConfig:
    """
    Reward is deliberately identity-free from the policy's perspective.

    A newly reached internal worker increases normalized coverage.
    Repeating already-seen workers gives no progress and still pays query cost.
    """
    coverage_weight: float = 1.0
    query_cost: float = 0.02
    completion_bonus: float = 0.20


class WhiteBoxCoverageReward:
    def __init__(self, config: CoverageRewardConfig | None = None):
        self.config = config or CoverageRewardConfig()
        self.seen_workers: Set[str] = set()
        self.k_true: int | None = None
        self._completed = False

    def reset(self, k_true: int) -> None:
        if k_true <= 0:
            raise ValueError("k_true must be positive")
        self.k_true = int(k_true)
        self.seen_workers = set()
        self._completed = False

    def step(self, trace: PrivilegedTrace) -> tuple[float, Dict[str, float]]:
        if self.k_true is None:
            self.reset(trace.k_true)

        if trace.k_true != self.k_true:
            raise ValueError("Trace K changed inside one episode.")

        before = len(self.seen_workers)
        self.seen_workers.add(trace.worker_id)
        after = len(self.seen_workers)

        delta_coverage = (after - before) / self.k_true

        reward = (
            self.config.coverage_weight * delta_coverage
            - self.config.query_cost
        )

        completed_now = after == self.k_true and not self._completed
        if completed_now:
            reward += self.config.completion_bonus
            self._completed = True

        diagnostics = {
            "new_worker": float(after > before),
            "coverage": after / self.k_true,
            "delta_coverage": delta_coverage,
            "completed": float(completed_now),
        }
        return float(reward), diagnostics
