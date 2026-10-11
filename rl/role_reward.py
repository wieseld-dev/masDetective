from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Set, Tuple

from targets.hierarchical_mas import PrivilegedExecutionTrace


@dataclass(frozen=True)
class RoleRewardConfig:
    role_brier_weight: float = 2.00
    k_quality_weight: float = 0.50
    worker_coverage_weight: float = 0.30
    edge_coverage_weight: float = 0.15
    path_novelty_weight: float = 0.10
    repeat_path_penalty: float = 0.10
    query_cost: float = 0.02
    token_cost_per_1k: float = 0.005
    completion_bonus: float = 0.10
    role_delta_clip: float = 0.25
    k_delta_clip: float = 0.25


class RoleWhiteBoxReward:
    """Training-only grader; privileged role truth never enters the policy observation."""

    def __init__(self, candidate_roles, config: RoleRewardConfig | None = None):
        self.candidate_roles = tuple(candidate_roles)
        self.config = config or RoleRewardConfig()
        self.reset()

    def reset(self) -> None:
        self.seen_workers: Set[str] = set()
        self.seen_edges: Set[Tuple[str, str]] = set()
        self.seen_paths: Set[Tuple[str, ...]] = set()
        self._completed = False

    def prime(self, trace: PrivilegedExecutionTrace) -> None:
        """Register the mandatory anchor route without assigning policy credit for it."""
        self.seen_workers.add(trace.selected_worker)
        self.seen_edges.update(trace.path_edges)
        self.seen_paths.add(tuple(trace.path_nodes))

    def _brier(self, belief: Mapping[str, float], true_roles: Set[str]) -> float:
        vals = []
        for role in self.candidate_roles:
            y = 1.0 if role in true_roles else 0.0
            p = float(belief.get(role, 0.5))
            vals.append((p - y) ** 2)
        return sum(vals) / max(len(vals), 1)

    def step(
        self,
        trace: PrivilegedExecutionTrace,
        *,
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
        r_role_raw = b0 - b1
        r_role = max(-c.role_delta_clip, min(c.role_delta_clip, r_role_raw))

        ek0 = sum(float(prev_role_belief.get(r, 0.0)) for r in self.candidate_roles)
        ek1 = sum(float(new_role_belief.get(r, 0.0)) for r in self.candidate_roles)
        denom = max(len(self.candidate_roles), 1)
        kerr0 = abs(ek0 - k) / denom
        kerr1 = abs(ek1 - k) / denom
        r_k_raw = kerr0 - kerr1
        r_k = max(-c.k_delta_clip, min(c.k_delta_clip, r_k_raw))

        worker_before = len(self.seen_workers)
        self.seen_workers.add(trace.selected_worker)
        worker_delta = (len(self.seen_workers) - worker_before) / max(k, 1)

        edge_before = len(self.seen_edges)
        self.seen_edges.update(trace.path_edges)
        edge_delta = (len(self.seen_edges) - edge_before) / max(len(trace.all_edges), 1)

        path = tuple(trace.path_nodes)
        new_path = float(path not in self.seen_paths)
        repeat_path = float(path in self.seen_paths)
        self.seen_paths.add(path)

        approx_tokens = (len(query) + len(output)) / 4.0
        token_cost = c.token_cost_per_1k * (approx_tokens / 1000.0)

        reward = (
            c.role_brier_weight * r_role
            + c.k_quality_weight * r_k
            + c.worker_coverage_weight * worker_delta
            + c.edge_coverage_weight * edge_delta
            + c.path_novelty_weight * new_path
            - c.repeat_path_penalty * repeat_path
            - c.query_cost
            - token_cost
        )

        completed_now = len(self.seen_workers) == k and not self._completed
        if completed_now:
            reward += c.completion_bonus
            self._completed = True

        return float(reward), {
            "r_role_raw": float(r_role_raw),
            "r_role": float(r_role),
            "brier_before": float(b0),
            "brier_after": float(b1),
            "r_k_raw": float(r_k_raw),
            "r_k": float(r_k),
            "expected_k_before": float(ek0),
            "expected_k_after": float(ek1),
            "worker_coverage": len(self.seen_workers) / max(k, 1),
            "worker_delta": float(worker_delta),
            "edge_coverage": len(self.seen_edges) / max(len(trace.all_edges), 1),
            "edge_delta": float(edge_delta),
            "new_path": new_path,
            "repeat_path": repeat_path,
            "token_cost": float(token_cost),
            "completed": float(completed_now),
        }
