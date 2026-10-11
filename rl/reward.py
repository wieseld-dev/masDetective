from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Mapping, Set, Tuple

from targets.hierarchical_mas import PrivilegedExecutionTrace


@dataclass(frozen=True)
class RewardConfig:
    k_information_weight: float = 1.00
    worker_coverage_weight: float = 0.30
    edge_coverage_weight: float = 0.15
    path_novelty_weight: float = 0.10
    repeat_path_penalty: float = 0.10
    query_cost: float = 0.02
    token_cost_per_1k: float = 0.005
    completion_bonus: float = 0.10
    eps: float = 1e-6


class FullWhiteBoxReward:
    """
    Training-only grader.

    White-box state is used only to compute a scalar reward/diagnostics. The policy
    observation is constructed elsewhere from external query/output history only.
    """

    def __init__(self, config: RewardConfig | None = None):
        self.config = config or RewardConfig()
        self.reset()

    def reset(self) -> None:
        self.seen_workers: Set[str] = set()
        self.seen_edges: Set[Tuple[str, str]] = set()
        self.seen_paths: Set[Tuple[str, ...]] = set()
        self._completed = False

    def step(
        self,
        trace: PrivilegedExecutionTrace,
        *,
        prev_k_belief: Mapping[int, float],
        new_k_belief: Mapping[int, float],
        query: str,
        output: str,
    ) -> tuple[float, Dict[str, float]]:
        c = self.config
        k = int(trace.k_true)
        p0 = max(float(prev_k_belief.get(k, 0.0)), c.eps)
        p1 = max(float(new_k_belief.get(k, 0.0)), c.eps)
        r_k_raw = math.log(p1) - math.log(p0)
        r_k = max(-2.0, min(2.0, r_k_raw))

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
            c.k_information_weight * r_k
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

        diag = {
            "r_k_raw": float(r_k_raw),
            "r_k": float(r_k),
            "p_k_true_before": float(p0),
            "p_k_true_after": float(p1),
            "worker_coverage": len(self.seen_workers) / max(k, 1),
            "worker_delta": float(worker_delta),
            "edge_coverage": len(self.seen_edges) / max(len(trace.all_edges), 1),
            "edge_delta": float(edge_delta),
            "new_path": new_path,
            "repeat_path": repeat_path,
            "token_cost": float(token_cost),
            "completed": float(completed_now),
        }
        return float(reward), diag
