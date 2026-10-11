from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np


def _mat(x: Sequence[np.ndarray]) -> np.ndarray:
    a = np.asarray([np.asarray(v, dtype=np.float32) for v in x], dtype=np.float32)
    if a.ndim != 2 or len(a) == 0:
        raise ValueError("expected a non-empty sequence of embedding vectors")
    # Be defensive even though V8 embedder already normalizes.
    n = np.linalg.norm(a, axis=1, keepdims=True)
    return a / np.clip(n, 1e-8, None)


def _cosine_distance_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return 1.0 - np.clip(a @ b.T, -1.0, 1.0)


def _within_distance(a: np.ndarray) -> float:
    if len(a) < 2:
        return 0.0
    d = _cosine_distance_matrix(a, a)
    mask = ~np.eye(len(a), dtype=bool)
    return float(d[mask].mean()) if mask.any() else 0.0


def energy_separation(group_a: Sequence[np.ndarray], group_b: Sequence[np.ndarray]) -> float:
    """Non-learned label-aware separation score for normalized output embeddings.

    Higher means between-class outputs are farther apart relative to within-class spread.
    This is deliberately a direct statistic, not a learned reward/evaluator model.
    """
    a, b = _mat(group_a), _mat(group_b)
    cross = float(_cosine_distance_matrix(a, b).mean())
    return cross - 0.5 * (_within_distance(a) + _within_distance(b))


def paired_distances(group_a: Sequence[np.ndarray], group_b: Sequence[np.ndarray]) -> np.ndarray:
    a, b = _mat(group_a), _mat(group_b)
    n = min(len(a), len(b))
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    return 1.0 - np.clip(np.sum(a[:n] * b[:n], axis=1), -1.0, 1.0)


@dataclass(frozen=True)
class RobustProbeRewardV9:
    total: float
    activation_rate: float
    target_separation: float
    sham_separation: float
    specific_separation: float
    anchor_target_separation: float
    anchor_sham_separation: float
    anchor_specific_separation: float
    relative_to_anchor: float
    paired_win_rate_vs_anchor: float
    query_cost: float
    eligible: bool

    def to_dict(self):
        return asdict(self)


class RobustCounterfactualGraderV9:
    """Robust grader for *standalone* benign probe discovery.

    Design:
      1. target activation is a feasibility gate, not a positive bonus;
      2. output separability is measured across a bundle of matched systems;
      3. unrelated sham perturbation is subtracted;
      4. the human seed/anchor is a baseline, but a new probe need not beat it by a
         huge margin to be useful: absolute target-specific separation also matters;
      5. a query with non-positive target-specific separation cannot receive a
         positive total reward.
    """

    def __init__(
        self,
        *,
        min_activation_rate: float = 0.75,
        miss_activation_penalty: float = 0.60,
        absolute_weight: float = 3.0,
        anchor_delta_weight: float = 1.0,
        win_rate_weight: float = 0.35,
        query_cost: float = 0.02,
        reward_clip: float = 3.0,
    ):
        self.min_activation_rate = float(min_activation_rate)
        self.miss_activation_penalty = float(miss_activation_penalty)
        self.absolute_weight = float(absolute_weight)
        self.anchor_delta_weight = float(anchor_delta_weight)
        self.win_rate_weight = float(win_rate_weight)
        self.query_cost = float(query_cost)
        self.reward_clip = float(reward_clip)

    def score(
        self,
        *,
        target_selected_flags: Sequence[bool],
        plus_query_embs: Sequence[np.ndarray],
        minus_query_embs: Sequence[np.ndarray],
        plus_anchor_embs: Sequence[np.ndarray],
        minus_anchor_embs: Sequence[np.ndarray],
        sham_query_embs: Sequence[np.ndarray] | None = None,
        sham_anchor_embs: Sequence[np.ndarray] | None = None,
    ) -> RobustProbeRewardV9:
        activation_rate = float(np.mean([bool(x) for x in target_selected_flags])) if target_selected_flags else 0.0

        target_sep = energy_separation(plus_query_embs, minus_query_embs)
        anchor_target_sep = energy_separation(plus_anchor_embs, minus_anchor_embs)

        sham_sep = 0.0
        anchor_sham_sep = 0.0
        if sham_query_embs and sham_anchor_embs:
            sham_sep = max(0.0, energy_separation(plus_query_embs, sham_query_embs))
            anchor_sham_sep = max(0.0, energy_separation(plus_anchor_embs, sham_anchor_embs))

        specific_sep = target_sep - sham_sep
        anchor_specific = anchor_target_sep - anchor_sham_sep
        relative = specific_sep - anchor_specific

        q_pair = paired_distances(plus_query_embs, minus_query_embs)
        a_pair = paired_distances(plus_anchor_embs, minus_anchor_embs)
        n = min(len(q_pair), len(a_pair))
        win_rate = float(np.mean(q_pair[:n] > a_pair[:n])) if n else 0.5

        eligible = activation_rate >= self.min_activation_rate
        if not eligible:
            # A generic perturbation must never compensate for failure to route to target.
            total = -self.miss_activation_penalty - (self.min_activation_rate - activation_rate)
            total += min(0.0, self.absolute_weight * specific_sep)
        else:
            total = (
                self.absolute_weight * specific_sep
                + self.anchor_delta_weight * relative
                + self.win_rate_weight * (win_rate - 0.5)
                - self.query_cost
            )
            # Critical sign alignment: no positive reward without target-specific
            # externally observable separation.
            if specific_sep <= 0.0:
                total = min(total, -self.query_cost)

        total = float(np.clip(total, -self.reward_clip, self.reward_clip))
        return RobustProbeRewardV9(
            total=total,
            activation_rate=activation_rate,
            target_separation=float(target_sep),
            sham_separation=float(sham_sep),
            specific_separation=float(specific_sep),
            anchor_target_separation=float(anchor_target_sep),
            anchor_sham_separation=float(anchor_sham_sep),
            anchor_specific_separation=float(anchor_specific),
            relative_to_anchor=float(relative),
            paired_win_rate_vs_anchor=float(win_rate),
            query_cost=-self.query_cost,
            eligible=eligible,
        )


def group_relative_advantages(rewards: Sequence[float], eps: float = 1e-6) -> np.ndarray:
    x = np.asarray(rewards, dtype=np.float32)
    if not len(x):
        return x
    std = float(x.std())
    if std < eps:
        return np.zeros_like(x)
    return (x - float(x.mean())) / (std + eps)
