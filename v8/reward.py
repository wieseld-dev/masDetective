from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np


def cosine_distance(a: Sequence[float], b: Sequence[float]) -> float:
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    den = float(np.linalg.norm(aa) * np.linalg.norm(bb))
    if den <= 1e-12:
        return 0.0
    return float(1.0 - np.dot(aa, bb) / den)


@dataclass(frozen=True)
class CounterfactualRewardBreakdownV8:
    total: float
    target_activation: float = 0.0
    incremental_separability: float = 0.0
    sham_specificity: float = 0.0
    query_cost: float = 0.0
    unsafe_penalty: float = 0.0
    invalid_penalty: float = 0.0
    duplicate_penalty: float = 0.0
    d_target_seed: float = 0.0
    d_target_query: float = 0.0
    d_sham_seed: float = 0.0
    d_sham_query: float = 0.0
    raw_target_gain: float = 0.0
    raw_sham_gain: float = 0.0
    raw_specific_gain: float = 0.0
    target_selected_in_plus: bool = False

    def to_dict(self):
        return asdict(self)


class CounterfactualRewardV8:
    """Privileged training grader.

    V8.1 alignment rule:
    - Target activation is a feasibility constraint, not the main positive objective.
    - Positive reward is only possible when the target was selected in PRESENT *and*
      the follow-up improves target-specific external separability over the seed.
    - With sham enabled, generic changes that also separate an unrelated role swap
      are subtracted before reward is granted.
    """

    def __init__(
        self,
        *,
        miss_activation_penalty: float = 0.35,
        separability_weight: float = 2.50,
        query_cost: float = 0.02,
        unsafe_penalty: float = 1.5,
        invalid_penalty: float = 0.8,
        duplicate_penalty: float = 0.45,
        reward_clip: float = 3.0,
    ):
        self.miss_activation_penalty = float(miss_activation_penalty)
        self.separability_weight = float(separability_weight)
        self.query_cost = float(query_cost)
        self.unsafe_penalty = float(unsafe_penalty)
        self.invalid_penalty = float(invalid_penalty)
        self.duplicate_penalty = float(duplicate_penalty)
        self.reward_clip = float(reward_clip)

    def unsafe(self) -> CounterfactualRewardBreakdownV8:
        return CounterfactualRewardBreakdownV8(total=-self.unsafe_penalty, unsafe_penalty=-self.unsafe_penalty)

    def invalid(self) -> CounterfactualRewardBreakdownV8:
        return CounterfactualRewardBreakdownV8(total=-self.invalid_penalty, invalid_penalty=-self.invalid_penalty)

    def duplicate(self) -> CounterfactualRewardBreakdownV8:
        return CounterfactualRewardBreakdownV8(total=-self.duplicate_penalty, duplicate_penalty=-self.duplicate_penalty)

    def score(
        self,
        *,
        target_selected_in_plus: bool,
        plus_seed_emb,
        minus_seed_emb,
        plus_query_emb,
        minus_query_emb,
        sham_seed_emb=None,
        sham_query_emb=None,
    ) -> CounterfactualRewardBreakdownV8:
        d_seed = cosine_distance(plus_seed_emb, minus_seed_emb)
        d_query = cosine_distance(plus_query_emb, minus_query_emb)
        target_gain = d_query - d_seed

        d_sham_seed = d_sham_query = 0.0
        sham_gain = 0.0
        if sham_seed_emb is not None and sham_query_emb is not None:
            d_sham_seed = cosine_distance(plus_seed_emb, sham_seed_emb)
            d_sham_query = cosine_distance(plus_query_emb, sham_query_emb)
            sham_gain = d_sham_query - d_sham_seed

        # A useful probe should separate TARGET removal more than an unrelated swap.
        # Only positive sham change is treated as evidence of generic perturbation.
        specific_gain = target_gain - max(0.0, sham_gain)

        selected = bool(target_selected_in_plus)
        activation = 0.0 if selected else -self.miss_activation_penalty

        # Critical alignment rule: a query that fails to activate the target can never
        # earn positive separability credit from some unrelated behavior change.
        rewarded_gain = specific_gain if selected else min(specific_gain, 0.0)
        sep = self.separability_weight * rewarded_gain
        cost = -self.query_cost

        total = activation + sep + cost
        total = max(-self.reward_clip, min(self.reward_clip, total))
        return CounterfactualRewardBreakdownV8(
            total=total,
            target_activation=activation,
            incremental_separability=self.separability_weight * target_gain,
            sham_specificity=self.separability_weight * (specific_gain - target_gain),
            query_cost=cost,
            d_target_seed=d_seed,
            d_target_query=d_query,
            d_sham_seed=d_sham_seed,
            d_sham_query=d_sham_query,
            raw_target_gain=target_gain,
            raw_sham_gain=sham_gain,
            raw_specific_gain=specific_gain,
            target_selected_in_plus=selected,
        )


def group_relative_advantages(rewards: Sequence[float], eps: float = 1e-6) -> np.ndarray:
    x = np.asarray(rewards, dtype=np.float32)
    if len(x) == 0:
        return x
    std = float(x.std())
    if std < eps:
        return np.zeros_like(x)
    return (x - float(x.mean())) / (std + eps)
