from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np


def _unit(v: np.ndarray) -> np.ndarray:
    x = np.asarray(v, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(x))
    return x / max(n, 1e-8)


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    aa = _unit(a)
    bb = _unit(b)
    return float(1.0 - np.clip(float(np.dot(aa, bb)), -1.0, 1.0))


def _q25(xs: Sequence[float]) -> float:
    if not xs:
        return 0.0
    return float(np.quantile(np.asarray(xs, dtype=np.float32), 0.25))


def _median(xs: Sequence[float]) -> float:
    return float(np.median(np.asarray(xs, dtype=np.float32))) if xs else 0.0


def _mean(xs: Sequence[float]) -> float:
    return float(np.mean(np.asarray(xs, dtype=np.float32))) if xs else 0.0


@dataclass(frozen=True)
class RobustProbeRewardV10:
    total: float
    eligible: bool

    # Robust routing feasibility. The target must be selected when it is present both
    # in the original PRESENT branch and in the SHAM branch where an unrelated role
    # was swapped. This avoids rewarding composition-fragile routing.
    activation_rate: float
    plus_activation_rate: float
    sham_activation_rate: float
    pair_count: int

    # Pairwise causal signal: distance(PRESENT, ABSENT) - distance(PRESENT, SHAM)
    target_pairwise: tuple[float, ...]
    sham_pairwise: tuple[float, ...]
    specific_pairwise: tuple[float, ...]
    specific_mean: float
    specific_median: float
    specific_q25: float
    specific_min: float
    specific_max: float
    robust_specific: float
    positive_specific_pair_rate: float

    # Anchor is diagnostic/reference only. It is NOT the primary reward objective.
    anchor_specific_pairwise: tuple[float, ...]
    delta_vs_anchor_pairwise: tuple[float, ...]
    delta_mean: float
    delta_median: float
    delta_q25: float
    beat_anchor_pair_rate: float

    semantic_max_similarity: float
    novelty_penalty: float
    query_cost: float

    strict_specific_pass: bool
    strict_delta_pass: bool

    def to_dict(self) -> dict:
        return asdict(self)


class PairwiseRobustCounterfactualGraderV10:
    """Robust counterfactual reward for cross-composition benign probe discovery.

    Design:
      * Every matched composition is scored independently.
      * Pair i signal is d(PRESENT_i, ABSENT_i) - d(PRESENT_i, SHAM_i).
      * Robust aggregation uses median + lower quartile, so one lucky system cannot
        dominate a probe's reward.
      * The target must route reliably in both PRESENT and SHAM branches.
      * The human anchor is reported as a baseline rather than a hurdle.
      * Semantic novelty is a soft penalty after routing feasibility is met.
    """

    def __init__(
        self,
        *,
        min_activation_rate: float = 0.75,
        miss_activation_penalty: float = 0.75,
        signal_weight: float = 4.0,
        consistency_weight: float = 0.30,
        query_cost: float = 0.02,
        novelty_free_similarity: float = 0.90,
        novelty_weight: float = 0.25,
        reward_clip: float = 3.0,
    ):
        self.min_activation_rate = float(min_activation_rate)
        self.miss_activation_penalty = float(miss_activation_penalty)
        self.signal_weight = float(signal_weight)
        self.consistency_weight = float(consistency_weight)
        self.query_cost = float(query_cost)
        self.novelty_free_similarity = float(novelty_free_similarity)
        self.novelty_weight = float(novelty_weight)
        self.reward_clip = float(reward_clip)

    def _novelty_penalty(self, similarity: float) -> float:
        s = float(np.clip(similarity, 0.0, 1.0))
        if s <= self.novelty_free_similarity:
            return 0.0
        denom = max(1e-6, 1.0 - self.novelty_free_similarity)
        return float(self.novelty_weight * (s - self.novelty_free_similarity) / denom)

    def score(
        self,
        *,
        target_selected_flags: Sequence[bool],
        sham_target_selected_flags: Sequence[bool] | None = None,
        plus_query_embs: Sequence[np.ndarray],
        minus_query_embs: Sequence[np.ndarray],
        plus_anchor_embs: Sequence[np.ndarray],
        minus_anchor_embs: Sequence[np.ndarray],
        sham_query_embs: Sequence[np.ndarray] | None = None,
        sham_anchor_embs: Sequence[np.ndarray] | None = None,
        semantic_max_similarity: float = 0.0,
    ) -> RobustProbeRewardV10:
        n = min(
            len(plus_query_embs),
            len(minus_query_embs),
            len(plus_anchor_embs),
            len(minus_anchor_embs),
        )
        if n <= 0:
            raise ValueError("V10 reward needs at least one matched pair")

        have_sham_q = sham_query_embs is not None and len(sham_query_embs) >= n
        have_sham_a = sham_anchor_embs is not None and len(sham_anchor_embs) >= n

        target_pair: list[float] = []
        sham_pair: list[float] = []
        specific_pair: list[float] = []
        anchor_specific_pair: list[float] = []
        delta_pair: list[float] = []

        for i in range(n):
            t = cosine_distance(plus_query_embs[i], minus_query_embs[i])
            sh = cosine_distance(plus_query_embs[i], sham_query_embs[i]) if have_sham_q else 0.0
            sp = t - sh

            at = cosine_distance(plus_anchor_embs[i], minus_anchor_embs[i])
            ash = cosine_distance(plus_anchor_embs[i], sham_anchor_embs[i]) if have_sham_a else 0.0
            asp = at - ash

            target_pair.append(float(t))
            sham_pair.append(float(sh))
            specific_pair.append(float(sp))
            anchor_specific_pair.append(float(asp))
            delta_pair.append(float(sp - asp))

        plus_flags = [bool(x) for x in target_selected_flags[:n]]
        plus_activation = float(np.mean(plus_flags)) if plus_flags else 0.0

        if sham_target_selected_flags is not None and len(sham_target_selected_flags) >= n:
            sham_flags = [bool(x) for x in sham_target_selected_flags[:n]]
            sham_activation = float(np.mean(sham_flags)) if sham_flags else 0.0
        else:
            # Backward-compatible fallback for tests/callers without SHAM truth.
            sham_activation = plus_activation

        robust_activation = min(plus_activation, sham_activation)

        spec_mean = _mean(specific_pair)
        spec_med = _median(specific_pair)
        spec_q25 = _q25(specific_pair)
        robust_specific = 0.5 * spec_med + 0.5 * spec_q25
        positive_pair_rate = float(np.mean(np.asarray(specific_pair) > 0.0))

        delta_mean = _mean(delta_pair)
        delta_med = _median(delta_pair)
        delta_q25 = _q25(delta_pair)
        beat_anchor_pair_rate = float(np.mean(np.asarray(delta_pair) > 0.0))

        eligible = robust_activation >= self.min_activation_rate
        novelty_penalty = self._novelty_penalty(semantic_max_similarity) if eligible else 0.0

        if not eligible:
            total = -self.miss_activation_penalty - (self.min_activation_rate - robust_activation)
            total += min(0.0, self.signal_weight * robust_specific)
        else:
            total = (
                self.signal_weight * robust_specific
                + self.consistency_weight * (positive_pair_rate - 0.5)
                - novelty_penalty
                - self.query_cost
            )
            if robust_specific <= 0.0:
                total = min(total, -self.query_cost - novelty_penalty)

        strict_specific = bool(
            eligible
            and spec_med > 0.0
            and spec_q25 >= 0.0
            and positive_pair_rate >= 0.75
        )
        strict_delta = bool(
            strict_specific
            and delta_med > 0.0
            and delta_q25 >= 0.0
            and beat_anchor_pair_rate >= 0.75
        )

        total = float(np.clip(total, -self.reward_clip, self.reward_clip))
        return RobustProbeRewardV10(
            total=total,
            eligible=eligible,
            activation_rate=float(robust_activation),
            plus_activation_rate=float(plus_activation),
            sham_activation_rate=float(sham_activation),
            pair_count=n,
            target_pairwise=tuple(target_pair),
            sham_pairwise=tuple(sham_pair),
            specific_pairwise=tuple(specific_pair),
            specific_mean=spec_mean,
            specific_median=spec_med,
            specific_q25=spec_q25,
            specific_min=float(min(specific_pair)),
            specific_max=float(max(specific_pair)),
            robust_specific=float(robust_specific),
            positive_specific_pair_rate=positive_pair_rate,
            anchor_specific_pairwise=tuple(anchor_specific_pair),
            delta_vs_anchor_pairwise=tuple(delta_pair),
            delta_mean=delta_mean,
            delta_median=delta_med,
            delta_q25=delta_q25,
            beat_anchor_pair_rate=beat_anchor_pair_rate,
            semantic_max_similarity=float(semantic_max_similarity),
            novelty_penalty=float(novelty_penalty),
            query_cost=-self.query_cost,
            strict_specific_pass=strict_specific,
            strict_delta_pass=strict_delta,
        )


def group_relative_advantages(rewards: Sequence[float], eps: float = 1e-6) -> np.ndarray:
    x = np.asarray(rewards, dtype=np.float32)
    if not len(x):
        return x
    std = float(x.std())
    if std < eps:
        return np.zeros_like(x)
    return (x - float(x.mean())) / (std + eps)
