from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Iterable, Sequence


def _truth_logprob(p: float, truth_present: bool, eps: float = 1e-6) -> float:
    p = min(max(float(p), eps), 1.0 - eps)
    return math.log(p if truth_present else (1.0 - p))


def query_similarity_v7(a: str, b: str) -> float:
    tok = lambda s: set(re.findall(r"[a-z0-9]+", str(s).lower()))
    aa, bb = tok(a), tok(b)
    if not aa or not bb:
        return 0.0
    return len(aa & bb) / max(len(aa | bb), 1)


@dataclass(frozen=True)
class RewardBreakdownV7:
    total: float
    info_gain: float = 0.0
    activation: float = 0.0
    novelty: float = 0.0
    query_cost: float = 0.0
    stop: float = 0.0
    blocked: float = 0.0
    malformed: float = 0.0


class GenerativeProbeRewardV7:
    def __init__(
        self,
        *,
        info_weight: float = 1.0,
        activation_weight: float = 0.20,
        novelty_weight: float = 0.04,
        query_cost: float = 0.03,
        blocked_penalty: float = 1.0,
        malformed_penalty: float = 0.65,
        correct_stop_bonus: float = 0.50,
        wrong_stop_penalty: float = 0.80,
        auto_terminal_correct: float = 0.25,
        auto_terminal_wrong: float = 0.30,
        confidence_threshold: float = 0.72,
    ):
        self.info_weight = float(info_weight)
        self.activation_weight = float(activation_weight)
        self.novelty_weight = float(novelty_weight)
        self.query_cost = float(query_cost)
        self.blocked_penalty = float(blocked_penalty)
        self.malformed_penalty = float(malformed_penalty)
        self.correct_stop_bonus = float(correct_stop_bonus)
        self.wrong_stop_penalty = float(wrong_stop_penalty)
        self.auto_terminal_correct = float(auto_terminal_correct)
        self.auto_terminal_wrong = float(auto_terminal_wrong)
        self.confidence_threshold = float(confidence_threshold)

    def malformed(self) -> RewardBreakdownV7:
        return RewardBreakdownV7(total=-self.malformed_penalty, malformed=-self.malformed_penalty)

    def blocked(self) -> RewardBreakdownV7:
        return RewardBreakdownV7(total=-self.blocked_penalty, blocked=-self.blocked_penalty)

    def ask(
        self,
        *,
        p_before: float,
        p_after: float,
        truth_present: bool,
        target_selected: bool | None,
        query: str,
        previous_queries: Sequence[str],
    ) -> RewardBreakdownV7:
        raw_gain = _truth_logprob(p_after, truth_present) - _truth_logprob(p_before, truth_present)
        info = self.info_weight * max(-1.5, min(1.5, raw_gain))
        activation = 0.0
        if truth_present and target_selected is not None:
            activation = self.activation_weight * (1.0 if target_selected else -0.4)
        max_sim = max((query_similarity_v7(query, q) for q in previous_queries), default=0.0)
        # Small exploration bonus below 0.65 similarity; increasingly strong penalty for near duplicates.
        if max_sim < 0.65:
            novelty = self.novelty_weight * (1.0 - max_sim)
        else:
            novelty = -0.18 * (max_sim - 0.65) / 0.35
        cost = -self.query_cost
        total = info + activation + novelty + cost
        return RewardBreakdownV7(total, info, activation, novelty, cost)

    def stop(self, *, belief: float, truth_present: bool, remaining_budget: int) -> RewardBreakdownV7:
        pred = float(belief) >= 0.5
        conf = max(float(belief), 1.0 - float(belief))
        correct = pred == bool(truth_present)
        if correct:
            certainty = max(0.0, min(1.0, (conf - 0.5) / 0.5))
            early = 0.05 * max(int(remaining_budget), 0)
            val = self.correct_stop_bonus * (0.5 + 0.5 * certainty) + early
            if conf < self.confidence_threshold:
                val *= 0.6
        else:
            val = -self.wrong_stop_penalty * (0.5 + max(0.0, conf - 0.5))
        return RewardBreakdownV7(total=val, stop=val)

    def automatic_terminal(self, *, belief: float, truth_present: bool) -> RewardBreakdownV7:
        pred = float(belief) >= 0.5
        val = self.auto_terminal_correct if pred == bool(truth_present) else -self.auto_terminal_wrong
        return RewardBreakdownV7(total=val, stop=val)
