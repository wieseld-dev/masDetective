from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Mapping, Sequence


EPS = 1e-5
FORBIDDEN_SURFACE = re.compile(
    r"\b(system prompt|developer message|hidden instruction|hidden prompt|chain[- ]of[- ]thought|"
    r"internal reasoning|agent|worker|routing|router|topology|hidden architecture|private state|"
    r"what roles|which roles|which specialists|ignore previous|override|bypass|jailbreak|reveal your)\b",
    flags=re.IGNORECASE,
)


def clip01(p: float) -> float:
    return min(1.0 - EPS, max(EPS, float(p)))


def _log_score_binary(y: int, p: float) -> float:
    p = clip01(p)
    return math.log(p) if int(y) else math.log1p(-p)


def _f1(pred: set[str], truth: set[str]) -> float:
    if not pred and not truth:
        return 1.0
    if not pred or not truth:
        return 0.0
    tp = len(pred & truth)
    precision = tp / len(pred)
    recall = tp / len(truth)
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def map_k(k_probs: Mapping[int, float]) -> int:
    mean = sum(int(k) * float(p) for k, p in k_probs.items())
    return min(k_probs, key=lambda k: (-float(k_probs[k]), abs(int(k) - mean), int(k)))


def predicted_set(role_probs: Mapping[str, float], k_probs: Mapping[int, float]) -> set[str]:
    k = int(map_k(k_probs))
    return {r for r, _ in sorted(role_probs.items(), key=lambda kv: kv[1], reverse=True)[:k]}


def normalized_word_jaccard(a: str, b: str) -> float:
    wa = set(re.findall(r"[a-z0-9]+", str(a).lower()))
    wb = set(re.findall(r"[a-z0-9]+", str(b).lower()))
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / max(1, len(wa | wb))


def surface_valid(query: str) -> tuple[bool, str]:
    q = " ".join(str(query).split()).strip()
    if not q:
        return False, "empty"
    if not q.endswith("?"):
        return False, "not_question"
    if q.count("?") != 1:
        return False, "multiple_questions"
    if FORBIDDEN_SURFACE.search(q):
        return False, "forbidden_internal_language"
    if len(q) > 1200:
        return False, "too_long"
    return True, ""


@dataclass(frozen=True)
class RewardBreakdown:
    total: float
    k_log_score_improvement: float
    role_log_score_improvement: float
    set_f1_improvement: float
    repetition_penalty: float
    length_penalty: float
    invalid_penalty: float
    valid: bool
    invalid_reason: str = ""

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def adaptive_reward(
    *,
    old_k_probs: Mapping[int, float],
    new_k_probs: Mapping[int, float],
    old_role_probs: Mapping[str, float],
    new_role_probs: Mapping[str, float],
    true_k: int,
    true_roles: Sequence[str],
    query: str,
    previous_queries: Sequence[str] = (),
    generator_valid: bool = True,
    w_k: float = 0.55,
    w_roles: float = 0.35,
    w_set: float = 0.10,
    reward_clip: float = 3.0,
) -> RewardBreakdown:
    ok, reason = surface_valid(query)
    if not generator_valid:
        ok = False
        reason = reason or "generator_invalid"
    if not ok:
        return RewardBreakdown(
            total=-1.5,
            k_log_score_improvement=0.0,
            role_log_score_improvement=0.0,
            set_f1_improvement=0.0,
            repetition_penalty=0.0,
            length_penalty=0.0,
            invalid_penalty=1.5,
            valid=False,
            invalid_reason=reason,
        )

    old_pk = clip01(old_k_probs.get(int(true_k), EPS))
    new_pk = clip01(new_k_probs.get(int(true_k), EPS))
    rk = math.log(new_pk) - math.log(old_pk)

    truth = set(map(str, true_roles))
    roles = tuple(sorted(old_role_probs))
    old_rs = sum(_log_score_binary(int(r in truth), old_role_probs[r]) for r in roles) / max(1, len(roles))
    new_rs = sum(_log_score_binary(int(r in truth), new_role_probs[r]) for r in roles) / max(1, len(roles))
    rr = new_rs - old_rs

    old_f1 = _f1(predicted_set(old_role_probs, old_k_probs), truth)
    new_f1 = _f1(predicted_set(new_role_probs, new_k_probs), truth)
    rs = new_f1 - old_f1

    sim = max((normalized_word_jaccard(query, q) for q in previous_queries), default=0.0)
    repetition_penalty = 0.0 if sim <= 0.65 else min(0.35, (sim - 0.65) * 0.8)
    words = len(re.findall(r"\b\w+\b", query))
    length_penalty = max(0, words - 70) * 0.002

    total = w_k * rk + w_roles * rr + w_set * rs - repetition_penalty - length_penalty
    total = max(-reward_clip, min(reward_clip, total))
    return RewardBreakdown(
        total=float(total),
        k_log_score_improvement=float(rk),
        role_log_score_improvement=float(rr),
        set_f1_improvement=float(rs),
        repetition_penalty=float(repetition_penalty),
        length_penalty=float(length_penalty),
        invalid_penalty=0.0,
        valid=True,
        invalid_reason="",
    )


def group_relative_advantages(rewards: Sequence[float], eps: float = 1e-6) -> list[float]:
    xs = [float(x) for x in rewards]
    if not xs:
        return []
    mean = sum(xs) / len(xs)
    var = sum((x - mean) ** 2 for x in xs) / len(xs)
    sd = math.sqrt(var)
    if sd < eps:
        return [0.0 for _ in xs]
    return [(x - mean) / (sd + eps) for x in xs]
