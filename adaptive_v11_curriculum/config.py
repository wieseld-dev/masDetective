from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class StageConfig:
    stage: int
    name: str
    k_choices: Tuple[int, ...]
    modes: Tuple[str, ...]
    horizon: int
    train_systems_per_cycle: int
    dev_systems: int
    cycles: int = 2
    group_size: int = 6
    semantic_rule: str = "any"
    lr: float = 1e-5
    kl_beta: float = 0.02
    retention_weight: float = 0.08
    mc_weight: float = 0.35


STAGES = (
    StageConfig(1, "easy_distinct_3", (3,), ("single", "fanout2"), 2, 16, 8,
                semantic_rule="distinct", lr=1.0e-5, kl_beta=0.025, retention_weight=0.10, mc_weight=0.20),
    StageConfig(2, "easy_distinct_3_4", (3,4), ("single", "fanout2"), 3, 20, 8,
                semantic_rule="distinct", lr=1.0e-5, kl_beta=0.025, retention_weight=0.10, mc_weight=0.25),
    StageConfig(3, "medium_overlap_4_5", (4,5), ("fanout2", "fanout3"), 4, 24, 10,
                semantic_rule="one_confuser", lr=8.0e-6, kl_beta=0.03, retention_weight=0.10, mc_weight=0.30),
    StageConfig(4, "hard_overlap_5_6", (5,6), ("fanout2", "fanout3", "mixed"), 5, 28, 10,
                semantic_rule="multi_confuser", lr=6.0e-6, kl_beta=0.035, retention_weight=0.12, mc_weight=0.35),
    StageConfig(5, "large_mixed_6_8", (6,7,8), ("single", "fanout2", "fanout3", "mixed"), 6, 32, 12,
                semantic_rule="multi_confuser", lr=5.0e-6, kl_beta=0.04, retention_weight=0.12, mc_weight=0.40),
)


def get_stage(stage: int) -> StageConfig:
    for s in STAGES:
        if s.stage == int(stage):
            return s
    raise KeyError(stage)


def estimated_target_queries() -> int:
    train = sum(s.cycles * s.train_systems_per_cycle * s.horizon * s.group_size for s in STAGES)
    dev = sum(s.dev_systems * s.horizon for s in STAGES)
    return train + dev
