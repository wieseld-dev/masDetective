from __future__ import annotations

from dataclasses import replace

from adaptive_v11_curriculum.config import STAGES as BASE_STAGES

if len(BASE_STAGES) < 5:
    raise RuntimeError(f"expected at least 5 base curriculum stages, got {len(BASE_STAGES)}")

b1, b2, b3, b4, b5 = BASE_STAGES[:5]

# Deliberately gradual: each semantic difficulty is seen twice, with a longer horizon
# and more systems on the second pass. Hidden/base stage fields are preserved by replace().
STAGES = (
    replace(b1, stage=1,  horizon=2, train_systems_per_cycle=24, dev_systems=12, cycles=2, group_size=6,
            lr=8.0e-6, kl_beta=0.025, retention_weight=0.12, mc_weight=0.10),
    replace(b1, stage=2,  horizon=3, train_systems_per_cycle=28, dev_systems=12, cycles=2, group_size=6,
            lr=7.0e-6, kl_beta=0.028, retention_weight=0.12, mc_weight=0.15),
    replace(b2, stage=3,  horizon=3, train_systems_per_cycle=28, dev_systems=12, cycles=2, group_size=6,
            lr=6.0e-6, kl_beta=0.030, retention_weight=0.13, mc_weight=0.18),
    replace(b2, stage=4,  horizon=4, train_systems_per_cycle=32, dev_systems=14, cycles=2, group_size=6,
            lr=5.0e-6, kl_beta=0.032, retention_weight=0.13, mc_weight=0.22),
    replace(b3, stage=5,  horizon=4, train_systems_per_cycle=32, dev_systems=14, cycles=2, group_size=8,
            lr=4.5e-6, kl_beta=0.035, retention_weight=0.14, mc_weight=0.25),
    replace(b3, stage=6,  horizon=5, train_systems_per_cycle=36, dev_systems=14, cycles=2, group_size=8,
            lr=4.0e-6, kl_beta=0.038, retention_weight=0.14, mc_weight=0.28),
    replace(b4, stage=7,  horizon=5, train_systems_per_cycle=36, dev_systems=16, cycles=2, group_size=8,
            lr=3.5e-6, kl_beta=0.042, retention_weight=0.15, mc_weight=0.32),
    replace(b4, stage=8,  horizon=6, train_systems_per_cycle=40, dev_systems=16, cycles=2, group_size=8,
            lr=3.0e-6, kl_beta=0.046, retention_weight=0.15, mc_weight=0.36),
    replace(b5, stage=9,  horizon=7, train_systems_per_cycle=44, dev_systems=18, cycles=2, group_size=8,
            lr=2.5e-6, kl_beta=0.052, retention_weight=0.16, mc_weight=0.40),
    replace(b5, stage=10, horizon=8, train_systems_per_cycle=48, dev_systems=18, cycles=2, group_size=8,
            lr=2.0e-6, kl_beta=0.060, retention_weight=0.18, mc_weight=0.45),
)


def planned_train_candidate_executions() -> int:
    return int(sum(s.cycles * s.train_systems_per_cycle * s.horizon * s.group_size for s in STAGES))


def validate_curriculum() -> None:
    stage_ids = [int(s.stage) for s in STAGES]
    if stage_ids != list(range(1, len(STAGES) + 1)):
        raise RuntimeError(f"stage ids must be contiguous: {stage_ids}")
    horizons = [int(s.horizon) for s in STAGES]
    if any(b < a for a, b in zip(horizons, horizons[1:])):
        raise RuntimeError(f"horizons must be nondecreasing: {horizons}")
    train_n = [int(s.train_systems_per_cycle) for s in STAGES]
    if any(b < a for a, b in zip(train_n, train_n[1:])):
        raise RuntimeError(f"train system counts must be nondecreasing: {train_n}")
    for s in STAGES:
        if int(s.group_size) < 4:
            raise RuntimeError(f"group_size too small at stage {s.stage}: {s.group_size}")
        if float(s.lr) <= 0 or float(s.kl_beta) < 0:
            raise RuntimeError(f"invalid optimization config at stage {s.stage}")


validate_curriculum()
