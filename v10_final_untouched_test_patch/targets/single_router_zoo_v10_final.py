from __future__ import annotations

from collections import Counter
from typing import List, Tuple

from targets.single_router_fanout_mas import SingleRouterSystemSpec
from targets.single_router_zoo_v65 import TRAIN_NEW, VALIDATION, V63_TRAIN_COMPOSITIONS

# V10 FINAL UNTOUCHED PROBE-GENERALIZATION TEST
#
# Frozen before any result on this set is observed. The 16 compositions below were
# selected by a deterministic combinatorial design using ONLY structural constraints:
#   * no role composition overlaps V6.3, V6.5 train, or V6.5 development validation;
#   * exactly one base system for every (execution_mode, K) in
#       {single, fanout2, fanout3, mixed} x {3,4,5,6};
#   * every one of the 8 candidate roles occurs exactly 9 times;
#   * every role is represented in every execution mode and every K value.
# No V9/V10 performance numbers were used to choose these systems.
# Design seed recorded for provenance: 20260909.

FINAL_TEST: Tuple[Tuple[str, int, Tuple[str, ...]], ...] = (
    ("single", 3, ("research", "data", "operations")),
    ("single", 4, ("math", "research", "data", "design")),
    ("single", 5, ("code", "writing", "planning", "data", "operations")),
    ("single", 6, ("code", "writing", "math", "planning", "data", "operations")),

    ("fanout2", 3, ("code", "math", "design")),
    ("fanout2", 4, ("writing", "planning", "research", "design")),
    ("fanout2", 5, ("code", "writing", "planning", "research", "operations")),
    ("fanout2", 6, ("code", "math", "research", "data", "design", "operations")),

    ("fanout3", 3, ("math", "research", "operations")),
    ("fanout3", 4, ("code", "writing", "planning", "design")),
    ("fanout3", 5, ("math", "research", "data", "design", "operations")),
    ("fanout3", 6, ("code", "writing", "research", "data", "design", "operations")),

    ("mixed", 3, ("writing", "math", "planning")),
    ("mixed", 4, ("math", "planning", "design", "operations")),
    ("mixed", 5, ("code", "writing", "math", "planning", "data")),
    ("mixed", 6, ("code", "writing", "planning", "research", "data", "design")),
)

DESIGN_SEED = 20260909
CANDIDATE_ROLES = (
    "code", "writing", "math", "planning", "research", "data", "design", "operations"
)


def _key(roles) -> tuple[str, ...]:
    return tuple(sorted(map(str, roles)))


def _specs(rows) -> List[SingleRouterSystemSpec]:
    styles = ("s1", "s2", "s3")
    out: List[SingleRouterSystemSpec] = []
    for i, (mode, k, roles) in enumerate(rows):
        if len(roles) != k:
            raise ValueError(f"declared K={k}, actual roles={len(roles)}")
        out.append(
            SingleRouterSystemSpec(
                system_id=f"v10_final_{i:02d}_{mode}_k{k}",
                split="final_untouched",
                execution_mode=mode,
                style=styles[i % len(styles)],
                worker_roles=tuple(roles),
            )
        )
    return out


FINAL_TEST_SPECS = _specs(FINAL_TEST)


def get_v10_final_specs() -> List[SingleRouterSystemSpec]:
    return list(FINAL_TEST_SPECS)


def validate_final_design() -> None:
    if len(FINAL_TEST) != 16:
        raise AssertionError("final test must contain exactly 16 base systems")

    keys = [_key(x[2]) for x in FINAL_TEST]
    if len(set(keys)) != 16:
        raise AssertionError("final test compositions must be unique")

    prior = {_key(x[2]) for x in TRAIN_NEW}
    prior |= {_key(x[2]) for x in VALIDATION}
    prior |= {_key(x) for x in V63_TRAIN_COMPOSITIONS}
    overlap = set(keys) & prior
    if overlap:
        raise AssertionError(f"final-test composition leakage: {sorted(overlap)}")

    expected_slots = {(m, k) for m in ("single", "fanout2", "fanout3", "mixed") for k in (3, 4, 5, 6)}
    actual_slots = {(m, k) for m, k, _ in FINAL_TEST}
    if actual_slots != expected_slots:
        raise AssertionError(f"mode/K coverage mismatch: missing={sorted(expected_slots-actual_slots)}")

    counts = Counter(r for _, _, rs in FINAL_TEST for r in rs)
    if any(counts[r] != 9 for r in CANDIDATE_ROLES):
        raise AssertionError(f"role prevalence is not exactly balanced at 9 each: {dict(counts)}")

    for role in CANDIDATE_ROLES:
        modes = {m for m, _, rs in FINAL_TEST if role in rs}
        ks = {k for _, k, rs in FINAL_TEST if role in rs}
        if modes != {"single", "fanout2", "fanout3", "mixed"}:
            raise AssertionError(f"{role} missing execution-mode coverage: {modes}")
        if ks != {3, 4, 5, 6}:
            raise AssertionError(f"{role} missing K coverage: {ks}")


validate_final_design()
