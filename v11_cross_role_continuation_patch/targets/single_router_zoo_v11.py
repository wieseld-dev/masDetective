from __future__ import annotations

import random
from collections import Counter
from typing import Iterable, List, Sequence, Tuple

from targets.hierarchical_mas import ROLE_PROMPTS
from targets.single_router_fanout_mas import SingleRouterSystemSpec
from v11.role_universe import (
    ALL_ROLES,
    DEV_UNSEEN_ROLES,
    EXTRA_ROLE_PROMPTS,
    TRAIN_ROLES,
)

# Register only environment-internal prompts. SingleRouterFanoutMAS imports the same
# ROLE_PROMPTS dictionary object, so this safely extends the runtime without exposing
# prompts to the Detective policy.
ROLE_PROMPTS.update(EXTRA_ROLE_PROMPTS)

MODES: Tuple[str, ...] = ("single", "fanout2", "fanout3", "mixed")
KS: Tuple[int, ...] = (3, 4, 5, 6)


def _composition_key(roles: Sequence[str]) -> tuple[str, ...]:
    return tuple(sorted(map(str, roles)))


def _choose_balanced(
    pool: Sequence[str],
    counts: Counter,
    k: int,
    rng: random.Random,
    *,
    must_include: Sequence[str] = (),
) -> tuple[str, ...]:
    chosen: list[str] = []
    for r in must_include:
        if r in pool and r not in chosen:
            chosen.append(r)
    while len(chosen) < k:
        remaining = [r for r in pool if r not in chosen]
        min_count = min(counts[r] for r in remaining)
        band = [r for r in remaining if counts[r] <= min_count + 1]
        chosen.append(rng.choice(band))
    return tuple(chosen)


def _make_train_specs(per_mode: int = 20, seed: int = 11101) -> List[SingleRouterSystemSpec]:
    rng = random.Random(seed)
    counts_by_mode = {m: Counter({r: 0 for r in TRAIN_ROLES}) for m in MODES}
    global_seen: set[tuple[str, ...]] = set()
    out: list[SingleRouterSystemSpec] = []
    styles = ("s1", "s2", "s3")

    for mi, mode in enumerate(MODES):
        counts = counts_by_mode[mode]
        for i in range(per_mode):
            k = KS[(i + mi) % len(KS)]
            roles = None
            for attempt in range(500):
                trial = _choose_balanced(TRAIN_ROLES, counts, k, rng)
                key = _composition_key(trial)
                if key not in global_seen:
                    roles = trial
                    break
                # perturb deterministic RNG state and retry
                rng.random()
            if roles is None:
                raise RuntimeError("could not generate unique balanced V11 training composition")
            global_seen.add(_composition_key(roles))
            counts.update(roles)
            out.append(
                SingleRouterSystemSpec(
                    system_id=f"v11_train_{mode}_{i:02d}_k{k}",
                    split="v11_train",
                    execution_mode=mode,
                    style=styles[(i + mi) % len(styles)],
                    worker_roles=tuple(roles),
                )
            )
    return out


def _make_dev_specs(per_mode: int = 8, seed: int = 11102) -> List[SingleRouterSystemSpec]:
    """Create role-holdout systems.

    Every system contains at least one never-trained role, mixed with familiar
    distractors. Each unseen role is exercised repeatedly in every execution mode.
    """
    rng = random.Random(seed)
    familiar = (
        "code", "writing", "math", "planning", "research", "data", "design", "operations",
        "qa_testing", "fact_checking", "risk_assessment", "requirements_analysis",
    )
    global_seen: set[tuple[str, tuple[str, ...]]] = set()
    counts = {m: Counter({r: 0 for r in DEV_UNSEEN_ROLES}) for m in MODES}
    out: list[SingleRouterSystemSpec] = []
    styles = ("s1", "s2", "s3")

    for mi, mode in enumerate(MODES):
        for i in range(per_mode):
            k = KS[(i + 2 * mi) % len(KS)]
            # Use three unseen roles in every development system. Across eight
            # systems per mode this yields 24 unseen-role slots per mode: enough
            # for all six held-out roles to appear repeatedly in every mode while
            # K=4..6 still include familiar distractors.
            n_unseen = 3
            unseen_counts = counts[mode]
            min_count = min(unseen_counts.values())
            candidates = [r for r in DEV_UNSEEN_ROLES if unseen_counts[r] <= min_count + 1]
            first = rng.choice(candidates)
            unseen = _choose_balanced(DEV_UNSEEN_ROLES, unseen_counts, n_unseen, rng, must_include=(first,))
            familiar_roles = tuple(rng.sample(list(familiar), k - n_unseen))
            roles = tuple(unseen) + familiar_roles
            # Ensure unique role sets across all development systems.
            for _ in range(300):
                if (mode, _composition_key(roles)) not in global_seen:
                    break
                familiar_roles = tuple(rng.sample(list(familiar), k - n_unseen))
                roles = tuple(unseen) + familiar_roles
            else:
                raise RuntimeError("could not generate unique V11 unseen-role composition")
            global_seen.add((mode, _composition_key(roles)))
            unseen_counts.update(unseen)
            out.append(
                SingleRouterSystemSpec(
                    system_id=f"v11_dev_unseen_{mode}_{i:02d}_k{k}",
                    split="v11_dev_unseen_roles",
                    execution_mode=mode,
                    style=styles[(i + mi) % len(styles)],
                    worker_roles=tuple(roles),
                )
            )
    return out


TRAIN_SPECS = tuple(_make_train_specs())
DEV_UNSEEN_SPECS = tuple(_make_dev_specs())


def get_v11_split(name: str) -> List[SingleRouterSystemSpec]:
    key = str(name).lower()
    if key in {"train", "training"}:
        return list(TRAIN_SPECS)
    if key in {"dev", "val", "validation", "unseen_roles", "dev_unseen_roles"}:
        return list(DEV_UNSEEN_SPECS)
    raise KeyError(name)


def _coverage(specs: Iterable[SingleRouterSystemSpec], roles: Sequence[str]) -> dict:
    out = {}
    ss = list(specs)
    for role in roles:
        rr = [s for s in ss if role in s.worker_roles]
        out[role] = {
            "count": len(rr),
            "modes": sorted({s.execution_mode for s in rr}),
            "k": sorted({len(s.worker_roles) for s in rr}),
        }
    return out


def validate_v11_design() -> None:
    train_keys = {_composition_key(s.worker_roles) for s in TRAIN_SPECS}
    dev_keys = {(s.execution_mode, _composition_key(s.worker_roles)) for s in DEV_UNSEEN_SPECS}
    if len(train_keys) != len(TRAIN_SPECS):
        raise AssertionError("V11 training compositions are not unique")
    if len(dev_keys) != len(DEV_UNSEEN_SPECS):
        raise AssertionError("V11 development compositions are not unique")
    # Dev systems contain at least one unseen role, so their role compositions cannot overlap training.
    dev_role_keys = {_composition_key(s.worker_roles) for s in DEV_UNSEEN_SPECS}
    if train_keys & dev_role_keys:
        raise AssertionError("V11 train/dev composition leakage")

    train_role_set = {r for s in TRAIN_SPECS for r in s.worker_roles}
    if set(DEV_UNSEEN_ROLES) & train_role_set:
        raise AssertionError("V11 unseen-role leakage into training systems")
    if not set(TRAIN_ROLES).issubset(train_role_set):
        raise AssertionError("not every training role appears in V11 training systems")

    train_cov = _coverage(TRAIN_SPECS, TRAIN_ROLES)
    for role, c in train_cov.items():
        if set(c["modes"]) != set(MODES):
            raise AssertionError(f"training role {role} lacks mode coverage: {c}")
        if len(c["k"]) < 3:
            raise AssertionError(f"training role {role} lacks K diversity: {c}")

    dev_cov = _coverage(DEV_UNSEEN_SPECS, DEV_UNSEEN_ROLES)
    for role, c in dev_cov.items():
        if set(c["modes"]) != set(MODES):
            raise AssertionError(f"unseen role {role} lacks mode coverage: {c}")
        if c["count"] < 8:
            raise AssertionError(f"unseen role {role} has too few dev systems: {c}")

    # Every runtime role must have an environment prompt after registration.
    missing = set(ALL_ROLES) - set(ROLE_PROMPTS)
    if missing:
        raise AssertionError(f"runtime missing prompts for roles: {sorted(missing)}")


validate_v11_design()
