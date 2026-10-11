from __future__ import annotations

import hashlib
import random
from collections import Counter
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from targets.single_router_fanout_mas import SingleRouterSystemSpec
from v8.counterfactual import CounterfactualPairV8, clone_spec, get_roles
from v11.role_universe import ROLE_DESCRIPTIONS, TRAIN_ROLES
from targets.single_router_zoo_v11 import get_v11_split


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float32).reshape(-1)
    bb = np.asarray(b, dtype=np.float32).reshape(-1)
    return float(np.dot(aa, bb) / max(float(np.linalg.norm(aa) * np.linalg.norm(bb)), 1e-8))


def build_neighbor_map(embedder, roles: Sequence[str], top_k: int = 5) -> dict[str, list[str]]:
    """Automatic hard-neighbor graph from PUBLIC role descriptions only."""
    roles = tuple(map(str, roles))
    vec = {r: embedder.encode(ROLE_DESCRIPTIONS.get(r, r)) for r in roles}
    out: dict[str, list[str]] = {}
    for r in roles:
        ranked = sorted(
            (x for x in roles if x != r),
            key=lambda x: (_cos(vec[r], vec[x]), x),
            reverse=True,
        )
        out[r] = ranked[: max(1, int(top_k))]
    return out


def composition_key(roles: Sequence[str]) -> tuple[str, ...]:
    return tuple(sorted(map(str, roles)))


def make_fresh_fanout_specs(
    roles: Sequence[str],
    *,
    prefix: str,
    systems_per_cell: int,
    seed: int,
    excluded_compositions: set[tuple[str, ...]] | None = None,
) -> list[SingleRouterSystemSpec]:
    """Balanced fresh FANOUT2/FANOUT3 systems, K=3..6, composition-disjoint."""
    rng = random.Random(seed)
    excluded = set(excluded_compositions or set())
    # Always exclude V11 train/dev compositions.
    for s in list(get_v11_split("train")) + list(get_v11_split("dev_unseen_roles")):
        excluded.add(composition_key(s.worker_roles))
    used = set(excluded)
    counts = Counter({r: 0 for r in roles})
    out: list[SingleRouterSystemSpec] = []
    styles = ("s1", "s2", "s3")
    idx = 0
    for mode in ("fanout2", "fanout3"):
        for k in (3, 4, 5, 6):
            for rep in range(int(systems_per_cell)):
                chosen = None
                for _ in range(10000):
                    ordered = sorted(roles, key=lambda r: (counts[r], rng.random()))
                    pool = ordered[: max(k + 5, len(roles) // 2)]
                    trial = tuple(rng.sample(pool, k))
                    key = composition_key(trial)
                    if key not in used:
                        chosen = trial
                        break
                if chosen is None:
                    raise RuntimeError("could not construct fresh V13 FANOUT composition")
                used.add(composition_key(chosen))
                counts.update(chosen)
                out.append(
                    SingleRouterSystemSpec(
                        system_id=f"{prefix}_{mode}_k{k}_r{rep:02d}",
                        split=prefix,
                        execution_mode=mode,
                        style=styles[idx % len(styles)],
                        worker_roles=tuple(chosen),
                    )
                )
                idx += 1
    return out


def make_hard_pair(
    spec: SingleRouterSystemSpec,
    target_role: str,
    roles: Sequence[str],
    neighbor_map: Mapping[str, Sequence[str]],
    *,
    salt: str = "v13",
) -> CounterfactualPairV8 | None:
    """Matched PRESENT/ABSENT/SHAM pair where ABSENT swaps target for a hard neighbor."""
    target_role = str(target_role)
    plus_roles = list(get_roles(spec))
    if target_role not in plus_roles:
        return None
    present = set(plus_roles)
    absent = [r for r in roles if r not in present]
    if not absent:
        return None

    hard = [r for r in neighbor_map.get(target_role, ()) if r in absent]
    negative = hard[0] if hard else absent[0]
    base_id = spec.system_id
    pid = hashlib.sha256(f"{salt}|{base_id}|{target_role}|{negative}".encode()).hexdigest()[:16]

    minus_roles = list(plus_roles)
    minus_roles[minus_roles.index(target_role)] = negative
    plus = clone_spec(spec, system_id=f"v13p_{pid}_plus", worker_roles=plus_roles)
    minus = clone_spec(spec, system_id=f"v13p_{pid}_minus", worker_roles=minus_roles)

    # SHAM keeps target, but swaps a different role. Prefer a hard neighbor of the
    # removed role so unrelated composition drift is itself challenging.
    rng = random.Random(int(pid, 16))
    removable = [r for r in plus_roles if r != target_role]
    rng.shuffle(removable)
    sham = None
    sham_removed = None
    sham_replacement = None
    for removed in removable:
        candidates = [r for r in neighbor_map.get(removed, ()) if r not in present and r != negative]
        candidates += [r for r in absent if r not in candidates and r != negative]
        if not candidates:
            continue
        repl = candidates[0]
        sham_roles = list(plus_roles)
        sham_roles[sham_roles.index(removed)] = repl
        if len(set(sham_roles)) != len(sham_roles) or target_role not in sham_roles:
            continue
        sham = clone_spec(spec, system_id=f"v13p_{pid}_sham", worker_roles=sham_roles)
        sham_removed = removed
        sham_replacement = repl
        break

    return CounterfactualPairV8(
        pair_id=pid,
        target_role=target_role,
        plus_spec=plus,
        minus_spec=minus,
        sham_spec=sham,
        negative_replacement=negative,
        sham_removed_role=sham_removed,
        sham_replacement=sham_replacement,
    )


def build_hard_pairs(
    specs: Sequence[SingleRouterSystemSpec],
    roles: Sequence[str],
    neighbor_map: Mapping[str, Sequence[str]],
) -> list[CounterfactualPairV8]:
    out: list[CounterfactualPairV8] = []
    for s in specs:
        for r in roles:
            p = make_hard_pair(s, r, roles, neighbor_map)
            if p is not None and p.sham_spec is not None:
                out.append(p)
    return out


def excluded_v12_compositions(roles: Sequence[str], seed: int = 12012) -> set[tuple[str, ...]]:
    """Recreate the already-inspected V12 development grid so V13 does not train on it."""
    try:
        from analysis.adaptive_fanout_count_v12 import make_fresh_fanout_specs as make_v12
        return {composition_key(s.worker_roles) for s in make_v12(tuple(roles), systems_per_cell=1, seed=seed)}
    except Exception:
        return set()
