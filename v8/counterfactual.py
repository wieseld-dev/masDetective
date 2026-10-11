from __future__ import annotations

import copy
import dataclasses
import hashlib
import random
from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True)
class CounterfactualPairV8:
    """Matched systems with constant K/topology and a single role swap.

    plus: target role is present.
    minus: target role is replaced by one absent distractor, preserving K and execution mode.
    sham: target remains present while another role is replaced, preserving K/mode.
    """

    pair_id: str
    target_role: str
    plus_spec: object
    minus_spec: object
    sham_spec: object | None
    negative_replacement: str
    sham_removed_role: str | None
    sham_replacement: str | None


def _stable_int(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)


def clone_spec(spec: object, *, system_id: str, worker_roles: Sequence[str]):
    """Clone the project's system spec without assuming one exact dataclass definition."""
    roles = tuple(map(str, worker_roles))
    if dataclasses.is_dataclass(spec):
        names = {f.name for f in dataclasses.fields(spec)}
        updates = {}
        if "system_id" in names:
            updates["system_id"] = system_id
        if "worker_roles" in names:
            updates["worker_roles"] = roles
        elif "roles" in names:
            updates["roles"] = roles
        for kname in ("k_true", "K", "k"):
            if kname in names:
                updates[kname] = len(roles)
        return dataclasses.replace(spec, **updates)

    if isinstance(spec, dict):
        out = dict(spec)
        out["system_id"] = system_id
        if "worker_roles" in out or "roles" not in out:
            out["worker_roles"] = roles
        else:
            out["roles"] = roles
        for kname in ("k_true", "K", "k"):
            if kname in out:
                out[kname] = len(roles)
        return out

    if hasattr(spec, "_replace"):
        fields = set(getattr(spec, "_fields", ()))
        updates = {}
        if "system_id" in fields:
            updates["system_id"] = system_id
        if "worker_roles" in fields:
            updates["worker_roles"] = roles
        elif "roles" in fields:
            updates["roles"] = roles
        for kname in ("k_true", "K", "k"):
            if kname in fields:
                updates[kname] = len(roles)
        return spec._replace(**updates)

    out = copy.copy(spec)
    setattr(out, "system_id", system_id)
    if hasattr(out, "worker_roles"):
        setattr(out, "worker_roles", roles)
    elif hasattr(out, "roles"):
        setattr(out, "roles", roles)
    else:
        raise TypeError("system spec exposes neither worker_roles nor roles")
    for kname in ("k_true", "K", "k"):
        if hasattr(out, kname):
            try:
                setattr(out, kname, len(roles))
            except Exception:
                pass
    return out


def get_roles(spec: object) -> tuple[str, ...]:
    if isinstance(spec, dict):
        return tuple(map(str, spec.get("worker_roles", spec.get("roles", ()))))
    if hasattr(spec, "worker_roles"):
        return tuple(map(str, getattr(spec, "worker_roles")))
    return tuple(map(str, getattr(spec, "roles")))


def get_system_id(spec: object) -> str:
    if isinstance(spec, dict):
        return str(spec.get("system_id", "system"))
    return str(getattr(spec, "system_id", "system"))


def get_mode(spec: object) -> str:
    if isinstance(spec, dict):
        return str(spec.get("execution_mode", spec.get("mode", "unknown")))
    return str(getattr(spec, "execution_mode", getattr(spec, "mode", "unknown")))


def get_k(spec: object) -> int:
    return len(get_roles(spec))


def make_counterfactual_pair(
    spec: object,
    target_role: str,
    candidate_roles: Sequence[str],
    *,
    include_sham: bool = True,
    salt: str = "v8",
) -> CounterfactualPairV8 | None:
    """Create a constant-K matched pair by role substitution, not deletion.

    Returns None if the target is absent or there is no candidate distractor absent from the plus system.
    """
    target_role = str(target_role)
    plus_roles = list(get_roles(spec))
    if target_role not in plus_roles:
        return None
    absent = [r for r in map(str, candidate_roles) if r not in set(plus_roles)]
    if not absent:
        return None

    base_id = get_system_id(spec)
    seed = _stable_int(f"{salt}|{base_id}|{target_role}")
    rng = random.Random(seed)
    negative_replacement = rng.choice(absent)

    minus_roles = list(plus_roles)
    minus_roles[minus_roles.index(target_role)] = negative_replacement
    pair_id = hashlib.sha256(
        f"{base_id}|{target_role}|{negative_replacement}".encode("utf-8")
    ).hexdigest()[:16]
    plus = clone_spec(spec, system_id=f"v8p_{pair_id}_plus", worker_roles=plus_roles)
    minus = clone_spec(spec, system_id=f"v8p_{pair_id}_minus", worker_roles=minus_roles)

    sham = None
    sham_removed = None
    sham_replacement = None
    if include_sham and len(plus_roles) >= 2:
        removable = [r for r in plus_roles if r != target_role]
        rng.shuffle(removable)
        if removable:
            sham_removed = removable[0]
            # Prefer a distractor different from the negative replacement when possible.
            sham_candidates = [r for r in absent if r != negative_replacement] or absent
            sham_replacement = rng.choice(sham_candidates)
            sham_roles = list(plus_roles)
            sham_roles[sham_roles.index(sham_removed)] = sham_replacement
            # Do not accidentally remove or duplicate the target.
            if target_role in sham_roles and len(set(sham_roles)) == len(sham_roles):
                sham = clone_spec(spec, system_id=f"v8p_{pair_id}_sham", worker_roles=sham_roles)

    return CounterfactualPairV8(
        pair_id=pair_id,
        target_role=target_role,
        plus_spec=plus,
        minus_spec=minus,
        sham_spec=sham,
        negative_replacement=negative_replacement,
        sham_removed_role=sham_removed,
        sham_replacement=sham_replacement,
    )


def build_counterfactual_pairs(
    specs: Iterable[object],
    candidate_roles: Sequence[str],
    *,
    include_sham: bool = True,
) -> list[CounterfactualPairV8]:
    out: list[CounterfactualPairV8] = []
    for spec in specs:
        for role in candidate_roles:
            pair = make_counterfactual_pair(spec, role, candidate_roles, include_sham=include_sham)
            if pair is not None:
                out.append(pair)
    return out


def split_base_specs(specs: Sequence[object], *, detector_fraction: float = 1 / 3, seed: int = 808) -> tuple[list[object], list[object]]:
    """Split original training systems before making counterfactual pairs.

    Stratifies coarsely by execution mode, keeping a base system wholly in one split.
    """
    rng = random.Random(seed)
    by_mode: dict[str, list[object]] = {}
    for spec in specs:
        by_mode.setdefault(get_mode(spec), []).append(spec)
    rl_train: list[object] = []
    detector_train: list[object] = []
    for mode in sorted(by_mode):
        group = sorted(by_mode[mode], key=get_system_id)
        rng.shuffle(group)
        n_det = max(1, int(round(len(group) * float(detector_fraction)))) if len(group) > 1 else 0
        n_det = min(n_det, max(len(group) - 1, 0))
        detector_train.extend(group[:n_det])
        rl_train.extend(group[n_det:])
    return rl_train, detector_train
