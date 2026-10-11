from __future__ import annotations

from typing import List, Tuple

from targets.single_router_fanout_mas import SingleRouterSystemSpec

# V6.5 deliberately uses MANY composition-unique systems. The old V6.3 train set
# had only four unique role compositions repeated across four execution modes,
# which made strict composition generalization impossible to assess reliably.

TRAIN_NEW: Tuple[Tuple[str, int, Tuple[str, ...]], ...] = (
    ("single", 3, ("code", "writing", "data")),
    ("single", 3, ("math", "design", "operations")),
    ("single", 4, ("code", "math", "design", "operations")),
    ("single", 4, ("code", "writing", "research", "design")),
    ("single", 5, ("code", "planning", "research", "data", "operations")),
    ("single", 6, ("writing", "math", "planning", "research", "data", "design")),

    ("fanout2", 3, ("code", "planning", "data")),
    ("fanout2", 4, ("writing", "math", "planning", "design")),
    ("fanout2", 4, ("writing", "math", "research", "operations")),
    ("fanout2", 5, ("code", "writing", "math", "planning", "research")),
    ("fanout2", 5, ("writing", "math", "research", "data", "operations")),
    ("fanout2", 6, ("writing", "planning", "research", "data", "design", "operations")),

    ("fanout3", 3, ("planning", "data", "design")),
    ("fanout3", 4, ("code", "math", "research", "operations")),
    ("fanout3", 5, ("code", "writing", "math", "research", "operations")),
    ("fanout3", 5, ("writing", "planning", "research", "design", "operations")),
    ("fanout3", 6, ("code", "math", "planning", "data", "design", "operations")),
    ("fanout3", 6, ("code", "planning", "research", "data", "design", "operations")),

    ("mixed", 3, ("planning", "data", "operations")),
    ("mixed", 3, ("writing", "data", "operations")),
    ("mixed", 4, ("math", "planning", "research", "data")),
    ("mixed", 5, ("code", "writing", "planning", "design", "operations")),
    ("mixed", 6, ("code", "writing", "math", "planning", "data", "design")),
    ("mixed", 6, ("code", "writing", "math", "research", "data", "design")),
)

# Completely unseen compositions for validation. None occurs in TRAIN_NEW or in
# the four V6.3 training compositions.
VALIDATION: Tuple[Tuple[str, int, Tuple[str, ...]], ...] = (
    ("single", 3, ("writing", "math", "design")),
    ("single", 4, ("code", "writing", "data", "design")),
    ("fanout2", 4, ("writing", "planning", "design", "operations")),
    ("fanout2", 5, ("planning", "research", "data", "design", "operations")),
    ("fanout3", 5, ("code", "writing", "math", "planning", "operations")),
    ("fanout3", 6, ("code", "math", "planning", "research", "data", "operations")),
    ("mixed", 6, ("writing", "math", "planning", "research", "data", "operations")),
    ("mixed", 3, ("code", "research", "design")),
)

V63_TRAIN_COMPOSITIONS = {
    tuple(sorted(("code", "math", "research"))),
    tuple(sorted(("writing", "planning", "data", "design"))),
    tuple(sorted(("code", "writing", "research", "design", "operations"))),
    tuple(sorted(("math", "planning", "research", "data", "design", "operations"))),
}


def _specs(split: str, rows) -> List[SingleRouterSystemSpec]:
    styles = ("s1", "s2", "s3")
    out: List[SingleRouterSystemSpec] = []
    for i, (mode, k, roles) in enumerate(rows):
        if len(roles) != k:
            raise ValueError(f"declared K={k}, actual roles={len(roles)}")
        out.append(
            SingleRouterSystemSpec(
                system_id=f"v65_{split}_{i:02d}_{mode}_k{k}",
                split=split,
                execution_mode=mode,
                style=styles[i % len(styles)],
                worker_roles=tuple(roles),
            )
        )
    return out


TRAIN_NEW_SPECS = _specs("train", TRAIN_NEW)
VAL_SPECS = _specs("val", VALIDATION)


def get_v65_split(name: str) -> List[SingleRouterSystemSpec]:
    key = name.lower()
    if key in {"train", "train_new"}:
        return list(TRAIN_NEW_SPECS)
    if key in {"val", "validation"}:
        return list(VAL_SPECS)
    raise KeyError(name)


def composition_key(roles) -> str:
    return "|".join(sorted(map(str, roles)))


def validate_design() -> None:
    train_keys = {composition_key(s.worker_roles) for s in TRAIN_NEW_SPECS}
    val_keys = {composition_key(s.worker_roles) for s in VAL_SPECS}
    old_keys = {"|".join(c) for c in V63_TRAIN_COMPOSITIONS}
    if len(train_keys) != len(TRAIN_NEW_SPECS):
        raise AssertionError("V6.5 train compositions must be unique")
    if len(val_keys) != len(VAL_SPECS):
        raise AssertionError("V6.5 validation compositions must be unique")
    if train_keys & val_keys:
        raise AssertionError("train/validation composition leakage")
    if val_keys & old_keys:
        raise AssertionError("V6.5 validation overlaps V6.3 training compositions")


validate_design()
