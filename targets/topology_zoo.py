from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

from targets.hierarchical_mas import MASSystemSpec


# Eight roles allow held-out compositions without changing the target K definition.
R = ("code", "writing", "math", "planning", "research", "data", "design", "operations")


def _roles(offset: int, k: int) -> Tuple[str, ...]:
    return tuple(R[(offset + i) % len(R)] for i in range(k))


def _make_split(split: str, ks: Sequence[int], offsets: Sequence[int]) -> List[MASSystemSpec]:
    specs: List[MASSystemSpec] = []
    families = (("flat", 1), ("hier2", 2), ("hier3", 3))
    for fi, (family, depth) in enumerate(families):
        for j, k in enumerate(ks):
            offset = offsets[(fi * len(ks) + j) % len(offsets)]
            style = ("s1", "s2", "s3")[(fi + j) % 3]
            specs.append(
                MASSystemSpec(
                    system_id=f"{split}_{family}_k{k}_{j}",
                    split=split,
                    family=family,
                    depth=depth,
                    style=style,
                    worker_roles=_roles(offset, k),
                )
            )
    return specs


# 12 training systems: 3 families x K in {3,4,5,6}.
TRAIN_SPECS = _make_split("train", (3, 4, 5, 6), (0, 2, 4, 1, 5, 3))
# 6 validation systems: held-out compositions/styles, two K values per family.
VAL_SPECS = _make_split("val", (3, 5), (6, 1, 7, 2, 5, 0))
# 6 test systems: held-out compositions/styles, different K values per family.
TEST_SPECS = _make_split("test", (4, 6), (3, 7, 1, 6, 2, 5))

ALL_SPECS = TRAIN_SPECS + VAL_SPECS + TEST_SPECS
BY_ID: Dict[str, MASSystemSpec] = {s.system_id: s for s in ALL_SPECS}


def get_split(name: str) -> List[MASSystemSpec]:
    key = name.lower()
    if key == "train":
        return list(TRAIN_SPECS)
    if key == "val":
        return list(VAL_SPECS)
    if key == "test":
        return list(TEST_SPECS)
    raise KeyError(f"unknown split: {name}")


def get_spec(system_id: str) -> MASSystemSpec:
    return BY_ID[system_id]
