from __future__ import annotations

from typing import Dict, List, Tuple

from targets.single_router_fanout_mas import SingleRouterSystemSpec, VALID_MODES


# Same overcomplete capability vocabulary used by the Detective.
R = ("code", "writing", "math", "planning", "research", "data", "design", "operations")

# Training compositions are deliberately reused across execution modes. This prevents the role
# estimator from confusing "mode" with "which roles exist" and forces it to learn output behavior.
TRAIN_COMPOSITIONS: Dict[int, Tuple[str, ...]] = {
    3: ("code", "math", "research"),
    4: ("writing", "planning", "data", "design"),
    5: ("code", "writing", "research", "design", "operations"),
    6: ("math", "planning", "research", "data", "design", "operations"),
}
VAL_COMPOSITIONS: Dict[int, Tuple[str, ...]] = {
    3: ("writing", "data", "operations"),
    5: ("code", "math", "planning", "data", "operations"),
}
TEST_COMPOSITIONS: Dict[int, Tuple[str, ...]] = {
    4: ("code", "research", "design", "operations"),
    6: ("code", "writing", "math", "planning", "data", "operations"),
}


def _make(split: str, comps: Dict[int, Tuple[str, ...]]) -> List[SingleRouterSystemSpec]:
    specs: List[SingleRouterSystemSpec] = []
    styles = ("s1", "s2", "s3")
    # Interleave K within each mode, then callers can use the full balanced set.
    for mi, mode in enumerate(VALID_MODES):
        for ki, (k, roles) in enumerate(sorted(comps.items())):
            specs.append(
                SingleRouterSystemSpec(
                    system_id=f"{split}_sr_{mode}_k{k}",
                    split=split,
                    execution_mode=mode,
                    style=styles[(mi + ki) % len(styles)],
                    worker_roles=roles,
                )
            )
    return specs


TRAIN_SPECS = _make("train", TRAIN_COMPOSITIONS)  # 4 modes x 4 K values = 16
VAL_SPECS = _make("val", VAL_COMPOSITIONS)        # 4 modes x 2 K values = 8
TEST_SPECS = _make("test", TEST_COMPOSITIONS)     # 4 modes x 2 K values = 8
ALL_SPECS = TRAIN_SPECS + VAL_SPECS + TEST_SPECS
BY_ID = {s.system_id: s for s in ALL_SPECS}


def get_split(name: str) -> List[SingleRouterSystemSpec]:
    key = name.lower()
    if key == "train":
        return list(TRAIN_SPECS)
    if key in ("val", "validation"):
        return list(VAL_SPECS)
    if key == "test":
        return list(TEST_SPECS)
    raise KeyError(f"unknown split: {name}")


def get_spec(system_id: str) -> SingleRouterSystemSpec:
    return BY_ID[system_id]
