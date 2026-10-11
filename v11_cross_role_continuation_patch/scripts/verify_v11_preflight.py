from __future__ import annotations

from collections import Counter

from targets.single_router_zoo_v11 import DEV_UNSEEN_SPECS, TRAIN_SPECS
from v11.hashcheck import assert_exact_v10_start
from v11.role_universe import ALL_ROLES, DEV_UNSEEN_ROLES, TRAIN_ROLES
from v8.counterfactual import build_counterfactual_pairs, get_k, get_mode


def main() -> None:
    fp = assert_exact_v10_start("artifacts/v10_policy", "artifacts/v10_summary.json", strict=True)
    train_pairs = build_counterfactual_pairs(TRAIN_SPECS, TRAIN_ROLES, include_sham=True)
    dev_pairs = [
        p for p in build_counterfactual_pairs(DEV_UNSEEN_SPECS, ALL_ROLES, include_sham=True)
        if p.target_role in set(DEV_UNSEEN_ROLES)
    ]
    print("=== V11 PREFLIGHT ===")
    print("EXACT V10.1 START: PASS")
    print("policy sha256:", fp["policy_sha256"])
    print("train roles:", len(TRAIN_ROLES), list(TRAIN_ROLES))
    print("unseen dev roles:", len(DEV_UNSEEN_ROLES), list(DEV_UNSEEN_ROLES))
    print("train base systems:", len(TRAIN_SPECS))
    print("unseen-role dev systems:", len(DEV_UNSEEN_SPECS))
    print("train matched pairs:", len(train_pairs))
    print("dev matched pairs:", len(dev_pairs))
    print("train modes:", dict(Counter(get_mode(p.plus_spec) for p in train_pairs)))
    print("train K:", dict(Counter(get_k(p.plus_spec) for p in train_pairs)))
    print("dev modes:", dict(Counter(get_mode(p.plus_spec) for p in dev_pairs)))
    print("dev K:", dict(Counter(get_k(p.plus_spec) for p in dev_pairs)))
    train_runtime = {r for s in TRAIN_SPECS for r in s.worker_roles}
    assert not (set(DEV_UNSEEN_ROLES) & train_runtime)
    assert not (set(DEV_UNSEEN_ROLES) & {p.negative_replacement for p in train_pairs})
    print("ROLE-TYPE LEAKAGE CHECK: PASS")
    print("V10 artifacts are read-only inputs; V11 outputs use artifacts/v11_* paths.")


if __name__ == "__main__":
    main()
