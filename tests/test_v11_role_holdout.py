from __future__ import annotations

from collections import Counter

from targets.single_router_zoo_v11 import DEV_UNSEEN_SPECS, TRAIN_SPECS, MODES
from v11.role_universe import ALL_ROLES, DEV_UNSEEN_ROLES, TRAIN_ROLES, ROLE_DESCRIPTIONS
from v8.counterfactual import build_counterfactual_pairs, get_roles


def test_train_and_dev_role_types_are_disjoint():
    assert set(TRAIN_ROLES).isdisjoint(DEV_UNSEEN_ROLES)
    train_runtime_roles = {r for s in TRAIN_SPECS for r in s.worker_roles}
    assert set(DEV_UNSEEN_ROLES).isdisjoint(train_runtime_roles)


def test_unseen_roles_cannot_enter_training_counterfactuals():
    pairs = build_counterfactual_pairs(TRAIN_SPECS, TRAIN_ROLES, include_sham=True)
    unseen = set(DEV_UNSEEN_ROLES)
    assert pairs
    for p in pairs:
        assert p.target_role not in unseen
        assert p.negative_replacement not in unseen
        assert not (unseen & set(get_roles(p.plus_spec)))
        assert not (unseen & set(get_roles(p.minus_spec)))
        if p.sham_spec is not None:
            assert not (unseen & set(get_roles(p.sham_spec)))


def test_every_train_role_has_broad_mode_and_k_coverage():
    for role in TRAIN_ROLES:
        ss = [s for s in TRAIN_SPECS if role in s.worker_roles]
        assert {s.execution_mode for s in ss} == set(MODES)
        assert len({len(s.worker_roles) for s in ss}) >= 3


def test_every_unseen_dev_role_has_all_modes_and_many_pairs():
    dev_pairs = [
        p for p in build_counterfactual_pairs(DEV_UNSEEN_SPECS, ALL_ROLES, include_sham=True)
        if p.target_role in set(DEV_UNSEEN_ROLES)
    ]
    for role in DEV_UNSEEN_ROLES:
        rr = [p for p in dev_pairs if p.target_role == role]
        assert len(rr) >= 8
        assert {p.plus_spec.execution_mode for p in rr} == set(MODES)
        assert {len(p.plus_spec.worker_roles) for p in rr} == {3, 4, 5, 6}


def test_role_descriptions_exist_for_every_role():
    assert set(ALL_ROLES) <= set(ROLE_DESCRIPTIONS)
