from targets.single_router_zoo_v65 import TRAIN_NEW_SPECS, VAL_SPECS, composition_key


def test_v65_compositions_are_unique_and_disjoint():
    tr = [composition_key(s.worker_roles) for s in TRAIN_NEW_SPECS]
    va = [composition_key(s.worker_roles) for s in VAL_SPECS]
    assert len(tr) == len(set(tr)) == 24
    assert len(va) == len(set(va)) == 8
    assert not (set(tr) & set(va))


def test_v65_train_is_balanced_over_modes_and_k():
    assert {m: sum(s.execution_mode == m for s in TRAIN_NEW_SPECS) for m in {s.execution_mode for s in TRAIN_NEW_SPECS}} == {
        "single": 6, "fanout2": 6, "fanout3": 6, "mixed": 6
    }
    assert {k: sum(s.k_true == k for s in TRAIN_NEW_SPECS) for k in {s.k_true for s in TRAIN_NEW_SPECS}} == {
        3: 6, 4: 6, 5: 6, 6: 6
    }


def test_v65_role_prevalence_is_balanced():
    roles = sorted({r for s in TRAIN_NEW_SPECS for r in s.worker_roles})
    counts = {r: sum(r in s.worker_roles for s in TRAIN_NEW_SPECS) for r in roles}
    assert max(counts.values()) - min(counts.values()) <= 1
