from targets.topology_zoo import TRAIN_SPECS, VAL_SPECS, TEST_SPECS


def test_zoo_sizes_and_train_grid():
    assert len(TRAIN_SPECS) == 12
    assert len(VAL_SPECS) == 6
    assert len(TEST_SPECS) == 6
    assert {s.depth for s in TRAIN_SPECS} == {1, 2, 3}
    assert {s.k_true for s in TRAIN_SPECS} == {3, 4, 5, 6}
