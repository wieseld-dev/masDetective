from targets.single_router_zoo import TRAIN_SPECS, VAL_SPECS, TEST_SPECS


def test_v63_zoo_is_one_router_layer_and_balanced_modes():
    assert len(TRAIN_SPECS) == 16
    assert len(VAL_SPECS) == 8
    assert len(TEST_SPECS) == 8
    assert {s.depth for s in TRAIN_SPECS} == {1}
    assert {s.execution_mode for s in TRAIN_SPECS} == {"single", "fanout2", "fanout3", "mixed"}
    assert {s.k_true for s in TRAIN_SPECS} == {3, 4, 5, 6}
    for mode in {s.execution_mode for s in TRAIN_SPECS}:
        assert {s.k_true for s in TRAIN_SPECS if s.execution_mode == mode} == {3,4,5,6}
