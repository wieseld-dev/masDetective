import numpy as np

from v10.reward import PairwiseRobustCounterfactualGraderV10


def u(x, y):
    v = np.asarray([x, y], dtype=np.float32)
    return v / np.linalg.norm(v)


def test_anchor_is_reference_not_hurdle():
    g = PairwiseRobustCounterfactualGraderV10(min_activation_rate=0.75, novelty_weight=0.0)
    plus = [u(1, 0)] * 4
    # Query has a consistent positive PRESENT/ABSENT distance on all pairs.
    minus = [u(0.94, 0.34)] * 4
    sham = [u(1, 0)] * 4
    # Anchor is deliberately even stronger; query should still be useful/positive.
    anchor_minus = [u(0.75, 0.66)] * 4
    rb = g.score(
        target_selected_flags=[True] * 4,
        plus_query_embs=plus,
        minus_query_embs=minus,
        sham_query_embs=sham,
        plus_anchor_embs=plus,
        minus_anchor_embs=anchor_minus,
        sham_anchor_embs=sham,
    )
    assert rb.strict_specific_pass
    assert not rb.strict_delta_pass
    assert rb.total > 0


def test_low_activation_cannot_be_bought_by_separation():
    g = PairwiseRobustCounterfactualGraderV10(min_activation_rate=0.75, novelty_weight=0.0)
    plus = [u(1, 0)] * 4
    minus = [u(0, 1)] * 4
    sham = [u(1, 0)] * 4
    rb = g.score(
        target_selected_flags=[True, False, False, False],
        plus_query_embs=plus,
        minus_query_embs=minus,
        sham_query_embs=sham,
        plus_anchor_embs=plus,
        minus_anchor_embs=plus,
        sham_anchor_embs=sham,
    )
    assert not rb.eligible
    assert rb.total < 0


def test_tail_failure_blocks_strict_elite():
    g = PairwiseRobustCounterfactualGraderV10(min_activation_rate=0.75, novelty_weight=0.0)
    plus = [u(1, 0)] * 4
    minus = [u(0.95, 0.31), u(0.95, 0.31), u(0.95, 0.31), u(1, 0)]
    # On the last pair SHAM is farther than ABSENT, giving a negative specific signal.
    sham = [u(1, 0), u(1, 0), u(1, 0), u(0.8, 0.6)]
    rb = g.score(
        target_selected_flags=[True] * 4,
        plus_query_embs=plus,
        minus_query_embs=minus,
        sham_query_embs=sham,
        plus_anchor_embs=plus,
        minus_anchor_embs=plus,
        sham_anchor_embs=plus,
    )
    assert rb.specific_median > 0
    assert rb.specific_q25 < 0
    assert not rb.strict_specific_pass


def test_sham_activation_is_part_of_eligibility():
    g = PairwiseRobustCounterfactualGraderV10(min_activation_rate=0.75, novelty_weight=0.0)
    plus = [u(1, 0)] * 4
    minus = [u(0.9, 0.44)] * 4
    sham = [u(1, 0)] * 4
    rb = g.score(
        target_selected_flags=[True, True, True, True],
        sham_target_selected_flags=[True, False, False, False],
        plus_query_embs=plus,
        minus_query_embs=minus,
        sham_query_embs=sham,
        plus_anchor_embs=plus,
        minus_anchor_embs=plus,
        sham_anchor_embs=plus,
    )
    assert rb.plus_activation_rate == 1.0
    assert rb.sham_activation_rate == 0.25
    assert rb.activation_rate == 0.25
    assert not rb.eligible
    assert rb.total < 0
